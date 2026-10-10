"""
models/quantum_ml.py - Variational Quantum Classifier (PennyLane) for ANY feature dimension.

Key design points
-----------------
* Positive class (+1 target) = label 1.  Everything upstream maps the minority class to 1.
* Encoding: feature k is applied to qubit (k % n_qubits) in "round" (k // n_qubits).
  Even rounds use RX(x), odd rounds use RY(x).
    - n_features <= max_qubits : one feature per qubit, pure RX angle encoding
      (identical to the original 6-qubit design).
    - n_features  > max_qubits : several features per qubit (e.g. d=48 on 12 qubits = 4 per qubit)
      so large PCA dimensions stay simulable. A d-qubit statevector costs O(2^d) memory.
* Ansatz: qml.StronglyEntanglingLayers(n_layers) -> 3 * L * n_qubits trainable angles.
* Readout: <Z_0> + b ; predicted positive when score >= threshold (default 0.0).
* Loss: plain MSE against targets {-1,+1}. Train on a class-balanced set (no extra class
  weights, otherwise balance is compensated twice).
* The (non-trainable) encoding layer is computed exactly in NumPy as a batched product state and
  loaded with StatePrep. This is mathematically identical to RX/RY gates (verified to 1e-16) and
  avoids a PennyLane autograd batching bug seen at >= ~10 qubits. Only the ansatz weights and
  the bias are differentiated.
* Batch size is auto-reduced to fit a memory budget, so large qubit counts do not blow up RAM.
"""
import math
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


_StatePrep = getattr(qml, "StatePrep", None) or getattr(qml, "QubitStateVector")


def product_state(X, n_qubits):
    """Batched product state after the angle-encoding layer. X: (B, n_features) -> (B, 2**n_qubits).
    Feature k acts on qubit k % n_qubits in round k // n_qubits (even round RX, odd round RY).
    Wire 0 is the most-significant bit (PennyLane convention)."""
    X = np.asarray(X, dtype=float)
    B, nf = X.shape
    rounds = math.ceil(nf / n_qubits)
    vecs = []
    for q in range(n_qubits):
        v = np.zeros((B, 2), dtype=complex)
        v[:, 0] = 1.0
        for r in range(rounds):
            k = r * n_qubits + q
            if k >= nf:
                break
            a = X[:, k] / 2.0
            c, s_ = np.cos(a), np.sin(a)
            if r % 2 == 0:  # RX
                v0 = c * v[:, 0] - 1j * s_ * v[:, 1]
                v1 = -1j * s_ * v[:, 0] + c * v[:, 1]
            else:           # RY
                v0 = c * v[:, 0] - s_ * v[:, 1]
                v1 = s_ * v[:, 0] + c * v[:, 1]
            v = np.stack([v0, v1], axis=1)
        vecs.append(v)
    st = vecs[0]
    for q in range(1, n_qubits):
        st = (st[:, :, None] * vecs[q][:, None, :]).reshape(B, -1)
    return st


class QuantumClassifierModel:
    def __init__(self, n_features=None, n_qubits=None, n_layers=3, lr=0.1, epochs=50,
                 batch_size=32, max_qubits=12, seed=0, init="uniform",
                 mem_budget_gb=1.0, verbose=True, allow_large=False):
        if n_features is None and n_qubits is None:
            n_features = 4
        if n_features is None:
            n_features = n_qubits
        if n_qubits is None:
            n_qubits = min(n_features, max_qubits)
        if n_qubits > 20 and not allow_large:
            raise ValueError(
                f"{n_qubits} qubits needs a 2^{n_qubits}-amplitude statevector per sample. "
                "Lower --max-qubits (features are then packed several per qubit) "
                "or pass --allow-large if you really have the RAM/time.")
        self.n_features = int(n_features)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.init = init
        self.mem_budget_gb = mem_budget_gb
        self.verbose = verbose
        self.features_per_qubit = math.ceil(self.n_features / self.n_qubits)
        self.n_params = 3 * self.n_layers * self.n_qubits + 1  # + bias
        self.threshold = 0.0
        self.history = {"loss": []}
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_circuit()
        self._init_params()

    # ------------------------------------------------------------------ circuit
    def _build_circuit(self):
        nq = self.n_qubits

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(weights, state):
            _StatePrep(state, wires=range(nq))
            qml.StronglyEntanglingLayers(weights, wires=range(nq))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        n_ops = 2 * nq * self.n_layers + 1
        state_bytes = 16 * (2 ** nq)
        budget = self.mem_budget_gb * 1e9
        self._bs_max = max(1, int(budget // (3 * n_ops * state_bytes)))
        self.eff_batch = max(1, min(self.batch_size, self._bs_max))

    def _states(self, X):
        return pnp.array(product_state(X, self.n_qubits), requires_grad=False)

    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        if self.init == "small":
            w0 = rng.normal(0.0, 0.3, size=shape)
        else:
            w0 = rng.uniform(0.0, 2 * np.pi, size=shape)
        self.weights = pnp.array(w0, requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)

    # ------------------------------------------------------------------ training
    def _loss(self, w, b, sb, tb):
        pred = self.circuit(w, sb) + b
        return pnp.mean((pred - tb) ** 2)

    def fit(self, X, y01):
        """X: (N, n_features) angles in [0, pi]; y01: labels in {0,1} (1 = positive class)."""
        X = np.asarray(X, dtype=float)
        t = 2.0 * np.asarray(y01, dtype=float) - 1.0
        rng = np.random.default_rng(self.seed + 1)
        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.history = {"loss": []}
        n = len(X)
        bs = self.eff_batch
        if self.verbose:
            note = "" if bs == self.batch_size else f" (reduced from {self.batch_size} by memory budget)"
            print(f"   [QML] {self.n_features} features -> {self.n_qubits} qubits "
                  f"({self.features_per_qubit}/qubit), {self.n_params - 1}+1 params, "
                  f"batch={bs}{note}, epochs={self.epochs}")
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            losses = []
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                sb = self._states(X[idx])
                tb = pnp.array(t[idx], requires_grad=False)
                (self.weights, self.bias), loss = opt.step_and_cost(
                    lambda w, b: self._loss(w, b, sb, tb), self.weights, self.bias)
                losses.append(float(loss) * len(idx))
            ep_loss = sum(losses) / n
            self.history["loss"].append(ep_loss)
            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == self.epochs):
                print(f"   [QML] epoch {epoch + 1:03d}/{self.epochs:03d} | loss {ep_loss:.4f}")
        return self

    # ------------------------------------------------------------------ inference
    def decision_function(self, X):
        """Returns <Z_0> + b for each row (higher = more likely positive class)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = []
        for i in range(0, len(X), self.eff_batch):
            sb = self._states(X[i:i + self.eff_batch])
            out.append(np.asarray(self.circuit(self.weights, sb)).reshape(-1))
        return np.concatenate(out) + float(self.bias)

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)

    def calibrate_threshold(self, X, y01):
        """Pick the threshold maximizing balanced accuracy on (X, y01). Use TRAIN data only."""
        s = self.decision_function(X)
        y = np.asarray(y01).astype(int)
        cands = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
        mids = np.concatenate([[cands[0] - 1e-6], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
        best_thr, best_ba = 0.0, -1.0
        for thr in mids:
            p = (s >= thr).astype(int)
            tpr = (p[y == 1] == 1).mean() if (y == 1).any() else 0.0
            tnr = (p[y == 0] == 0).mean() if (y == 0).any() else 0.0
            ba = 0.5 * (tpr + tnr)
            if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and abs(thr) < abs(best_thr)):
                best_thr, best_ba = float(thr), ba
        self.threshold = best_thr
        return best_thr

    # ---- backwards-compatible names used by older scripts
    def fit_preprocessed(self, X, y):
        return self.fit(X, y)

    def predict_preprocessed(self, X):
        return self.predict(X)


QuantumClassifier = QuantumClassifierModel  # alias for older imports
