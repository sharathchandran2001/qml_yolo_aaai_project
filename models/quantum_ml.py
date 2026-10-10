import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

class QuantumClassifierModel:
    def __init__(self, n_features, n_layers=2, lr=0.01, epochs=50, 
                 batch_size=16, max_qubits=10, seed=None, init="random", 
                 mem_budget_gb=4.0, verbose=False, allow_large=False, **kwargs):
        
        self.n_features = n_features
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        
        # Determine number of qubits based on your encoding strategy
        self.n_qubits = n_features if max_qubits is None else min(n_features, max_qubits)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            pnp.random.seed(self.seed)
        
        # Initialize the PennyLane device BEFORE defining the QNode
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define QNode circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # 1. Angle Encoding via RX rotations
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)

            # 2. Parametrized Strongly Entangling Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # 3. Measurement: Pauli-Z expectation value on Qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Initialize the trainable parameters and tracking history
        self.weights = self._init_weights()
        self.bias = pnp.array(0.0, requires_grad=True)
        self.history = {"loss": []}
        
        # Metadata properties expected by the reporting script
        self.n_params = 3 * self.n_layers * self.n_qubits + 1
        self.features_per_qubit = 1
        self.threshold = 0.0

    def _init_weights(self):
        """Initializes trainable circuit weights for StronglyEntanglingLayers."""
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        return pnp.random.random(size=shape, requires_grad=True)

    def _cost(self, weights, bias, X, y_targets):
        """Mean Squared Error cost function on target expectation values."""
        predictions = [self.circuit(weights, x) + bias for x in X]
        return pnp.mean((pnp.array(predictions) - y_targets) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the VQC directly on pre-transformed features.
        
        Args:
            X: Numpy array of shape (N, n_features).
            y: Target binary class labels in {0, 1}.
        """
        # Map binary class labels {0, 1} to Pauli-Z target expectation domain {-1, +1}
        y_targets = pnp.array([1.0 if label == 1 else -1.0 for label in y], requires_grad=False)
        X_data = pnp.array(X, requires_grad=False)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.history["loss"] = []

        print(f"   [QML] Training {self.n_qubits}-Qubit VQC over {self.epochs} epochs...")
        for epoch in range(self.epochs):
            (self.weights, self.bias), loss = opt.step_and_cost(
                lambda w, b: self._cost(w, b, X_data, y_targets),
                self.weights,
                self.bias
            )
            loss_val = float(loss)
            self.history["loss"].append(loss_val)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   [QML] Epoch {epoch + 1:02d}/{self.epochs:02d} | Loss: {loss_val:.4f}")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns continuous expectation values (<Z_0> + bias) for evaluation metrics."""
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        scores = []
        for x in X:
            exp_val = self.circuit(self.weights, x) + self.bias
            scores.append(float(exp_val))
        return np.array(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels directly from pre-transformed features.
        
        Args:
            X: Numpy array of shape (N, n_features) or single sample vector.
            
        Returns:
            Numpy array of predicted binary labels {0, 1}.
        """
        scores = self.decision_function(X)
        return (scores >= self.threshold).astype(int)