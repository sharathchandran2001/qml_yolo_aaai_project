import pennylane as qml
from pennylane import numpy as pnp
from sklearn.decomposition import PCA

class QuantumClassifierModel:
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pca = PCA(n_components=n_qubits)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Initialize trainable variational weights
        pnp.random.seed(42)
        raw_weights = 0.01 * pnp.random.randn(n_layers, n_qubits, 3)
        self.weights = pnp.array(raw_weights, requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)

        # Define QNode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # 1. Feature Map: Angle Encoding (RX rotations)
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            
            # 2. Variational Entangling Layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # 3. Measurement (Expectation value of PauliZ on qubit 0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def fit_pca(self, X_train):
        """Fit PCA on high-dimensional crop feature matrix (1024-D to 4-D)."""
        X_reduced = self.pca.fit_transform(X_train)
        # Normalize features into [-pi, pi] for quantum angle encoding
        X_norm = pnp.pi * (X_reduced / pnp.max(pnp.abs(X_reduced)))
        return X_norm

    def transform_pca(self, X):
        X_reduced = self.pca.transform(X)
        return pnp.pi * (X_reduced / pnp.max(pnp.abs(X_reduced)))

    def cost_fn(self, weights, bias, X_batch, y_batch):
        """Square loss function for variational quantum optimization."""
        predictions = [self.circuit(weights, x) + bias for x in X_batch]
        # Map binary target labels {0, 1} to {-1, 1} for PauliZ expectation values
        y_mapped = pnp.where(y_batch == 0, -1, 1)
        return pnp.mean((y_mapped - pnp.array(predictions)) ** 2)

    def train(self, X_train, y_train, epochs=15, lr=0.1):
        """Train variational weights using PennyLane Adam Optimizer."""
        X_q = self.fit_pca(X_train)
        opt = qml.AdamOptimizer(stepsize=lr)
        
        loss_history = []
        print(f"[+] Training Quantum VQC ({self.n_qubits} Qubits, {self.n_layers} Layers)...")
        
        for epoch in range(epochs):
            # Unpack nested tuple: ((new_weights, new_bias), loss)
            (self.weights, self.bias), loss = opt.step_and_cost(
                lambda w, b: self.cost_fn(w, b, X_q, y_train),
                self.weights, self.bias
            )
            loss_history.append(float(loss))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:02d}/{epochs} | Loss: {loss:.4f}")
                
        return loss_history

    def predict(self, X_test):
        X_q = self.transform_pca(X_test)
        preds_raw = [self.circuit(self.weights, x) + self.bias for x in X_q]
        # Threshold PauliZ expectation predictions into classes {0, 1}
        return pnp.where(pnp.array(preds_raw) >= 0, 1, 0)