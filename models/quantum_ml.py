import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

class QuantumClassifierModel:
    """
    Variational Quantum Classifier (VQC) implemented using PennyLane.
    Accepts 4-dimensional angularly encoded feature vectors.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, lr: float = 0.1, epochs: int = 20):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs

        # Device setup
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = self._init_weights()
        self.bias = pnp.array(0.0, requires_grad=True)

        # Define QNode circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # 1. Angle Encoding via RX rotations across 4 qubits
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)

            # 2. Parametrized Strongly Entangling Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # 3. Measurement: Pauli-Z expectation value on Qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def _init_weights(self):
        """Initializes trainable circuit weights for StronglyEntanglingLayers."""
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        return pnp.random.random(size=shape, requires_grad=True)

    def _cost(self, weights, bias, X, y_targets):
        """Mean Squared Error cost function on target expectation values."""
        predictions = [self.circuit(weights, x) + bias for x in X]
        return pnp.mean((pnp.array(predictions) - y_targets) ** 2)

    def fit_preprocessed(self, X_4d: np.ndarray, y: np.ndarray):
        """
        Trains the VQC directly on pre-transformed 4-D features.
        
        Args:
            X_4d: Numpy array of shape (N, 4) with pre-scaled PCA features.
            y: Target binary class labels in {0, 1}.
        """
        # Map binary class labels {0, 1} to Pauli-Z target expectation domain {-1, +1}
        y_targets = pnp.array([1.0 if label == 1 else -1.0 for label in y], requires_grad=False)
        X_data = pnp.array(X_4d, requires_grad=False)

        opt = qml.AdamOptimizer(stepsize=self.lr)

        print(f"   [QML] Training 4-Qubit VQC over {self.epochs} epochs...")
        for epoch in range(self.epochs):
            (self.weights, self.bias), loss = opt.step_and_cost(
                lambda w, b: self._cost(w, b, X_data, y_targets),
                self.weights,
                self.bias
            )
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   [QML] Epoch {epoch + 1:02d}/{self.epochs:02d} | Loss: {loss:.4f}")

    def predict_preprocessed(self, X_4d: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels directly from pre-transformed 4-D features.
        
        Args:
            X_4d: Numpy array of shape (N, 4) or single sample vector (4,).
            
        Returns:
            Numpy array of predicted binary labels {0, 1}.
        """
        # Handle single-sample inputs during latency benchmarking
        if X_4d.ndim == 1:
            X_4d = np.expand_dims(X_4d, axis=0)

        predictions = []
        for x in X_4d:
            exp_val = self.circuit(self.weights, x) + self.bias
            # Decision boundary: Expectation >= 0 maps to class 1, < 0 maps to class 0
            pred_label = 1 if float(exp_val) >= 0.0 else 0
            predictions.append(pred_label)

        return np.array(predictions)