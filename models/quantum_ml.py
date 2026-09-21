import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

class QuantumClassifierModel:
    """
    Variational Quantum Classifier (VQC) implemented using PennyLane.
    Accepts n-dimensional angularly encoded feature vectors.
    """
    def __init__(self, n_qubits: int = 6, n_layers: int = 3, lr: float = 0.1, epochs: int = 20):
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
            # 1. Angle Encoding via RX rotations across n qubits
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

    def _cost(self, weights, bias, X, y_targets, sample_weights):
        """Class-weighted Mean Squared Error cost function on target expectation values."""
        predictions = pnp.stack([self.circuit(weights, x) + bias for x in X])
        squared_errors = (predictions - y_targets) ** 2
        return pnp.mean(sample_weights * squared_errors)

    def fit_preprocessed(self, X_nd: np.ndarray, y: np.ndarray):
        """
        Trains the VQC directly on pre-transformed N-D features using class weighting.
        
        Args:
            X_nd: Numpy array of shape (N, n_qubits) with pre-scaled PCA features.
            y: Target binary class labels in {0, 1}.
        """
        # Calculate inverse class weights to penalize minority class classification errors
        classes, counts = np.unique(y, return_counts=True)
        class_weights = {cls: len(y) / (len(classes) * count) for cls, count in zip(classes, counts)}
        sample_weights = pnp.array([class_weights[label] for label in y], requires_grad=False)

        # Map binary class labels {0, 1} to Pauli-Z target expectation domain {-1, +1}
        y_targets = pnp.array([1.0 if label == 1 else -1.0 for label in y], requires_grad=False)
        X_data = pnp.array(X_nd, requires_grad=False)

        opt = qml.AdamOptimizer(stepsize=self.lr)

        print(f"   [QML] Training {self.n_qubits}-Qubit VQC over {self.epochs} epochs...")
        for epoch in range(self.epochs):
            (self.weights, self.bias), loss = opt.step_and_cost(
                lambda w, b: self._cost(w, b, X_data, y_targets, sample_weights),
                self.weights,
                self.bias
            )
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   [QML] Epoch {epoch + 1:02d}/{self.epochs:02d} | Weighted Loss: {loss:.4f}")

    def predict_preprocessed(self, X_nd: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels directly from pre-transformed N-D features.
        
        Args:
            X_nd: Numpy array of shape (N, n_qubits) or single sample vector (n_qubits,).
            
        Returns:
            Numpy array of predicted binary labels {0, 1}.
        """
        if X_nd.ndim == 1:
            X_nd = np.expand_dims(X_nd, axis=0)

        predictions = []
        for x in X_nd:
            exp_val = self.circuit(self.weights, x) + self.bias
            pred_label = 1 if float(exp_val) >= 0.0 else 0
            predictions.append(pred_label)

        return np.array(predictions)