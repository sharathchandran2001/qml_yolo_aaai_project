import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# In models/classical_ml.py

class ClassicalClassifier:
    def __init__(self, model_type="svm", seed=None):
        self.model_type = model_type.lower()
        
        if self.model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=seed)
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=seed)
        elif self.model_type == "majority":
            from sklearn.dummy import DummyClassifier
            self.model = DummyClassifier(strategy="most_frequent", random_state=seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def scores(self, X):
        """Returns positive class probability or decision function scores."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Return class 1 probability if binary, otherwise full output
            return probs[:, 1] if probs.ndim > 1 and probs.shape[1] == 2 else probs.ravel()
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        return self.model.predict(X)

    def train(self, X_train, y_train):
        print(f"[+] Training Classical ML model ({self.model_type.upper()})...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", cm)
        return acc, report


# Alias to maintain compatibility with classical_pipeline.py
ClassicalPipeline = ClassicalClassifier