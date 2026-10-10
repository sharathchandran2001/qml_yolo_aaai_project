import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ClassicalClassifier:
    """SVM (RBF), Random Forest, or a majority-class baseline behind one interface.

    scores(X) returns a continuous score where HIGHER = more likely positive class (label 1),
    which is what ROC-AUC / PR-AUC need.
    """
    def __init__(self, model_type="svm", seed=42):
        self.model_type = model_type.lower()
        if self.model_type == "svm":
            self.model = SVC(kernel="rbf", C=1.0, gamma="scale")
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=seed)
        elif self.model_type in ("majority", "dummy"):
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train, verbose=False):
        if verbose:
            print(f"[+] Training Classical ML model ({self.model_type.upper()})...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def scores(self, X):
        if self.model_type == "svm":
            return self.model.decision_function(X)
        if self.model_type == "rf":
            proba = self.model.predict_proba(X)
            classes = list(self.model.classes_)
            return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        return np.zeros(len(X))  # majority baseline: constant score

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", cm)
        return acc, report


# Alias to maintain compatibility with classical_pipeline.py
ClassicalPipeline = ClassicalClassifier
