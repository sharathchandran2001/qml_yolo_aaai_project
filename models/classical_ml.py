import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ClassicalClassifier:
    def __init__(self, model_type="svm"):
        self.model_type = model_type.lower()
        if self.model_type == "svm":
            self.model = SVC(kernel='rbf', C=1.0, probability=True)
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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

    def predict(self, X):
        return self.model.predict(X)

# Alias to maintain compatibility with classical_pipeline.py
ClassicalPipeline = ClassicalClassifier