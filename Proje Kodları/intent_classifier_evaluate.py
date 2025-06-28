# intent_classifier_evaluate.py

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("data/data_set.xlsx")
df.columns = df.columns.str.strip()
df.dropna(subset=['user_input', 'intent', 'answer'], inplace=True)

# Load classifier
intent_clf = joblib.load("models/intent_classifier.pkl")

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Features and labels
X_texts = df['user_input'].tolist()
y = df['intent'].tolist()

# Generate embeddings
X_embeds = embedder.encode(X_texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_embeds, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(intent_clf, X_embeds, y, cv=cv, scoring='f1_macro')
print(f"\nCross-val F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Predict on test set
y_pred = intent_clf.predict(X_test)

# Classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y)))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Intent Classifier Confusion Matrix")
plt.tight_layout()
plt.show()
