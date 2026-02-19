"""
Script to train and save the TF-IDF vectorizer and models for the Policy Draft Comment Analysis project.
Run this script once to generate the model files (tfidf_vectorizer.pkl, svm_model.pkl, lr_model.pkl).
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
print("Loading dataset...")
df = pd.read_csv("final_clean_dataset (1).csv")

# Identify text and label columns
TEXT_COLUMN = 'comment_text' if 'comment_text' in df.columns else df.columns[0]
LABEL_COLUMN = 'stance_label' if 'stance_label' in df.columns else df.columns[1]

print(f"Using text column: {TEXT_COLUMN}")
print(f"Using label column: {LABEL_COLUMN}")

# Clean data
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
df = df[df[TEXT_COLUMN].astype(str).str.strip() != '']

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and fit TF-IDF vectorizer
print("Training TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save TF-IDF vectorizer
print("Saving TF-IDF vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Train and save Logistic Regression model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

print("Saving Logistic Regression model...")
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Train and save SVM model
print("Training SVM model...")
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

svm_acc = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_acc:.4f}")

print("Saving SVM model...")
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save test metrics for display in app
metrics = {
    'lr': {
        'accuracy': float(lr_acc),
        'precision': float(precision_score(y_test, y_pred_lr, average='weighted')),
        'recall': float(recall_score(y_test, y_pred_lr, average='weighted')),
        'f1': float(f1_score(y_test, y_pred_lr, average='weighted')),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    },
    'svm': {
        'accuracy': float(svm_acc),
        'precision': float(precision_score(y_test, y_pred_svm, average='weighted')),
        'recall': float(recall_score(y_test, y_pred_svm, average='weighted')),
        'f1': float(f1_score(y_test, y_pred_svm, average='weighted')),
        'confusion_matrix': confusion_matrix(y_test, y_pred_svm).tolist()
    },
    'labels': sorted(y.unique().tolist())
}

print("Saving model metrics...")
with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\nAll models and metrics saved successfully!")
print("You can now run the Streamlit app with: streamlit run app.py")

