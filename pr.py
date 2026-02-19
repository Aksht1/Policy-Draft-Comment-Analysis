import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("final_clean_dataset (1).csv")

TEXT_COLUMN = df.columns[0]
LABEL_COLUMN = df.columns[1]

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Support Vector Machine": LinearSVC()
}

results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append([name, acc, prec, rec, f1])

    print(name)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

plt.figure(figsize=(6, 4))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylim(0, 1)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(results_df["Model"], results_df["F1-Score"])
plt.ylim(0, 1)
plt.title("F1-Score Comparison")
plt.ylabel("F1-Score")
plt.show()

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table_data = results_df.copy()
table_data[["Accuracy", "Precision", "Recall", "F1-Score"]] = table_data[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].round(3)

ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc="center",
    loc="center"
)

plt.title("Performance Comparison of TF-IDF Based Models")
plt.show()

svm_model = models["Support Vector Machine"]
svm_predictions = svm_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, svm_predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Support Vector Machine")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
