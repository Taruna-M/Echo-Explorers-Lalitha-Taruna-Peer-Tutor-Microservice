import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, roc_auc_score, f1_score
import joblib
from app.transformer import BooleanToIntegerTransformer


def load_json(file_path):
    jsonData = []
    with open(file_path, 'r') as f:
        jsonData = json.load(f)
    data = pd.json_normalize(jsonData)
    data.columns = [col.replace("input.", "") for col in data.columns]
    return data


# load data
print("Loading data...")
snorkel = load_json('./data/datasetS.json')
if snorkel.empty:
    print("No data found in the JSON file.")
else:
    print("Snorkel Data loaded successfully.")
    print(snorkel.head())
heuristic = load_json('./data/datasetH.json')
if heuristic.empty:
    print("No data found in the JSON file.")
else:
    print("Heuristic Data loaded successfully.")
    print(heuristic.head())

# preprocess data
print("."*20)
print("\nPreprocessing data...")
X_snorkel = snorkel.drop("label", axis=1)
y_snorkel = snorkel["label"]
X_heuristic = heuristic.drop("label", axis=1)
y_heuristic = heuristic["label"]

# create a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('bool_to_int', BooleanToIntegerTransformer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
])

# train the model using the pipeline on Snorkel data
pipeline.fit(X_snorkel, y_snorkel)

# test the model on Heuristic data
y_pred_proba = pipeline.predict_proba(X_heuristic)[:, 1]
y_pred = (y_pred_proba >= 0.7).astype(int)


# evaluate model
print("."*20)
print("\nEvaluating model...")
print("Accuracy:", accuracy_score(y_heuristic, y_pred))
print("F1-score:", f1_score(y_heuristic, y_pred))
print("ROC AUC:", roc_auc_score(y_heuristic, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_heuristic, y_pred))

model = pipeline.named_steps['model']
features = {}
for f, i in zip(X_snorkel.columns, model.feature_importances_):
    features[f] = i
features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
print("\nFeature Importances:")
for f, i in features.items():
    print(f"{f}: {i:.4f}")

# Cross-validation
with open('./data/datasetS.json', 'r') as f:
    snorkel = json.load(f)

X = pd.DataFrame([d["input"] for d in snorkel])
y = pd.Series([d["label"] for d in snorkel])

# Define model
clf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)

# Define scoring
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

# Perform 5-fold cross-validation
cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

# Show results
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()} - Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

cm = confusion_matrix(y_heuristic, y_pred)
print("\nConfusion Matrix:\n", cm)

# save the model
print("."*20)
print("\nSaving model...")
joblib.dump(pipeline, './model/rfc_model.pkl')
print("Model saved successfully as rfc_model.pkl")
