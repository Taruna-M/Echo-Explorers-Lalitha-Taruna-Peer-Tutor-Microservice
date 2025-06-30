import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_json(file_path):
    jsonData = []
    with open(file_path, 'r') as f:
        jsonData = json.load(f)
    data = pd.json_normalize(jsonData)
    data.columns = [col.replace("input.", "") for col in data.columns]
    return data


snorkel = load_json('./data/datasetS.json')
# if snorkel.empty:
#     print("No data found in the JSON file.")
# else:
#     print("Snorkel Data loaded successfully.")
#     print(snorkel.head())
heuristic = load_json('./data/datasetH.json')
# if heuristic.empty:
#     print("No data found in the JSON file.")
# else:
#     print("Heuristic Data loaded successfully.")
#     print(heuristic.head())

def preprocess_data(df):
    # Convert boolean columns to integers
    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Split the data into training and test sets
    X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For tree models keep the raw
    X_train_tree = X_train1.copy()
    X_test_tree = X_test1.copy()

    # For non-tree models, scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train1)
    X_test_scaled = scaler.transform(X_test1)

    X_train={
        "raw": X_train_tree,
        "scaled": X_train_scaled
    }
    X_test={
        "raw": X_test_tree,
        "scaled": X_test_scaled
    }
    return X_train, X_test, y_train, y_test

X_train_snorkel, X_test_snorkel, y_train_snorkel, y_test_snorkel = preprocess_data(snorkel)
X_train_heuristic, X_test_heuristic, y_train_heuristic, y_test_heuristic = preprocess_data(heuristic)

def logisticRegression(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train['scaled'], y_train)
    y_pred = model.predict(X_test['scaled'])
    y_prob = model.predict_proba(X_test['scaled'])[:, 1]
    return metrics(y_test, y_pred, y_prob)

def randomForest(X_train, y_train, X_test, y_test):
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train['raw'], y_train)
    y_pred = model.predict(X_test['raw'])
    y_prob = model.predict_proba(X_test['raw'])[:, 1]
    return metrics(y_test, y_pred, y_prob)

def gradientBoostingClassifer(X_train, y_train, X_test, y_test):
    print("Training Gradient Boosting Classifier model...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train['raw'], y_train)
    y_pred = model.predict(X_test['raw'])
    y_prob = model.predict_proba(X_test['raw'])[:, 1]
    return metrics(y_test, y_pred, y_prob)

def gradientBoostingRegressor(X_train, y_train, X_test, y_test):
    print("Training Gradient Boosting Regressor model...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train['raw'], y_train)
    y_pred_gbr_cont = model.predict(X_test['raw'])
    y_pred_gbr = (y_pred_gbr_cont >= 0.5).astype(int)
    return metrics(y_test, y_pred_gbr, y_pred_gbr_cont)

def metrics(y_test, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)
    return metrics

def kfold_cv(df, dataset_name, n_splits=5):
    results = []

    # Convert bools to int
    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_num = 1
    for train_index, val_index in skf.split(X, y):
        print(f"Fold {fold_num}/{n_splits}")

        X_train_raw, X_val_raw = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        X_train = {"raw": X_train_raw, "scaled": X_train_scaled}
        X_val = {"raw": X_val_raw, "scaled": X_val_scaled}

        # Logistic Regression
        lr_metrics = logisticRegression(X_train, y_train, X_val, y_val)
        lr_metrics.update({"Model": "Logistic Regression", "Dataset": dataset_name, "Fold": fold_num})
        results.append(lr_metrics)

        # Random Forest
        rf_metrics = randomForest(X_train, y_train, X_val, y_val)
        rf_metrics.update({"Model": "Random Forest", "Dataset": dataset_name, "Fold": fold_num})
        results.append(rf_metrics)

        # Gradient Boosting Classifier
        gbc_metrics = gradientBoostingClassifer(X_train, y_train, X_val, y_val)
        gbc_metrics.update({"Model": "Gradient Boosting Classifier", "Dataset": dataset_name, "Fold": fold_num})
        results.append(gbc_metrics)

        # Gradient Boosting Regressor
        gbr_metrics = gradientBoostingRegressor(X_train, y_train, X_val, y_val)
        gbr_metrics.update({"Model": "Gradient Boosting Regressor", "Dataset": dataset_name, "Fold": fold_num})
        results.append(gbr_metrics)

        fold_num += 1

    return pd.DataFrame(results)

def train_and_evaluate_model(dataset, X_train, y_train, X_test, y_test):
    results = []

    def add_result(metrics, name, dataset_name):
        metrics["Model"] = name
        metrics["Dataset"] = dataset_name
        results.append(metrics)

    print(f"Training and evaluating model on {dataset} data...")
    # Logistic Regression
    metrics = logisticRegression(X_train, y_train, X_test, y_test)
    add_result(metrics, "Logistic Regression", dataset)

    # Random Forest
    metrics = randomForest(X_train, y_train, X_test, y_test)
    add_result(metrics, "Random Forest", dataset)

    # Gradient Boosting Classifier
    metrics = gradientBoostingClassifer(X_train, y_train, X_test, y_test)
    add_result(metrics, "Gradient Boosting Classifier", dataset)

    # Gradient Boosting Regressor
    metrics = gradientBoostingRegressor(X_train, y_train, X_test, y_test)
    add_result(metrics, "Gradient Boosting Regressor", dataset)

    return pd.DataFrame(results)

# Train and evaluate models on Snorkel data
df_heuristic = train_and_evaluate_model("Heuristic", X_train_heuristic, y_train_heuristic, X_test_heuristic, y_test_heuristic)
df_snorkel = train_and_evaluate_model("Snorkel", X_train_snorkel, y_train_snorkel, X_test_snorkel, y_test_snorkel)



df_heuristic_cv = kfold_cv(heuristic, "Heuristic")
df_snorkel_cv = kfold_cv(snorkel, "Snorkel")

# Average results per model/dataset
df_heuristic_avg = df_heuristic_cv.drop(columns="Fold").groupby(["Model", "Dataset"]).mean().reset_index()
df_snorkel_avg = df_snorkel_cv.drop(columns="Fold").groupby(["Model", "Dataset"]).mean().reset_index()

print("\nAverage metrics for Heuristic dataset:")
print(df_heuristic_cv.groupby("Model")[["Accuracy", "F1-score", "ROC AUC"]].std())

print("\nAverage metrics for Snorkel dataset:")
print(df_snorkel_cv.groupby("Model")[["Accuracy", "F1-score", "ROC AUC"]].std())

print("\nAverage metrics for Heuristic dataset:")
print(df_heuristic_avg)
print("\nAverage metrics for Snorkel dataset:")
print(df_snorkel_avg)

# Print results
print("\nHeuristic Dataset Results:")
print(df_heuristic)
print("\nSnorkel Dataset Results:")
print(df_snorkel)