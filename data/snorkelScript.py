import pandas as pd
import numpy as np
import random
import json
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

random.seed(42)
np.random.seed(42)

MAX_INACTIVE_DAYS = 14

# --------- Labeling Functions --------- #

@labeling_function()
def lf_strong_karma(x):
    return 1 if x.karma_in_topic >= 80 else -1

@labeling_function()
def lf_mid_karma_support(x):
    return 1 if 60 <= x.karma_in_topic < 80 else -1

@labeling_function()
def lf_weak_karma(x):
    return 0 if x.karma_in_topic <= 15 else -1

@labeling_function()
def lf_moderate_recency(x):
    return 1 if x.days_since_last_help <= 4 else -1

@labeling_function()
def lf_exceeded_inactive(x):
    return 0 if x.days_since_last_help > MAX_INACTIVE_DAYS else -1

@labeling_function()
def lf_branch_plus_karma(x):
    return 1 if x.same_branch and x.karma_in_topic >= 40 else -1

@labeling_function()
def lf_college_if_active(x):
    return 1 if x.same_college and x.days_since_last_help <= 7 else -1

@labeling_function()
def lf_peer_year_subtle(x):
    return 1 if x.peer_year_match and x.karma_in_topic >= 50 else -1

@labeling_function()
def lf_combo_priority_karma(x):
    pos = (
        2 * (x.karma_in_topic >= 70) +
        (x.days_since_last_help <= 3) +
        (x.same_branch) +
        (x.same_college) +
        (x.peer_year_match)
    )
    return 1 if pos >= 4 else -1

@labeling_function()
def lf_combo_penalty_days(x):
    neg = sum([
        x.days_since_last_help > MAX_INACTIVE_DAYS,
        x.karma_in_topic < 25,
        not x.same_branch,
        not x.same_college
    ])
    return 0 if neg >= 3 else -1

# --------- Dataset Code --------- #

def generate_random_features(n):
    return pd.DataFrame({
        "karma_in_topic": np.random.randint(0, 101, n),
        "same_college": np.random.choice([True, False], n),
        "days_since_last_help": np.random.randint(0, 31, n),
        "same_branch": np.random.choice([True, False], n),
        "peer_year_match": np.random.choice([True, False], n)
    })

def convert_to_json_format(df):
    return [
        {
            "input": {
                "karma_in_topic": int(row["karma_in_topic"]),
                "same_college": bool(row["same_college"]),
                "days_since_last_help": int(row["days_since_last_help"]),
                "same_branch": bool(row["same_branch"]),
                "peer_year_match": bool(row["peer_year_match"]),
            },
            "label": int(row["label"]),
        }
        for _, row in df.iterrows()
    ]

def main():
    df = generate_random_features(1000)

    lfs = [
        lf_strong_karma,
        lf_mid_karma_support,
        lf_weak_karma,
        lf_moderate_recency,
        lf_exceeded_inactive,
        lf_branch_plus_karma,
        lf_college_if_active,
        lf_peer_year_subtle,
        lf_combo_priority_karma,
        lf_combo_penalty_days
    ]

    applier = PandasLFApplier(lfs)
    L = applier.apply(df)

    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L, n_epochs=500, seed=42)

    probs = label_model.predict_proba(L)
    labels = []
    for p in probs:
        if p[1] > 0.95:
            labels.append(1)
        elif p[1] < 0.05:
            labels.append(0)
        elif p[1] > 0.5:
            labels.append(1 if random.random() < 0.8 else 0)
        else:
            labels.append(0 if random.random() < 0.8 else 1)

    df["label"] = labels

    # Balance the dataset
    pos, neg = df[df.label == 1], df[df.label == 0]
    keep = max(300, min(len(pos), len(neg)))
    df_balanced = pd.concat([
        pos.sample(keep, random_state=42),
        neg.sample(keep, random_state=42)
    ])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    json_data = convert_to_json_format(df_balanced)

    with open("datasetS.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Generated dataset with {len(json_data)} samples.")
    print(json.dumps(json_data[:3], indent=2))

if __name__ == "__main__":
    main()
