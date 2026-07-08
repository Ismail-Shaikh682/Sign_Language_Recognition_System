import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import numpy as np

print("Loading dataset...")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dataset.csv")

print(f"Total rows BEFORE cleaning : {len(df)}")
print(f"Unique labels BEFORE        : {df.iloc[:, -1].nunique()}")

# =====================
# REMOVE RARE CLASSES
# =====================
counts       = df.iloc[:, -1].value_counts()
valid_labels = counts[counts >= 5].index          # need at least 5 samples
df           = df[df.iloc[:, -1].isin(valid_labels)]

print(f"\nAfter removing rare labels:")
print(f"Total rows    : {len(df)}")
print(f"Unique labels : {df.iloc[:, -1].nunique()}")

# =====================
# FEATURES & LABELS
# =====================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"\nFeature count  : {X.shape[1]}")   # should be 42
print(f"Unique labels  : {sorted(df.iloc[:, -1].unique())}")

# =====================
# CHECK FEATURE COUNT
# =====================
if X.shape[1] != 42:
    print(f"\nWARNING: Expected 42 features (21 landmarks × x,y) but got {X.shape[1]}.")
    print("Re-run 2_create_dataset.py to regenerate dataset.csv with fixed normalizer.")

# =====================
# TRAIN / TEST SPLIT  (stratified = proportional classes)
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # re-enabled — safe now that rare labels removed
)

print(f"\nTrain : {len(X_train)},  Test : {len(X_test)}")

# =====================
# TRAIN MODEL
# =====================
print("\nTraining RandomForest... (1-2 min)")

model = RandomForestClassifier(
    n_estimators=300,       # more trees = better accuracy
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced" # ← helps when some signs have fewer samples
)

model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy : {acc * 100:.2f}%")

# Cross-validation for reliable estimate
cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
print(f"5-Fold CV     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================
# SAVE
# =====================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("model.pkl saved!")

with open("model_labels.json", "w", encoding="utf-8") as f:
    labels = sorted(df.iloc[:, -1].unique().tolist())
    json.dump(labels, f, ensure_ascii=False, indent=2)
print("model_labels.json saved!")

print("\n✅ Training complete!")
print("Next → Run main.py")
