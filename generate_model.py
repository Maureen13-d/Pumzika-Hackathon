"""
generate_model.py
=================
Run this ONCE locally (or in Colab) to produce pumzika_model.pkl.
The pkl is then committed to the repo and loaded by app.py at runtime.

Usage:
    python generate_model.py --listings listings.csv
"""

import argparse
import pickle
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--listings", default="listings.csv", help="Path to listings.csv")
parser.add_argument("--out",      default="pumzika_model.pkl")
args = parser.parse_args()

# ── LOAD ─────────────────────────────────────────────────────
print(f"Loading {args.listings}...")
df = pd.read_csv(args.listings, low_memory=False)
TARGET = "review_scores_rating"
df = df.dropna(subset=[TARGET]).copy()
y  = df[TARGET].astype(float).values
print(f"  Rows after dropping missing target: {len(df):,}")

# ── PARSE AMENITIES ───────────────────────────────────────────
def parse_amenities(raw):
    if pd.isna(raw): return set()
    s = raw.strip("{}")
    items = re.findall(r'"([^"]+)"|([^,]+)', s)
    cleaned = set()
    for a, b in items:
        item = (a or b).strip().lower()
        item = re.sub(r"[^a-z0-9 ]", " ", item)
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            cleaned.add(item)
    return cleaned

print("Parsing amenities...")
amenity_sets = df["amenities"].apply(parse_amenities)
all_a  = [a for s in amenity_sets for a in s]
counts = Counter(all_a)
n      = len(df)
pct    = {k: v / n for k, v in counts.items()}
kept   = [a for a, p in pct.items() if 0.05 <= p <= 0.95]
print(f"  Amenities after 5%-95% filter: {len(kept)}")

amenity_matrix = pd.DataFrame(
    [{a: int(a in s) for a in kept} for s in amenity_sets],
    index=df.index,
)

# ── STRUCTURAL FEATURES ───────────────────────────────────────
structural_cols = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "room_type", "property_type",
    "minimum_nights", "maximum_nights", "instant_bookable",
]
struct_df = df[[c for c in structural_cols if c in df.columns]].copy()
for col in ["room_type", "property_type", "instant_bookable"]:
    if col in struct_df.columns:
        struct_df[col] = struct_df[col].astype("category").cat.codes
struct_df = struct_df.apply(pd.to_numeric, errors="coerce")

# ── COMBINE ───────────────────────────────────────────────────
X = pd.concat(
    [struct_df.reset_index(drop=True), amenity_matrix.reset_index(drop=True)],
    axis=1,
)
X = X.fillna(X.median(numeric_only=True))
feature_names = list(X.columns)

ratio = X.shape[1] / len(X)
print(f"  Feature/sample ratio: {ratio:.4f} ({X.shape[1]} features / {len(X):,} samples)")
assert ratio < 0.5, "Feature/sample ratio too high — reduce features before proceeding."

# ── HOLDOUT SPLIT (15%) — done once, final ────────────────────
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X.values, y, test_size=0.15, random_state=42, shuffle=True
)
print(f"  Dev: {len(X_dev):,} | Holdout: {len(X_holdout):,}")

# ── TRAIN (model matches Phase 4 best-single RF) ─────────────
print("Training Random Forest (300 trees, depth=10, min_leaf=10)...")
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )),
])
model.fit(X_dev, y_dev)

# ── EVALUATE ─────────────────────────────────────────────────
y_pred = np.clip(model.predict(X_holdout), 20, 100)
rmse   = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
r2     = float(r2_score(y_holdout, y_pred))
print(f"  Holdout RMSE: {rmse:.4f}  R²: {r2:.4f}")

# ── IMPORTANCE TABLE ─────────────────────────────────────────
importances = model.named_steps["model"].feature_importances_
feat_imp    = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp    = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)

# ── ADOPTION RATES ────────────────────────────────────────────
adoption_rates = {a: amenity_matrix[a].mean() for a in kept}

# ── SAVE ─────────────────────────────────────────────────────
artifacts = {
    "model":           model,
    "feature_names":   feature_names,
    "feat_imp":        feat_imp,
    "kept_amenities":  kept,
    "adoption_rates":  adoption_rates,
    "X_holdout":       X_holdout,
    "y_holdout":       y_holdout,
    "holdout_rmse":    rmse,
    "holdout_r2":      r2,
    "y_dev_mean":      float(y_dev.mean()),
    "y_dev_std":       float(y_dev.std()),
}
with open(args.out, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n✓ Saved to {args.out}")
print("  Commit this file to your repo alongside app.py")
