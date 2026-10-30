import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

#load data
df = pd.read_csv("k13_binary.csv")
df.columns = [c.strip() for c in df.columns]
df["Resistance"] = df["Resistance"].str.strip()
df["label"] = (df["Resistance"] != "Sensitive").astype(int)

feature_names = ["Propensity", "PI", "II", "GRAVY", "ALIN", "SH", "SASA", "HB", "BDE", "RMSD"]
X = df[feature_names].values
y = df["label"].values

#fine grid to find regularisation strength
C_GRID = np.logspace(-2, 2, 50)

#ranking all features by selection stability
loo = LeaveOneOut()
selection_counts = Counter()
coef_magnitudes = {f: [] for f in feature_names}

for train_idx, test_idx in loo.split(X):
    X_train, y_train = X[train_idx], y[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso_selector = LogisticRegressionCV(
        Cs=C_GRID, cv=5, penalty="l1", solver="liblinear",
        class_weight="balanced", max_iter=5000
    )
    lasso_selector.fit(X_train_scaled, y_train)
    coefs = lasso_selector.coef_.ravel()

    for f, c in zip(feature_names, coefs):
        if c != 0:
            selection_counts[f] += 1
            coef_magnitudes[f].append(abs(c))

n_folds = len(y)
mean_abs_coef = {f: (np.mean(coef_magnitudes[f]) if coef_magnitudes[f] else 0) for f in feature_names}

ranked_features = sorted(
    feature_names,
    key=lambda f: (selection_counts[f], mean_abs_coef[f]),
    reverse=True
)

print("Feature ranking by LOOCV Lasso stability:")
for rank, f in enumerate(ranked_features, 1):
    pct = 100 * selection_counts[f] / n_folds
    print(f"  {rank}. {f:12s} selected {selection_counts[f]}/{n_folds} folds ({pct:.0f}%), mean|coef|={mean_abs_coef[f]:.3f}")

def loocv_accuracy(X_subset, y, threshold=0.4):
    loo_inner = LeaveOneOut()
    yt, yp = [], []
    for tr_idx, te_idx in loo_inner.split(X_subset):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_subset[tr_idx])
        X_te = sc.transform(X_subset[te_idx])
        clf = LogisticRegression(penalty="l2", class_weight="balanced", solver="liblinear", max_iter=5000)
        clf.fit(X_tr, y[tr_idx])
        prob = clf.predict_proba(X_te)[0, 1]
        yt.append(y[te_idx][0])
        yp.append(int(prob >= threshold))
    return accuracy_score(yt, yp)


k_values = list(range(1, 11))
accuracies = []


for k in k_values:
    top_k = ranked_features[:k]
    X_subset = df[top_k].values
    acc = loocv_accuracy(X_subset, y, threshold=0.4)
    accuracies.append(acc)
    print(f"k={k:2d}  features={top_k}  accuracy={acc:.4f}")


best_k = k_values[int(np.argmax(accuracies))]

plt.figure(figsize=(9, 5.5))
plt.plot(k_values, accuracies, color="mediumpurple", linewidth=3, label="LOOCV accuracy")
plt.axvline(3, color="crimson", linestyle="dotted", label="k=3 (AI+II+BDE)")
plt.xlabel("Number of top-ranked features used (k)")
plt.ylabel("Model Accuracy")
plt.title("Accuracy vs k sweep")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=300)
plt.close()

print(f"\nBest k at threshold=0.4: {best_k} (accuracy={max(accuracies):.4f})")
print("Saved accuracy_vs_k.png")
