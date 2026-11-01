import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import json


#load data
df = pd.read_csv("k13_pc.csv")
df.columns = [c.strip() for c in df.columns]

#core features
core_features = ["ALIN"]
X = df[core_features].values
half_life = df["Resistance"].values
y = np.log(half_life) 

print(f"Samples: {len(y)}")
print(f"Core feature: {core_features}\n")

#LOOCV performance
loo = LeaveOneOut()
y_true_all, y_pred_all = [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Lasso(alpha=0.01, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    y_true_all.append(y_test[0])
    y_pred_all.append(model.predict(X_test_scaled)[0])

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

press = np.sum((y_true_all - y_pred_all) ** 2)
tss = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
q2 = 1 - press / tss

print("LOOCV performance")
print(f"Q2: {q2:.3f}")

#final refit
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

final_model = Lasso(alpha=0.01, max_iter=10000)
final_model.fit(X_scaled_final, y)

#coefficient and intercept by OLS for r2
slope, intercept, r_value, p_value_slope, std_err = stats.linregress(df["ALIN"].values, y)
r2 = r_value ** 2

print("\nFinal deployed model")
print(f"Feature: {core_features}")
print(f"In-sample R2: {r2:.3f}  |  OLS slope p-value: {p_value_slope:.4f}")

x = df["ALIN"].values
n = len(y)
resid = y - (intercept + slope * x)

#creating config.json
reg_cfg = {
    "features": core_features,
    "mean": scaler_final.mean_.tolist(),
    "scale": scaler_final.scale_.tolist(),
    "coef": final_model.coef_.ravel().tolist(),
    "intercept": float(np.ravel(final_model.intercept_)[0]),
    "pi": {
        "n": int(n),
        "x_mean": float(x.mean()),
        "sxx": float(((x - x.mean()) ** 2).sum()),
        "s": float(np.sqrt((resid ** 2).sum() / (n - 2))),
        "t_crit": float(stats.t.ppf(0.975, n - 2)),
    },
}
with open("regression_config.json", "w") as f:
    json.dump(reg_cfg, f, indent=2)
print("Saved regression_config.json")

#coefficient extraction
def get_coefficients():
    coef_dict = {"intercept": final_model.intercept_}
    coef_dict.update(dict(zip(core_features, final_model.coef_)))
    return coef_dict

#prediction function
def predict_half_life(feature_dict):
    x_raw = np.array([[feature_dict[f] for f in core_features]])
    x_scaled = scaler_final.transform(x_raw)

    pred_log = final_model.predict(x_scaled)[0]
    pred_half_life = np.exp(pred_log) 

    return pred_half_life, pred_log

#integration and testing
if __name__ == "__main__":
    print("\nFinal model coefficients:")
    for name, val in get_coefficients().items():
        print(f"  {name}: {val:.4f}")
    pre = predict_half_life({"ALIN": 78})
    print(f"\nSample prediction:\n Predicted Parasite Clearance half-life = {pre[0]:.3f})")