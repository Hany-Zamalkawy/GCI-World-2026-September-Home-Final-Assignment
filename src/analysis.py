from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SEED_LABEL = "deterministic Customer_ID hash split"
FEATURES = [
    "rev_Mean", "mou_Mean", "totmrc_Mean", "ovrmou_Mean", "ovrrev_Mean",
    "roam_Mean", "change_mou", "change_rev", "drop_vce_Mean", "blck_vce_Mean",
    "unan_vce_Mean", "custcare_Mean", "cc_mou_Mean", "comp_vce_Mean",
    "attempt_Mean", "callwait_Mean", "months", "eqpdays", "avg3rev", "avg3mou",
    "avg6rev", "avg6mou", "avg3qty", "hnd_price", "phones", "models",
    "uniqsubs", "actvsubs",
]


def _holdout_id(customer_id: object) -> bool:
    """Stable, well-mixed 20% holdout assignment, identical across reruns."""
    x = int(customer_id) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x % 5 == 0


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _fit_preprocessor(x_train: np.ndarray, x_valid: np.ndarray):
    train_scaled, valid_scaled, missing_flags = [], [], []
    parameters = []
    for j in range(x_train.shape[1]):
        tr = _signed_log(x_train[:, j])
        va = _signed_log(x_valid[:, j])
        finite = np.isfinite(tr)
        ordered = np.sort(tr[finite])
        if len(ordered) == 0:
            median, low, high = 0.0, 0.0, 0.0
        else:
            median = float(ordered[len(ordered) // 2])
            low = float(ordered[int(0.01 * (len(ordered) - 1))])
            high = float(ordered[int(0.99 * (len(ordered) - 1))])
        tr_missing = ~finite
        va_missing = ~np.isfinite(va)
        tr = np.where(tr_missing, median, np.clip(tr, low, high))
        va = np.where(va_missing, median, np.clip(va, low, high))
        mean = float(tr.mean())
        scale = float(tr.std()) or 1.0
        train_scaled.append((tr - mean) / scale)
        valid_scaled.append((va - mean) / scale)
        if tr_missing.any():
            missing_flags.append((tr_missing.astype(float), va_missing.astype(float)))
        parameters.append({"median": median, "low": low, "high": high, "mean": mean, "scale": scale})
    xtr = np.column_stack(train_scaled + [a for a, _ in missing_flags])
    xva = np.column_stack(valid_scaled + [b for _, b in missing_flags])
    return xtr, xva, parameters


def _fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 160, learning_rate: float = 0.18):
    """Deterministic L2-regularized logistic regression optimized with batch gradient descent."""
    n = len(y)
    positives = float(y.sum())
    intercept = math.log((positives + 0.5) / (n - positives + 0.5))
    coef = np.zeros(x.shape[1], dtype=np.float64)
    penalty = 1.0 / n
    for _ in range(epochs):
        logits = np.clip(x @ coef + intercept, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y
        intercept -= learning_rate * float(error.mean())
        gradient = x.T @ error / n + penalty * coef
        coef -= learning_rate * gradient
    return intercept, coef


def _rate_table(df: pd.DataFrame, column: str, bins: list[float], labels: list[str]):
    group = pd.cut(df[column], bins=bins, labels=labels, right=False, include_lowest=True)
    result = df.assign(_bin=group).groupby("_bin", observed=False)["churn"].agg(["size", "sum", "mean"])
    return [
        {"band": str(idx), "customers": int(row["size"]), "churners": int(row["sum"]), "churn_rate": float(row["mean"])}
        for idx, row in result.iterrows()
    ]


def run_analysis(data_dir: str | Path, output_dir: str | Path) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    client_path, record_path = data_dir / "Client.csv", data_dir / "Record.csv"
    for path in (client_path, record_path):
        if not path.exists():
            raise FileNotFoundError(f"Required source file not found: {path}")

    client = pd.read_csv(client_path)
    record = pd.read_csv(record_path)
    if client["Customer_ID"].duplicated().any() or record["Customer_ID"].duplicated().any():
        raise ValueError("Customer_ID must be unique in both source tables for the one-to-one join.")
    df = record.merge(client, on="Customer_ID", how="inner", validate="one_to_one")
    if len(df) != len(client) or len(df) != len(record):
        raise ValueError("The source tables did not form a complete one-to-one customer join.")
    df["churn"] = pd.to_numeric(df["churn"], errors="raise").astype(int)
    if not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 and 1.")

    missing = df.isna().sum().sort_values(ascending=False)
    top_missing = [
        {"field": str(name), "missing": int(count), "share": float(count / len(df))}
        for name, count in missing[missing > 0].head(15).items()
    ]
    class_summary = {}
    for label in (0, 1):
        subset = df.loc[df["churn"] == label]
        class_summary[str(label)] = {
            "customers": int(len(subset)),
            "mean_monthly_revenue": float(subset["rev_Mean"].mean()),
            "mean_monthly_minutes": float(subset["mou_Mean"].mean()),
            "mean_equipment_age_days": float(subset["eqpdays"].mean()),
            "median_tenure_months": float(subset["months"].median()),
        }
    equipment = _rate_table(df, "eqpdays", [-np.inf, 180, 365, 730, np.inf], ["0-179", "180-364", "365-729", "730+"])
    tenure = _rate_table(df, "months", [-np.inf, 7, 13, 25, 37, np.inf], ["0-6", "7-12", "13-24", "25-36", "37+"])

    y = df["churn"].to_numpy(dtype=float)
    valid_mask = np.fromiter((_holdout_id(v) for v in df["Customer_ID"].to_numpy()), dtype=bool, count=len(df))
    train_mask = ~valid_mask
    if valid_mask.sum() == 0 or train_mask.sum() == 0:
        raise ValueError("The deterministic holdout split is empty.")
    raw = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x_train, x_valid, params = _fit_preprocessor(raw[train_mask], raw[valid_mask])
    y_train, y_valid = y[train_mask], y[valid_mask]
    intercept, coef = _fit_logistic(x_train, y_train)
    valid_logits = np.clip(x_valid @ coef + intercept, -30, 30)
    valid_prob = 1.0 / (1.0 + np.exp(-valid_logits))
    auc = float(roc_auc_score(y_valid, valid_prob))
    order = np.argsort(-valid_prob, kind="mergesort")
    top_n = int(math.ceil(len(order) * 0.10))
    top_index = order[:top_n]
    top_churn = int(y_valid[top_index].sum())
    top_rev = pd.to_numeric(df.loc[valid_mask, "rev_Mean"], errors="coerce").to_numpy()[top_index]
    top_rev = top_rev[np.isfinite(top_rev)]
    base_rate = float(y_valid.mean())
    top_rate = float(y_valid[top_index].mean())
    capture = float(top_churn / y_valid.sum())
    fpr, tpr, _ = roc_curve(y_valid, valid_prob)

    # Deciles are ranked by predicted risk; decile 1 is the highest-risk 10%.
    deciles = []
    for k in range(10):
        lo = int(math.ceil(k * len(order) / 10))
        hi = int(math.ceil((k + 1) * len(order) / 10))
        idx = order[lo:hi]
        deciles.append({"decile": k + 1, "customers": int(len(idx)), "churn_rate": float(y_valid[idx].mean())})

    # Illustrative scenario, not a forecast: users must confirm currency and costs.
    monitored = 100_000
    target_rate = 0.10
    target_customers = int(monitored * target_rate)
    monthly_revenue = float(np.mean(top_rev))
    months_retained = 6
    contact_cost = 5.0
    campaign_cost = target_customers * contact_cost
    scenarios = []
    for reduction_pp in (0.01, 0.03, 0.05):
        saved = int(round(target_customers * reduction_pp))
        gross = saved * monthly_revenue * months_retained
        scenarios.append({
            "absolute_60d_churn_reduction": reduction_pp,
            "customers_retained_per_100k": saved,
            "gross_revenue_units_preserved": gross,
            "contact_cost_units": campaign_cost,
            "net_before_offer_and_service_costs": gross - campaign_cost,
        })
    breakeven_pp = campaign_cost / (target_customers * monthly_revenue * months_retained)

    metrics = {
        "source": "GCI World 2026 September Final Assignment course packet",
        "rows": int(len(df)),
        "client_rows": int(len(client)),
        "record_rows": int(len(record)),
        "client_columns": int(len(client.columns)),
        "record_columns": int(len(record.columns)),
        "merged_columns": int(len(df.columns)),
        "churners": int(y.sum()),
        "non_churners": int((1 - y).sum()),
        "churn_rate": float(y.mean()),
        "target_definition": "churn between 31 and 60 days after the observation date; per supplied dataset overview",
        "top_missing": top_missing,
        "class_summary": class_summary,
        "churn_by_equipment_age": equipment,
        "churn_by_tenure_months": tenure,
        "model": {
            "name": "L2-regularized logistic regression",
            "features": FEATURES,
            "feature_count": int(x_train.shape[1]),
            "preprocessing": "signed log1p transform; train-only 1st/99th percentile clipping; median imputation; standardization; missingness indicators",
            "split": "deterministic 80/20 holdout using a fixed integer hash of Customer_ID",
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(valid_mask.sum()),
            "validation_churn_rate": base_rate,
            "roc_auc": auc,
            "top_decile_customers": top_n,
            "top_decile_churn_rate": top_rate,
            "top_decile_lift": float(top_rate / base_rate),
            "top_decile_churn_capture": capture,
            "top_decile_mean_monthly_revenue_units": monthly_revenue,
            "risk_deciles": deciles,
            "largest_positive_coefficients": sorted(
                [{"feature": FEATURES[i], "coefficient_per_sd": float(coef[i])} for i in range(len(FEATURES))],
                key=lambda z: z["coefficient_per_sd"], reverse=True
            )[:8],
        },
        "illustrative_scenario": {
            "monitored_customers": monitored,
            "target_share": target_rate,
            "months_retained": months_retained,
            "assumed_contact_cost_per_targeted_customer": contact_cost,
            "assumption_note": "Same currency units as the dataset revenue field; currency is not specified. Scenario only, not a forecast.",
            "break_even_absolute_churn_reduction": breakeven_pp,
            "cases": scenarios,
            "exclusions": "Offer discounts, variable service costs, discounting, and actual incremental treatment effect are not included.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "validation_scores.npz", y_true=y_valid, y_score=valid_prob, fpr=fpr, tpr=tpr)
    return metrics

