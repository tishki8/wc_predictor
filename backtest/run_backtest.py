"""
Backtest the Dixon-Coles model against a past World Cup.

Metrics:
  - Brier score: mean squared error between predicted probabilities and
    actual outcomes (lower is better, 0 = perfect)
  - Log loss: penalizes confident wrong predictions more harshly
  - Accuracy: simple % of matches where the highest-probability outcome
    matched the actual result
"""

import pickle
import numpy as np
import pandas as pd
import json
import pathlib

from models.dixon_coles import load_data, predict_from_row


def brier_score(probs_list, actual_list):
    """
    probs_list: list of (p_h, p_d, p_a) tuples
    actual_list: list of "H"/"D"/"A" strings
    """
    total = 0.0
    for (p_h, p_d, p_a), actual in zip(probs_list, actual_list):
        y_h = 1.0 if actual == "H" else 0.0
        y_d = 1.0 if actual == "D" else 0.0
        y_a = 1.0 if actual == "A" else 0.0
        total += (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2
    return total / len(probs_list)


def log_loss_score(probs_list, actual_list, eps=1e-15):
    total = 0.0
    for (p_h, p_d, p_a), actual in zip(probs_list, actual_list):
        p = {"H": p_h, "D": p_d, "A": p_a}[actual]
        p = max(p, eps)
        total += -np.log(p)
    return total / len(probs_list)


def run_backtest(tournament_year: int, tournament_name: str = "FIFA World Cup"):
    with open("models/dixon_coles_model.pkl", "rb") as f:
        saved = pickle.load(f)
    mod_h, mod_a, rho = saved["mod_h"], saved["mod_a"], saved["rho"]

    df = load_data()
    test_matches = df[
        (df["date"].dt.year == tournament_year) &
        (df["tournament"] == tournament_name)
    ].copy()

    print(f"Backtesting on {len(test_matches)} matches from {tournament_year} {tournament_name}")

    probs_list = []
    actual_list = []
    correct = 0

    rows = []
    for _, r in test_matches.iterrows():
        p = predict_from_row(r, mod_h, mod_a, rho)
        probs = (p["p_h"], p["p_d"], p["p_a"])
        actual = r["outcome"]

        probs_list.append(probs)
        actual_list.append(actual)

        pred = max({"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}, key=lambda k: {"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}[k])
        if pred == actual:
            correct += 1

        rows.append({
            "match": f"{r['home_team']} v {r['away_team']}",
            "date": r["date"].strftime("%Y-%m-%d"),
            "P(H)": round(p["p_h"], 3),
            "P(D)": round(p["p_d"], 3),
            "P(A)": round(p["p_a"], 3),
            "actual": actual,
            "correct": pred == actual,
        })

    brier = brier_score(probs_list, actual_list)
    logloss = log_loss_score(probs_list, actual_list)
    accuracy = correct / len(test_matches)

    report = {
        "tournament": f"{tournament_year} {tournament_name}",
        "n_matches": len(test_matches),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "accuracy": round(accuracy, 4),
    }

    print("\n" + "=" * 50)
    print(f"Brier score : {brier:.4f}  (lower is better, 0 = perfect)")
    print(f"Log loss    : {logloss:.4f}  (lower is better)")
    print(f"Accuracy    : {accuracy:.1%}")
    print("=" * 50)

    out_dir = pathlib.Path("backtest")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "backtest_detail.csv", index=False)

    print(f"\nSaved: backtest/backtest_report.json, backtest/backtest_detail.csv")
    return report
def run_baseline_comparison(tournament_year: int, tournament_name: str = "FIFA World Cup"):
    """Compare against a naive baseline: always predict the training-set base rates."""
    df = load_data()
    train = df[df["date"] < "2022-01-01"]
    base_rates = train["outcome"].value_counts(normalize=True)
    p_h_base = base_rates.get("H", 0.33)
    p_d_base = base_rates.get("D", 0.33)
    p_a_base = base_rates.get("A", 0.33)

    test_matches = df[
        (df["date"].dt.year == tournament_year) &
        (df["tournament"] == tournament_name)
    ]

    probs_list = [(p_h_base, p_d_base, p_a_base)] * len(test_matches)
    actual_list = test_matches["outcome"].tolist()

    brier = brier_score(probs_list, actual_list)
    logloss = log_loss_score(probs_list, actual_list)

    print(f"\nBASELINE (always predict {p_h_base:.1%}/{p_d_base:.1%}/{p_a_base:.1%}):")
    print(f"  Brier score : {brier:.4f}")
    print(f"  Log loss    : {logloss:.4f}")
def calibration_check(tournament_year: int, tournament_name: str = "FIFA World Cup"):
    """For matches where the model was >70% confident, how often was it right?"""
    with open("models/dixon_coles_model.pkl", "rb") as f:
        saved = pickle.load(f)
    mod_h, mod_a, rho = saved["mod_h"], saved["mod_a"], saved["rho"]

    df = load_data()
    test_matches = df[
        (df["date"].dt.year == tournament_year) &
        (df["tournament"] == tournament_name)
    ]

    high_conf = []
    for _, r in test_matches.iterrows():
        p = predict_from_row(r, mod_h, mod_a, rho)
        probs = {"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}
        pred = max(probs, key=probs.get)
        conf = probs[pred]
        if conf > 0.70:
            high_conf.append({"match": f"{r['home_team']} v {r['away_team']}",
                             "confidence": round(conf, 3), "predicted": pred,
                             "actual": r["outcome"], "correct": pred == r["outcome"]})

    hc_df = pd.DataFrame(high_conf)
    if len(hc_df):
        print(f"\n{len(hc_df)} matches with >70% confidence:")
        print(hc_df.to_string(index=False))
        print(f"\nAccuracy on these high-confidence picks: {hc_df['correct'].mean():.1%}")
        print("(should be close to 70%+ if well-calibrated; much lower = overconfident)")
if __name__ == "__main__":
    run_backtest(2022)
    run_baseline_comparison(2022)
    calibration_check(2022)