"""
Generate live predictions for upcoming 2026 World Cup fixtures.

Reads data/processed/future_fixtures.csv, predicts each match using the
knockout-stage layered model, and logs the predictions with a timestamp
to live_predictions.csv. Safe to re-run — appends new predictions and
keeps the fixture list current as new rounds get scheduled.

Usage: python -m app.live_predict
"""

import pickle
import pandas as pd
from datetime import datetime
import pathlib

from models.dixon_coles import load_data
from models.knockout import predict_match

# All 9 current fixtures are Round of 32 (first knockout round in the
# 2026 48-team format). Update this if/when later rounds get added —
# ideally this would come from a "round" column in future_fixtures.csv,
# but that wasn't captured during scraping, so it's set manually here.
DEFAULT_STAGE = "r32"
WC_IMPORTANCE = 50  # from fifa_match_importance_weights.csv, up-to-QF value


def load_model():
    with open("models/dixon_coles_model.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["mod_h"], saved["mod_a"], saved["rho"]


def generate_predictions(stage: str = DEFAULT_STAGE):
    mod_h, mod_a, rho = load_model()
    df = load_data()
    fixtures = pd.read_csv("data/processed/future_fixtures.csv", parse_dates=["date"])

    rows = []
    for _, fx in fixtures.iterrows():
        result = predict_match(
            fx["home_team"], fx["away_team"], fx["neutral"], WC_IMPORTANCE,
            df, mod_h, mod_a, rho, stage=stage
        )

        r = result["result"]
        if result["stage"] == "group":
            row = {
                "predicted_at": datetime.now().isoformat(timespec="seconds"),
                "match_date": fx["date"].strftime("%Y-%m-%d"),
                "home_team": fx["home_team"],
                "away_team": fx["away_team"],
                "stage": stage,
                "p_home": round(r["p_h"], 4),
                "p_draw": round(r["p_d"], 4),
                "p_away": round(r["p_a"], 4),
                "p_home_advances": None,
                "p_away_advances": None,
            }
        else:
            row = {
                "predicted_at": datetime.now().isoformat(timespec="seconds"),
                "match_date": fx["date"].strftime("%Y-%m-%d"),
                "home_team": fx["home_team"],
                "away_team": fx["away_team"],
                "stage": stage,
                "p_home": round(r["ninety_min"]["home"], 4),
                "p_draw": round(r["ninety_min"]["draw"], 4),
                "p_away": round(r["ninety_min"]["away"], 4),
                "p_home_advances": round(r["advances"]["home"], 4),
                "p_away_advances": round(r["advances"]["away"], 4),
            }
        rows.append(row)

    predictions = pd.DataFrame(rows)
    return predictions


def save_predictions(predictions: pd.DataFrame, path="app/live_predictions.csv"):
    pathlib.Path("app").mkdir(exist_ok=True)
    out_path = pathlib.Path(path)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, predictions], ignore_index=True)
    else:
        combined = predictions

    combined.to_csv(out_path, index=False)
    print(f"Saved {len(predictions)} predictions -> {out_path}")
    print(f"Total rows in file (including past runs): {len(combined)}")


if __name__ == "__main__":
    preds = generate_predictions(stage="r32")
    print(preds.to_string(index=False))
    save_predictions(preds)