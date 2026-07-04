"""
Flask backend for the World Cup Match Predictor UI.

All routes — static files AND /api/* — are served by this single Flask
function. vercel.json rewrites every request here explicitly, so there's
no dependency on Vercel's automatic public/ CDN detection.

Routes
------
GET  /                    → index.html
GET  /<filename>          → any other static file in public/ (style.css, app.js, ...)
GET  /api/teams           → sorted list of all team names
GET  /api/backtest        → backtest_report.json contents
GET  /api/live-fixtures   → live_predictions.csv as JSON
POST /api/predict         → {home, away, stage, neutral} → prediction dict

Stage → importance mapping (FIFA weights):
  group        → 25
  r32 / r16    → 50
  qf / sf / final → 60

Paths: all file references are resolved relative to the project root
(parent of this file's directory) so they work identically on Vercel
and locally.
"""

import json
import pathlib
import pickle
import sys

# Ensure the project root is on sys.path so 'models.*' imports work whether
# this file is run as 'python app/server.py' (cwd = project root) or via
# Vercel's serverless runner (cwd may vary).
_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "dixon_coles_model.pkl"
MATCHES_PATH = ROOT / "data" / "processed" / "matches_full.csv"
BACKTEST_PATH = ROOT / "backtest" / "backtest_report.json"
LIVE_PATH    = ROOT / "app" / "live_predictions.csv"
PUBLIC_PATH  = ROOT / "public"

# ── Stage → FIFA importance weight ───────────────────────────────────────────
STAGE_IMPORTANCE = {
    "group": 25,
    "r32":   50,
    "r16":   50,
    "qf":    60,
    "sf":    60,
    "final": 60,
}

# ── App + model init (loaded once at startup) ─────────────────────────────────
app = Flask(__name__)
CORS(app)

print("Loading model…", flush=True)
with open(MODEL_PATH, "rb") as fh:
    saved = pickle.load(fh)
mod_h = saved["mod_h"]
mod_a = saved["mod_a"]
rho   = saved["rho"]
print(f"  rho = {rho:.5f}", flush=True)

print("Loading feature data…", flush=True)
from models.dixon_coles import load_data
df = load_data()
print(f"  {len(df):,} rows loaded.", flush=True)

print("Loading team list…", flush=True)
_matches_df = pd.read_csv(MATCHES_PATH)
TEAMS = sorted(set(
    _matches_df["home_team"].dropna().tolist() +
    _matches_df["away_team"].dropna().tolist()
))
print(f"  {len(TEAMS)} teams.", flush=True)

# ── API routes ────────────────────────────────────────────────────────────────
# Static files (index.html, style.css, app.js) are NOT served by Flask at
# all — vercel.json routes "/" and everything else to Vercel's own CDN,
# which serves directly from public/. Flask here only ever handles /api/*.

@app.route("/api/teams")
def api_teams():
    return jsonify(TEAMS)


@app.route("/api/backtest")
def api_backtest():
    with open(BACKTEST_PATH) as fh:
        data = json.load(fh)
    return jsonify(data)


@app.route("/api/live-fixtures")
def api_live_fixtures():
    live_df = pd.read_csv(LIVE_PATH)
    # Replace NaN with None so JSON serialises cleanly
    records = live_df.where(pd.notnull(live_df), None).to_dict(orient="records")
    return jsonify(records)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    home    = body.get("home", "")
    away    = body.get("away", "")
    stage   = body.get("stage", "group").lower()
    neutral = bool(body.get("neutral", True))

    if not home or not away:
        return jsonify({"error": "home and away are required"}), 400
    if home == away:
        return jsonify({"error": "Teams must be different"}), 400
    if stage not in STAGE_IMPORTANCE:
        return jsonify({"error": f"Unknown stage '{stage}'"}), 400

    importance = STAGE_IMPORTANCE[stage]

    from models.knockout import predict_match
    result = predict_match(
        home=home, away=away,
        neutral=neutral,
        importance=importance,
        df=df,
        mod_h=mod_h, mod_a=mod_a, rho=rho,
        stage=stage,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)