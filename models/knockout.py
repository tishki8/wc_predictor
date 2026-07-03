"""
Knockout-stage probability layering, built on top of the Dixon-Coles model.

For knockout matches, a draw isn't a valid final result. This module takes
the 90-minute Win/Draw/Loss probabilities from dixon_coles.py and layers on:
  1. 90-minute result (from the existing model, unchanged)
  2. If drawn: extra-time result, using scaled-down expected goals
     (~1/3 of full-match λ/μ, since ET is 30 min vs 90)
  3. If still drawn after ET: penalty shootout, modeled as 50/50
  4. Combined "which team advances" probability, summing to 100%
"""

from models.dixon_coles import scoreline_grid, wdl_from_grid, predict_fixture

ET_SCALE = 30 / 90  # extra time is 30 minutes vs 90 for full match


def predict_knockout(home, away, neutral, importance, df, mod_h, mod_a, rho,
                     penalty_split=0.5):
    """
    Full layered knockout prediction.

    Returns a dict with each layer's numbers separately (for the UI to show
    the breakdown, not just the final number) plus the combined advance
    probabilities.
    """
    # ── Layer 1: 90-minute result (reuse the existing model as-is) ────
    ninety = predict_fixture(home, away, neutral, importance, df, mod_h, mod_a, rho)
    p_h_90, p_d_90, p_a_90 = ninety["p_h"], ninety["p_d"], ninety["p_a"]

    # ── Layer 2: extra time (only matters if 90 min was drawn) ────────
    lam_et = ninety["xG_h"] * ET_SCALE
    mu_et  = ninety["xG_a"] * ET_SCALE
    grid_et = scoreline_grid(lam_et, mu_et, rho)
    p_h_et, p_d_et, p_a_et, _, _ = wdl_from_grid(grid_et)

    # ── Layer 3: penalties (only matters if still drawn after ET) ─────
    p_h_pens = penalty_split
    p_a_pens = 1 - penalty_split

    # ── Combine into final advance probabilities ──────────────────────
    p_home_advances = p_h_90 + p_d_90 * (p_h_et + p_d_et * p_h_pens)
    p_away_advances = p_a_90 + p_d_90 * (p_a_et + p_d_et * p_a_pens)

    return {
        "ninety_min":  {"home": p_h_90, "draw": p_d_90, "away": p_a_90},
        "extra_time":  {"home": p_h_et, "draw": p_d_et, "away": p_a_et},
        "penalties":   {"home": p_h_pens, "away": p_a_pens},
        "advances":    {"home": p_home_advances, "away": p_away_advances},
        "xG_h": ninety["xG_h"],
        "xG_a": ninety["xG_a"],
    }


def predict_match(home, away, neutral, importance, df, mod_h, mod_a, rho, stage="group"):
    """
    Single entry point the UI/app should call. stage = one of:
    "group", "r32", "r16", "qf", "sf", "final"
    """
    if stage == "group":
        result = predict_fixture(home, away, neutral, importance, df, mod_h, mod_a, rho)
        return {"stage": "group", "result": result}
    else:
        result = predict_knockout(home, away, neutral, importance, df, mod_h, mod_a, rho)
        return {"stage": stage, "result": result}
    
if __name__ == "__main__":
    import pickle
    import pandas as pd
    from models.dixon_coles import load_data

    with open("models/dixon_coles_model.pkl", "rb") as f:
        saved = pickle.load(f)
    mod_h, mod_a, rho = saved["mod_h"], saved["mod_a"], saved["rho"]

    df = load_data()

    result = predict_match("Argentina", "France", neutral=True, importance=60,
                           df=df, mod_h=mod_h, mod_a=mod_a, rho=rho, stage="final")
    print(result)