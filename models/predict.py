"""
Match prediction API with knockout-stage layering.
====================================================

Architecture
------------
  ┌──────────────────────────────────────────────────────┐
  │ Layer 1 — 90-minute regulation                       │
  │   Dixon-Coles bivariate Poisson → P(A) / P(D) / P(B)│
  ├──────────────────────────────────────────────────────┤
  │ Layer 2 — Extra time  (30 min ≈ 1/3 of 90 min)      │
  │   Scaled Poisson (λ/3, μ/3) → P(A) / P(D) / P(B)   │
  ├──────────────────────────────────────────────────────┤
  │ Layer 3 — Penalty shootout                           │
  │   50/50 baseline  (refinable from shootouts.csv)     │
  └──────────────────────────────────────────────────────┘

  Group stage  → returns 3-way regulation result only
  Knockout     → full layered breakdown with path & advancement probs

Usage:
    python -X utf8 -m models.predict
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from models.dixon_coles import (
    scoreline_grid, wdl_from_grid, _latest_team_stats,
    HOME_FEAT, AWAY_FEAT,
)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Stage → FIFA match-importance weight
STAGE_IMPORTANCE: dict[str, int] = {
    "group": 50,        # FIFA World Cup (up to QF)
    "r32":   50,
    "r16":   50,
    "qf":    60,        # FIFA World Cup (QF onward)
    "sf":    60,
    "final": 60,
}

KNOCKOUT_STAGES = frozenset({"r32", "r16", "qf", "sf", "final"})

ET_SCALE   = 1.0 / 3.0     # 30 min extra time / 90 min regulation
PK_DEFAULT = 0.50           # baseline shoot-out win probability


# ═══════════════════════════════════════════════════════════════════════
# PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════

class MatchPredictor:
    """High-level match predictor with knockout-stage support.

    Wraps the fitted Dixon-Coles model and exposes a single
    ``predict_match(team_a, team_b, stage)`` entry-point that
    returns a structured dict with per-layer probabilities.
    """

    def __init__(
        self,
        model_path:    str = "models/dixon_coles_model.pkl",
        features_path: str = "features/match_features.csv",
    ):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.mod_h = bundle["mod_h"]
        self.mod_a = bundle["mod_a"]
        self.rho   = bundle["rho"]

        # Feature table — used only for _latest_team_stats() lookups
        self.df = pd.read_csv(features_path, parse_dates=["date"])
        self.df["is_home"]       = (~self.df["neutral"]).astype(float)
        self.df["elo_diff_sc"]   = self.df["elo_diff"] / 400.0
        self.df["importance_sc"] = self.df["importance"] / 10.0

    # ── internal helpers ─────────────────────────────────────────────

    def _xg(self, team_a: str, team_b: str,
            neutral: bool, importance: int) -> tuple[float, float]:
        """Expected goals (λ for team_a, μ for team_b)."""
        sa = _latest_team_stats(self.df, team_a)
        sb = _latest_team_stats(self.df, team_b)

        feat = {
            "elo_diff_sc":               (sa["elo"] - sb["elo"]) / 400.0,
            "home_goals_scored_last5":   sa["gs5"],
            "away_goals_conceded_last5": sb["gc5"],
            "home_win_rate_last5":       sa["wr5"],
            "away_goals_scored_last5":   sb["gs5"],
            "home_goals_conceded_last5": sa["gc5"],
            "away_win_rate_last5":       sb["wr5"],
            "is_home":                   0.0 if neutral else 1.0,
            "importance_sc":             importance / 10.0,
        }
        x_h = np.array([[1.0] + [feat[f] for f in HOME_FEAT]])
        x_a = np.array([[1.0] + [feat[f] for f in AWAY_FEAT]])
        return float(self.mod_h.predict(x_h)[0]), float(self.mod_a.predict(x_a)[0])

    def _layer_90(self, lam: float, mu: float) -> dict:
        """Layer 1 — 90-minute regulation."""
        grid = scoreline_grid(lam, mu, self.rho)
        p_a, p_d, p_b, ml, _ = wdl_from_grid(grid)
        return dict(xg_a=round(lam, 3), xg_b=round(mu, 3),
                    p_a_win=round(p_a, 4), p_draw=round(p_d, 4),
                    p_b_win=round(p_b, 4), most_likely=ml)

    def _layer_et(self, lam: float, mu: float) -> dict:
        """Layer 2 — extra time (30 min, scaled xG)."""
        lam_et, mu_et = lam * ET_SCALE, mu * ET_SCALE
        grid = scoreline_grid(lam_et, mu_et, self.rho)
        p_a, p_d, p_b, _, _ = wdl_from_grid(grid)
        return dict(xg_a=round(lam_et, 3), xg_b=round(mu_et, 3),
                    p_a_win=round(p_a, 4), p_draw=round(p_d, 4),
                    p_b_win=round(p_b, 4))

    @staticmethod
    def _layer_pk() -> dict:
        """Layer 3 — penalty shootout (50/50 baseline)."""
        return dict(p_a_win=PK_DEFAULT, p_b_win=1.0 - PK_DEFAULT)

    # ── public API ───────────────────────────────────────────────────

    def predict_match(
        self,
        team_a:  str,
        team_b:  str,
        stage:   str,
        neutral: bool = True,
    ) -> dict:
        """Predict a match with per-layer breakdown.

        Parameters
        ----------
        team_a, team_b : str
            Team names.  *team_a* occupies the model's "home" slot.
        stage : str
            ``group | r32 | r16 | qf | sf | final``
        neutral : bool
            ``True`` for a neutral venue (default for most WC matches).

        Returns
        -------
        dict
            Always contains ``team_a, team_b, stage, neutral, regulation``.
            For knockout stages also: ``extra_time, penalties, paths,
            advancement``.
        """
        if stage not in STAGE_IMPORTANCE:
            raise ValueError(
                f"Unknown stage '{stage}'. "
                f"Choose from: {', '.join(STAGE_IMPORTANCE)}")

        importance = STAGE_IMPORTANCE[stage]
        lam, mu = self._xg(team_a, team_b, neutral, importance)

        # ── Layer 1 ──────────────────────────────────────────────
        reg = self._layer_90(lam, mu)

        result = dict(team_a=team_a, team_b=team_b,
                      stage=stage, neutral=neutral,
                      regulation=reg)

        if stage not in KNOCKOUT_STAGES:
            return result                       # group → 3-way only

        # ── Layer 2 ──────────────────────────────────────────────
        et = self._layer_et(lam, mu)
        result["extra_time"] = et

        # ── Layer 3 ──────────────────────────────────────────────
        pk = self._layer_pk()
        result["penalties"] = pk

        # ── Path probabilities (unconditional) ───────────────────
        p_d_90 = reg["p_draw"]
        p_d_et = et["p_draw"]

        a_in_90 = reg["p_a_win"]
        a_in_et = p_d_90 * et["p_a_win"]
        a_in_pk = p_d_90 * p_d_et * pk["p_a_win"]

        b_in_90 = reg["p_b_win"]
        b_in_et = p_d_90 * et["p_b_win"]
        b_in_pk = p_d_90 * p_d_et * pk["p_b_win"]

        result["paths"] = dict(
            a_in_90=round(a_in_90, 4), a_in_et=round(a_in_et, 4),
            a_in_pk=round(a_in_pk, 4),
            b_in_90=round(b_in_90, 4), b_in_et=round(b_in_et, 4),
            b_in_pk=round(b_in_pk, 4),
        )

        result["advancement"] = dict(
            p_a=round(a_in_90 + a_in_et + a_in_pk, 4),
            p_b=round(b_in_90 + b_in_et + b_in_pk, 4),
        )

        return result


# ═══════════════════════════════════════════════════════════════════════
# PRETTY DISPLAY
# ═══════════════════════════════════════════════════════════════════════

STAGE_LABELS = {
    "group": "GROUP", "r32": "ROUND OF 32", "r16": "ROUND OF 16",
    "qf": "QUARTER-FINAL", "sf": "SEMI-FINAL", "final": "FINAL",
}

def display(pred: dict) -> str:
    """Render a prediction dict as a formatted string."""
    a, b = pred["team_a"], pred["team_b"]
    label = STAGE_LABELS.get(pred["stage"], pred["stage"].upper())
    venue = "neutral" if pred["neutral"] else f"{a} home"
    r = pred["regulation"]
    ko = pred["stage"] in KNOCKOUT_STAGES

    L = []
    L.append("")
    L.append(f"  {label}:  {a}  vs  {b}   ({venue})")
    L.append("  " + "─" * 58)

    # 90 min
    L.append(f"  90 MINUTES            xG  {r['xg_a']:.2f}  —  {r['xg_b']:.2f}")
    L.append(f"  ├─ {a:<22s} wins  {r['p_a_win']:6.1%}")
    draw_tag = "  → ET" if ko else ""
    L.append(f"  ├─ {'Draw':<22s}       {r['p_draw']:6.1%}{draw_tag}")
    L.append(f"  └─ {b:<22s} wins  {r['p_b_win']:6.1%}")

    if not ko:
        return "\n".join(L)

    et  = pred["extra_time"]
    pk  = pred["penalties"]
    pa  = pred["paths"]
    adv = pred["advancement"]

    # ET
    L.append("")
    L.append(f"  EXTRA TIME            xG  {et['xg_a']:.2f}  —  {et['xg_b']:.2f}")
    L.append(f"  ├─ {a:<22s} wins  {et['p_a_win']:6.1%}")
    L.append(f"  ├─ {'Still drawn':<22s}       {et['p_draw']:6.1%}  → PK")
    L.append(f"  └─ {b:<22s} wins  {et['p_b_win']:6.1%}")

    # PK
    L.append("")
    L.append(f"  PENALTIES")
    L.append(f"  ├─ {a:<22s} wins  {pk['p_a_win']:6.1%}")
    L.append(f"  └─ {b:<22s} wins  {pk['p_b_win']:6.1%}")

    # Advancement
    L.append("")
    L.append("  " + "═" * 58)
    L.append(f"  ADVANCES")
    L.append(f"  ├─ {a:<22s}      {adv['p_a']:6.1%}"
             f"   (90m {pa['a_in_90']:.1%}"
             f" + ET {pa['a_in_et']:.1%}"
             f" + PK {pa['a_in_pk']:.1%})")
    L.append(f"  └─ {b:<22s}      {adv['p_b']:6.1%}"
             f"   (90m {pa['b_in_90']:.1%}"
             f" + ET {pa['b_in_et']:.1%}"
             f" + PK {pa['b_in_pk']:.1%})")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# MAIN — demo
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading model …")
    mp = MatchPredictor()
    print(f"Model loaded.  rho = {mp.rho:.5f}\n")

    print("=" * 62)
    print("  DEMO: GROUP STAGE (3-way)")
    print("=" * 62)

    for a, b, neutral in [
        ("Argentina",     "Cape Verde", True),
        ("Mexico",        "England",    False),
        ("United States", "Belgium",    False),
    ]:
        pred = mp.predict_match(a, b, "group", neutral=neutral)
        print(display(pred))

    print("\n\n" + "=" * 62)
    print("  DEMO: KNOCKOUT STAGES (layered)")
    print("=" * 62)

    for a, b, stage, neutral in [
        ("Argentina",     "France",   "final", True),
        ("Portugal",      "Spain",    "qf",    True),
        ("Brazil",        "Norway",   "r16",   True),
        ("Mexico",        "England",  "sf",    False),
        ("United States", "Morocco",  "qf",    True),
    ]:
        pred = mp.predict_match(a, b, stage, neutral=neutral)
        print(display(pred))
        print()

    # ── Verify probabilities sum to 1 ────────────────────────────
    print("─" * 62)
    print("Sanity: advancement probabilities sum to 1.0?")
    for a, b, stage in [("Argentina", "France", "final"),
                        ("Brazil", "Norway", "r16")]:
        p = mp.predict_match(a, b, stage)
        total = p["advancement"]["p_a"] + p["advancement"]["p_b"]
        print(f"  {a} vs {b} ({stage}): "
              f"{p['advancement']['p_a']:.4f} + "
              f"{p['advancement']['p_b']:.4f} = {total:.4f}")


if __name__ == "__main__":
    main()
