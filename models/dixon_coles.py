"""
Dixon-Coles model for football match outcome prediction.
=========================================================

Architecture
------------
  1. Two Poisson GLMs  →  expected goals for home team (λ) and away team (μ)
  2. Dixon-Coles ρ     →  corrects the independence assumption for low-scoring
                          outcomes (0-0, 1-0, 0-1, 1-1) where correlation is
                          strongest
  3. Score-grid        →  P(home=i, away=j) for i,j ∈ 0..10, summed into
                          Win / Draw / Loss probabilities

Why Dixon-Coles > plain bivariate Poisson?
  - Independent Poisson systematically under-predicts draws (especially 0-0
    and 1-1) and over-predicts 1-0 / 0-1.  The ρ correction fixes this with
    a single extra parameter, adding minimal complexity.

Train set: all matches before 2022-01-01
Usage:     python -X utf8 -m models.dixon_coles
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import pickle, pathlib, warnings

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 25)
pd.set_option("display.max_colwidth", 22)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

MAX_GOALS    = 10                # score-grid truncation
TRAIN_CUTOFF = "2022-01-01"

# ── Feature sets for each sub-model ──────────────────────────────────
#   Named to match the columns produced by load_data().
#   Each list defines one Poisson GLM's right-hand side (before constant).

HOME_FEAT = [
    "elo_diff_sc",                # (elo_home − elo_away) / 400
    "home_goals_scored_last5",    # home team's attacking form
    "away_goals_conceded_last5",  # away team's defensive leakiness
    "home_win_rate_last5",        # home team's recent results
    "is_home",                    # 1 if home team genuinely at home
    "importance_sc",              # FIFA match importance / 10
]

AWAY_FEAT = [
    "elo_diff_sc",                # negative coeff expected
    "away_goals_scored_last5",
    "home_goals_conceded_last5",
    "away_win_rate_last5",
    "is_home",                    # negative coeff: home advantage hurts away
    "importance_sc",
]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv("features/match_features.csv", parse_dates=["date"])

    # Derived / scaled features
    df["is_home"]       = (~df["neutral"]).astype(float)
    df["elo_diff_sc"]   = df["elo_diff"] / 400.0
    df["importance_sc"] = df["importance"] / 10.0

    # Drop the small fraction of rows with NaN form features (teams' first
    # ever match — ~0.3 %)
    required = list(set(HOME_FEAT + AWAY_FEAT))
    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# DIXON-COLES τ  (low-score correction)
# ═══════════════════════════════════════════════════════════════════════

def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Scalar Dixon-Coles correction factor."""
    if   x == 0 and y == 0: return 1.0 - lam * mu * rho
    elif x == 1 and y == 0: return 1.0 + mu * rho
    elif x == 0 and y == 1: return 1.0 + lam * rho
    elif x == 1 and y == 1: return 1.0 - rho
    return 1.0


def _dc_tau_vec(hg, ag, lam, mu, rho):
    """Vectorised version for fast ρ fitting."""
    tau = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho
    return tau


# ═══════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════

def fit_glms(train: pd.DataFrame):
    """Fit two Poisson GLMs — one for home goals, one for away goals."""
    X_h = sm.add_constant(train[HOME_FEAT])
    X_a = sm.add_constant(train[AWAY_FEAT])
    mod_h = sm.GLM(train["home_score"], X_h, family=sm.families.Poisson()).fit()
    mod_a = sm.GLM(train["away_score"], X_a, family=sm.families.Poisson()).fit()
    return mod_h, mod_a


def fit_rho(train: pd.DataFrame, mod_h, mod_a) -> float:
    """Conditional MLE for Dixon-Coles ρ, holding λ/μ fixed.

    ρ is typically small and negative (around −0.03 to −0.10).  We use
    tight bounds to keep the optimizer in the feasible region where all
    τ values remain strictly positive.
    """
    X_h = sm.add_constant(train[HOME_FEAT])
    X_a = sm.add_constant(train[AWAY_FEAT])
    lam = mod_h.predict(X_h).values
    mu  = mod_a.predict(X_a).values
    hg  = train["home_score"].values.astype(float)
    ag  = train["away_score"].values.astype(float)

    # Poisson log-PMFs (independent of ρ)
    log_pois = (np.log(np.maximum(poisson.pmf(hg, lam), 1e-20))
              + np.log(np.maximum(poisson.pmf(ag, mu),  1e-20)))

    # Upper bound: ensure τ(0,0) = 1 − λμρ > 0  ⟹  ρ < 1/(max(λμ))
    max_lam_mu = np.max(lam * mu)
    upper = min(0.15, 0.95 / max_lam_mu)   # leave 5% margin
    # Lower bound: ensure τ(0,1) = 1 + λρ > 0  ⟹  ρ > −1/max(λ)
    lower = max(-0.15, -0.95 / np.max(lam))

    def neg_ll(rho):
        tau = _dc_tau_vec(hg, ag, lam, mu, rho)
        min_tau = tau.min()
        if min_tau <= 1e-10:
            return 1e15
        return -(log_pois + np.log(tau)).sum()

    res = minimize_scalar(neg_ll, bounds=(lower, upper), method="bounded")
    return res.x


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def scoreline_grid(lam: float, mu: float, rho: float,
                   max_g: int = MAX_GOALS) -> np.ndarray:
    """Return (max_g+1 × max_g+1) probability matrix P(home=i, away=j)."""
    grid = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        p_i = poisson.pmf(i, lam)
        for j in range(max_g + 1):
            grid[i, j] = p_i * poisson.pmf(j, mu) * _dc_tau(i, j, lam, mu, rho)
    grid /= grid.sum()          # normalise (truncation correction)
    return grid


def wdl_from_grid(grid: np.ndarray):
    """Extract win / draw / loss probs + most-likely score."""
    p_h = np.tril(grid, -1).sum()     # home_goals > away_goals
    p_d = np.trace(grid)              # home_goals == away_goals
    p_a = np.triu(grid,  1).sum()     # home_goals < away_goals
    idx = np.unravel_index(grid.argmax(), grid.shape)
    return p_h, p_d, p_a, f"{idx[0]}-{idx[1]}", grid[idx]


def predict_from_row(row, mod_h, mod_a, rho):
    """Predict a single match from a feature-table row."""
    x_h = np.array([[1.0] + [row[f] for f in HOME_FEAT]])
    x_a = np.array([[1.0] + [row[f] for f in AWAY_FEAT]])
    lam = mod_h.predict(x_h)[0]
    mu  = mod_a.predict(x_a)[0]
    grid = scoreline_grid(lam, mu, rho)
    p_h, p_d, p_a, ml_sc, ml_p = wdl_from_grid(grid)
    return dict(xG_h=lam, xG_a=mu, p_h=p_h, p_d=p_d, p_a=p_a,
                ml_score=ml_sc, ml_prob=ml_p)


# ── Look up latest team stats for ad-hoc predictions ─────────────────

def _latest_team_stats(df: pd.DataFrame, team: str) -> dict:
    """Extract approximate current features for *team* from its most recent
    appearance in the feature table (either as home or away)."""
    h = df.loc[df["home_team"] == team]
    a = df.loc[df["away_team"] == team]
    last_h = h["date"].max() if len(h) else pd.Timestamp.min
    last_a = a["date"].max() if len(a) else pd.Timestamp.min

    if last_h >= last_a and len(h):
        r = h.loc[h["date"] == last_h].iloc[-1]
        return dict(elo=r["elo_home"], gs5=r["home_goals_scored_last5"],
                    gc5=r["home_goals_conceded_last5"],
                    wr5=r["home_win_rate_last5"])
    elif len(a):
        r = a.loc[a["date"] == last_a].iloc[-1]
        return dict(elo=r["elo_away"], gs5=r["away_goals_scored_last5"],
                    gc5=r["away_goals_conceded_last5"],
                    wr5=r["away_win_rate_last5"])
    return dict(elo=1500.0, gs5=1.2, gc5=1.2, wr5=0.33)


def predict_fixture(home: str, away: str, neutral: bool, importance: float,
                    df, mod_h, mod_a, rho):
    """Predict a future / hypothetical match by looking up latest team stats."""
    hs = _latest_team_stats(df, home)
    aws = _latest_team_stats(df, away)

    feat = dict(
        elo_diff_sc              = (hs["elo"] - aws["elo"]) / 400.0,
        home_goals_scored_last5  = hs["gs5"],
        away_goals_conceded_last5= aws["gc5"],
        home_win_rate_last5      = hs["wr5"],
        away_goals_scored_last5  = aws["gs5"],
        home_goals_conceded_last5= hs["gc5"],
        away_win_rate_last5      = aws["wr5"],
        is_home                  = 0.0 if neutral else 1.0,
        importance_sc            = importance / 10.0,
    )

    x_h = np.array([[1.0] + [feat[f] for f in HOME_FEAT]])
    x_a = np.array([[1.0] + [feat[f] for f in AWAY_FEAT]])
    lam = mod_h.predict(x_h)[0]
    mu  = mod_a.predict(x_a)[0]
    grid = scoreline_grid(lam, mu, rho)
    p_h, p_d, p_a, ml_sc, ml_p = wdl_from_grid(grid)
    return dict(xG_h=lam, xG_a=mu, p_h=p_h, p_d=p_d, p_a=p_a,
                ml_score=ml_sc, ml_prob=ml_p)


# ═══════════════════════════════════════════════════════════════════════
# MAIN — fit, summarise, sanity-check
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Load ─────────────────────────────────────────────────────────
    df    = load_data()
    train = df[df["date"] < TRAIN_CUTOFF].copy()
    test  = df[df["date"] >= TRAIN_CUTOFF].copy()
    print(f"Train : {len(train):>7,} matches  (< {TRAIN_CUTOFF})")
    print(f"Test  : {len(test):>7,} matches  (>= {TRAIN_CUTOFF})")

    # ── Fit ──────────────────────────────────────────────────────────
    print("\n— Fitting Poisson GLMs …")
    mod_h, mod_a = fit_glms(train)

    print("\n" + "=" * 90)
    print("HOME GOALS MODEL  (λ)")
    print("=" * 90)
    print(mod_h.summary2().tables[1].to_string())

    print("\n" + "=" * 90)
    print("AWAY GOALS MODEL  (μ)")
    print("=" * 90)
    print(mod_a.summary2().tables[1].to_string())

    print("\n— Fitting Dixon-Coles ρ …")
    rho = fit_rho(train, mod_h, mod_a)
    print(f"   ρ = {rho:.5f}   (negative → boosts low-score draws, as expected)")

    # ── Save model artefacts ─────────────────────────────────────────
    out_dir = pathlib.Path("models")
    with open(out_dir / "dixon_coles_model.pkl", "wb") as f:
        pickle.dump({"mod_h": mod_h, "mod_a": mod_a, "rho": rho,
                      "home_feat": HOME_FEAT, "away_feat": AWAY_FEAT}, f)
    print(f"   Model saved → {out_dir / 'dixon_coles_model.pkl'}")

    # ══════════════════════════════════════════════════════════════════
    # SANITY CHECK  —  predict specific historical matches from the
    # test set and compare with actual outcomes
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("SANITY CHECK — individual match predictions  (test set)")
    print("=" * 90)

    target_matches = [
        # 2022 World Cup
        ("Argentina",  "Saudi Arabia",   2022),
        ("Spain",      "Costa Rica",     2022),
        ("Germany",    "Japan",          2022),
        ("Morocco",    "Belgium",        2022),
        ("Argentina",  "France",         2022),   # WC Final
        ("Brazil",     "South Korea",    2022),
        # 2026 World Cup (already played)
        ("Spain",      "Austria",        2026),
        ("Portugal",   "Croatia",        2026),
    ]

    rows = []
    for t_home, t_away, year in target_matches:
        # Search both orderings
        for h, a in [(t_home, t_away), (t_away, t_home)]:
            mask = ((df["home_team"] == h) & (df["away_team"] == a)
                    & (df["date"].dt.year == year)
                    & (df["date"] >= TRAIN_CUTOFF))
            hits = df[mask]
            if len(hits):
                break
        if len(hits) == 0:
            continue

        r = hits.iloc[-1]
        p = predict_from_row(r, mod_h, mod_a, rho)

        # Determine which prob the model favoured
        probs  = {"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}
        pred   = max(probs, key=probs.get)
        actual = r["outcome"]

        rows.append({
            "Date":    r["date"].strftime("%Y-%m-%d"),
            "Match":   f"{r['home_team']} v {r['away_team']}",
            "xG(H)":  f"{p['xG_h']:.2f}",
            "xG(A)":  f"{p['xG_a']:.2f}",
            "P(H)":   f"{p['p_h']:.1%}",
            "P(D)":   f"{p['p_d']:.1%}",
            "P(A)":   f"{p['p_a']:.1%}",
            "Pred":   f"{pred} ({p['ml_score']})",
            "Actual":  f"{int(r['home_score'])}-{int(r['away_score'])} ({actual})",
            "✓":       "✓" if pred == actual else "",
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    # ── Quick test-set accuracy ──────────────────────────────────────
    print("\n— Quick overall accuracy on test set …")
    correct = 0
    total   = 0
    for _, r in test.iterrows():
        p = predict_from_row(r, mod_h, mod_a, rho)
        probs = {"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}
        pred  = max(probs, key=probs.get)
        if pred == r["outcome"]:
            correct += 1
        total += 1
    print(f"   {correct}/{total} = {correct/total:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # UPCOMING 2026 FIXTURES  (bonus preview)
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("UPCOMING 2026 WC FIXTURE PREDICTIONS")
    print("=" * 90)

    fixtures = pd.read_csv("data/processed/future_fixtures.csv",
                           parse_dates=["date"])
    fixture_rows = []
    for _, fx in fixtures.iterrows():
        p = predict_fixture(fx["home_team"], fx["away_team"],
                            fx["neutral"], 50,       # WC importance = 50
                            df, mod_h, mod_a, rho)
        probs = {"H": p["p_h"], "D": p["p_d"], "A": p["p_a"]}
        fav   = max(probs, key=probs.get)
        fixture_rows.append({
            "Date":   fx["date"].strftime("%m-%d"),
            "Match":  f"{fx['home_team']} v {fx['away_team']}",
            "Venue":  "N" if fx["neutral"] else "H",
            "xG(H)":  f"{p['xG_h']:.2f}",
            "xG(A)":  f"{p['xG_a']:.2f}",
            "P(H)":  f"{p['p_h']:.1%}",
            "P(D)":  f"{p['p_d']:.1%}",
            "P(A)":  f"{p['p_a']:.1%}",
            "Pick":  f"{fav} ({p['ml_score']})",
        })
    print(pd.DataFrame(fixture_rows).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
