"""
Feature engineering pipeline for WC Predictor.

Computes (all in chronological order, using only pre-match information):
  1. Rolling Elo ratings (FIFA-style with importance weights & shootout convention)
  2. Recent form: rolling avg goals scored/conceded & win rate (last 5 / last 10)
  3. Head-to-head historical record between the two teams

Usage:
    python -m features.build_features
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_ELO = 1500
ELO_SCALING = 600          # FIFA uses 600 in the exponent denominator
FORM_WINDOWS = (5, 10)     # rolling windows for form features

# Dates & teams to snapshot for sanity-checking
SNAPSHOT_DATES = [
    "1950-07-01",   # first post-war World Cup
    "1970-06-21",   # Brazil's Pelé-era peak
    "1998-07-12",   # France wins at home
    "2014-07-13",   # Germany's 4th star
    "2022-12-18",   # Argentina wins in Qatar
    "2026-06-15",   # current WC group stage
]
SNAPSHOT_TEAMS = ["Brazil", "Germany", "Argentina", "France", "England", "Spain"]


# ═════════════════════════════════════════════════════════════════════════
# TOURNAMENT → FIFA IMPORTANCE WEIGHT
# ═════════════════════════════════════════════════════════════════════════

def _build_importance_mapper():
    """Return a function that maps a tournament name → int importance value.

    Rules are evaluated in order; first match wins.  The fallback for
    anything not matched is 20 (mid-tier), per the user's request.
    """
    # (predicate on lowercased name, importance value)
    rules = [
        # ── FIFA World Cup (final competition) ──────────────────────
        (lambda t: t == "fifa world cup", 50),

        # ── Qualifiers (WC & confederation) ─────────────────────────
        (lambda t: "qualification" in t or "qualifying" in t, 25),

        # ── Friendlies ──────────────────────────────────────────────
        (lambda t: "friendly" in t, 10),

        # ── Nations League ──────────────────────────────────────────
        (lambda t: "nations league" in t, 15),

        # ── FIFA Confederations Cup ─────────────────────────────────
        (lambda t: "confederations cup" in t, 40),

        # ── Major continental final competitions ────────────────────
        (lambda t: any(kw in t for kw in (
            "uefa euro", "european championship",
            "copa amé", "copa ame", "copa america",
            "african cup of nations", "africa cup of nations",
            "afc asian cup",
            "gold cup", "concacaf championship",
            "oceania nations cup", "ofc nations cup",
        )), 35),

        # ── Minor/regional cups & tournaments (catch-all for named
        #    competitions that are not qualifiers or friendlies) ─────
        (lambda t: any(kw in t for kw in (
            "cup", "championship", "games", "trophy", "tournament",
        )), 20),
    ]

    def _map(tournament_name: str) -> int:
        t = tournament_name.strip().lower()
        for pred, val in rules:
            if pred(t):
                return val
        return 20   # default mid-tier

    return _map


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _expected_score(r_self: float, r_opp: float) -> float:
    """FIFA-style expected result:  We = 1 / (10^(-dr/600) + 1)."""
    return 1.0 / (10.0 ** (-(r_self - r_opp) / ELO_SCALING) + 1.0)


def _rolling_form(history: list, n: int):
    """Mean goals-scored, goals-conceded, win-value over last *n* entries.

    Each entry in *history* is a (gs, gc, win_val) tuple.
    Returns (NaN, NaN, NaN) if history is empty.
    """
    window = history[-n:]
    if not window:
        return np.nan, np.nan, np.nan
    gs = sum(e[0] for e in window) / len(window)
    gc = sum(e[1] for e in window) / len(window)
    wr = sum(e[2] for e in window) / len(window)
    return gs, gc, wr


# ═════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def build_features() -> pd.DataFrame:
    # ── Load & sort ──────────────────────────────────────────────────
    print("Loading data …")
    matches = pd.read_csv("data/processed/matches_full.csv", parse_dates=["date"])
    matches.sort_values("date", inplace=True)
    matches.reset_index(drop=True, inplace=True)
    N = len(matches)
    print(f"  {N:,} matches  |  {matches['date'].min().date()} – {matches['date'].max().date()}")

    # Sanity: shootout matches — most should have tied scores
    so_mask = matches["went_to_shootout"]
    if so_mask.any():
        tied = (matches.loc[so_mask, "home_score"]
                == matches.loc[so_mask, "away_score"])
        n_not_tied = (~tied).sum()
        if n_not_tied > 0:
            print(f"  NOTE: {n_not_tied}/{so_mask.sum()} shootout matches have "
                  f"non-tied scores (older two-leg aggregates) — handled correctly")
        else:
            print(f"  {so_mask.sum()} shootout matches – all tied ✓")

    # ── Map tournament → importance ──────────────────────────────────
    imp_fn = _build_importance_mapper()
    matches["importance"] = matches["tournament"].apply(imp_fn)
    print("\nImportance distribution:")
    print(matches["importance"].value_counts().sort_index().to_string())

    # ── Allocate output arrays ───────────────────────────────────────
    elo_h   = np.empty(N, dtype=np.float64)
    elo_a   = np.empty(N, dtype=np.float64)
    elo_d   = np.empty(N, dtype=np.float64)

    hgs5  = np.full(N, np.nan);  hgc5  = np.full(N, np.nan)
    hgs10 = np.full(N, np.nan);  hgc10 = np.full(N, np.nan)
    hwr5  = np.full(N, np.nan);  hwr10 = np.full(N, np.nan)

    ags5  = np.full(N, np.nan);  agc5  = np.full(N, np.nan)
    ags10 = np.full(N, np.nan);  agc10 = np.full(N, np.nan)
    awr5  = np.full(N, np.nan);  awr10 = np.full(N, np.nan)

    h2ht  = np.zeros(N, dtype=np.int32)
    h2hw  = np.zeros(N, dtype=np.int32)
    h2haw = np.zeros(N, dtype=np.int32)
    h2hd  = np.zeros(N, dtype=np.int32)
    h2hp  = np.full(N, np.nan)

    # ── State dictionaries ───────────────────────────────────────────
    team_elo:     dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    team_history: dict[str, list]  = defaultdict(list)   # [(gs, gc, win_val), …]
    # h2h key = canonical pair tuple; value = {teamA_w, teamB_w, d}
    h2h_rec: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    # For Elo snapshots
    snap_dates = pd.to_datetime(SNAPSHOT_DATES)
    snap_idx = 0
    snapshots: dict[str, dict] = {}

    # ── Pre-extract column arrays for speed ──────────────────────────
    col_date    = matches["date"].values
    col_home    = matches["home_team"].values
    col_away    = matches["away_team"].values
    col_hs      = matches["home_score"].values.astype(np.int32)
    col_as      = matches["away_score"].values.astype(np.int32)
    col_so      = matches["went_to_shootout"].values
    col_sow     = matches["shootout_winner"].values
    col_imp     = matches["importance"].values.astype(np.float64)

    # ── Main chronological loop ──────────────────────────────────────
    print("\nProcessing …")
    t0 = time.time()

    for i in range(N):
        dt   = col_date[i]
        home = col_home[i]
        away = col_away[i]
        hs   = int(col_hs[i])
        as_  = int(col_as[i])
        so   = bool(col_so[i])
        sow  = col_sow[i]       # str or NaN
        imp  = col_imp[i]

        # ── Snapshot capture ─────────────────────────────────────────
        while snap_idx < len(snap_dates) and pd.Timestamp(dt) >= snap_dates[snap_idx]:
            label = str(snap_dates[snap_idx].date())
            snapshots[label] = {
                t: round(team_elo[t], 1)
                for t in SNAPSHOT_TEAMS if t in team_elo
            }
            snap_idx += 1

        # ── PRE-MATCH: record Elo ────────────────────────────────────
        rh = team_elo[home]
        ra = team_elo[away]
        elo_h[i] = rh
        elo_a[i] = ra
        elo_d[i] = rh - ra

        # ── PRE-MATCH: form ──────────────────────────────────────────
        hgs5[i],  hgc5[i],  hwr5[i]  = _rolling_form(team_history[home], 5)
        hgs10[i], hgc10[i], hwr10[i] = _rolling_form(team_history[home], 10)
        ags5[i],  agc5[i],  awr5[i]  = _rolling_form(team_history[away], 5)
        ags10[i], agc10[i], awr10[i] = _rolling_form(team_history[away], 10)

        # ── PRE-MATCH: head-to-head ──────────────────────────────────
        pair = (min(home, away), max(home, away))
        rec  = h2h_rec[pair]
        hw_cnt  = rec.get(f"{home}_w", 0)
        aw_cnt  = rec.get(f"{away}_w", 0)
        d_cnt   = rec.get("d", 0)
        tot     = hw_cnt + aw_cnt + d_cnt
        h2ht[i]  = tot
        h2hw[i]  = hw_cnt
        h2haw[i] = aw_cnt
        h2hd[i]  = d_cnt
        h2hp[i]  = hw_cnt / tot if tot > 0 else np.nan

        # ══════════════════════════════════════════════════════════════
        # POST-MATCH UPDATES
        # ══════════════════════════════════════════════════════════════

        # ── Determine actual result values ───────────────────────────
        if so and isinstance(sow, str):
            # Penalty-shootout convention: winner 0.75, loser 0.5
            w_h  = 0.75 if sow == home else 0.5
            w_a  = 0.75 if sow == away else 0.5
            fw_h = 1.0  if sow == home else 0.0    # form: binary win
            fw_a = 1.0  if sow == away else 0.0
            h2h_key = f"{sow}_w"
        elif hs > as_:
            w_h, w_a = 1.0, 0.0
            fw_h, fw_a = 1.0, 0.0
            h2h_key = f"{home}_w"
        elif hs < as_:
            w_h, w_a = 0.0, 1.0
            fw_h, fw_a = 0.0, 1.0
            h2h_key = f"{away}_w"
        else:
            w_h, w_a = 0.5, 0.5
            fw_h, fw_a = 0.5, 0.5
            h2h_key = "d"

        # ── Elo update ───────────────────────────────────────────────
        we_h = _expected_score(rh, ra)
        team_elo[home] = rh + imp * (w_h - we_h)
        team_elo[away] = ra + imp * (w_a - (1.0 - we_h))

        # ── Form history update ──────────────────────────────────────
        team_history[home].append((hs, as_, fw_h))
        team_history[away].append((as_, hs, fw_a))

        # ── H2H update ──────────────────────────────────────────────
        h2h_rec[pair][h2h_key] = rec.get(h2h_key, 0) + 1

        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:>6,}/{N:,} …")

    elapsed = time.time() - t0
    print(f"  {N:,}/{N:,} done in {elapsed:.1f}s")

    # Capture any remaining snapshots past the last match
    for si in range(snap_idx, len(snap_dates)):
        label = str(snap_dates[si].date())
        snapshots[label] = {t: round(team_elo[t], 1) for t in SNAPSHOT_TEAMS}

    # ══════════════════════════════════════════════════════════════════
    # ASSEMBLE FEATURE TABLE
    # ══════════════════════════════════════════════════════════════════

    feat = matches.drop(columns=["home_elo", "away_elo"]).copy()

    feat["elo_home"]  = np.round(elo_h, 1)
    feat["elo_away"]  = np.round(elo_a, 1)
    feat["elo_diff"]  = np.round(elo_d, 1)

    feat["home_goals_scored_last5"]    = np.round(hgs5,  2)
    feat["home_goals_conceded_last5"]  = np.round(hgc5,  2)
    feat["home_goals_scored_last10"]   = np.round(hgs10, 2)
    feat["home_goals_conceded_last10"] = np.round(hgc10, 2)
    feat["home_win_rate_last5"]        = np.round(hwr5,  3)
    feat["home_win_rate_last10"]       = np.round(hwr10, 3)

    feat["away_goals_scored_last5"]    = np.round(ags5,  2)
    feat["away_goals_conceded_last5"]  = np.round(agc5,  2)
    feat["away_goals_scored_last10"]   = np.round(ags10, 2)
    feat["away_goals_conceded_last10"] = np.round(agc10, 2)
    feat["away_win_rate_last5"]        = np.round(awr5,  3)
    feat["away_win_rate_last10"]       = np.round(awr10, 3)

    feat["h2h_total"]        = h2ht
    feat["h2h_home_wins"]    = h2hw
    feat["h2h_away_wins"]    = h2haw
    feat["h2h_draws"]        = h2hd
    feat["h2h_home_win_pct"] = np.round(h2hp, 3)

    # Target variable (final match result including shootout)
    feat["outcome"] = np.where(
        feat["went_to_shootout"],
        np.where(feat["shootout_winner"] == feat["home_team"], "H", "A"),
        np.where(feat["home_score"] > feat["away_score"], "H",
                 np.where(feat["home_score"] < feat["away_score"], "A", "D")),
    )

    return feat, snapshots, dict(team_elo)


# ═════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    feat, snapshots, final_elos = build_features()

    # ── Save ─────────────────────────────────────────────────────────
    out = "features/match_features.csv"
    feat.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({feat.shape[0]:,} rows × {feat.shape[1]} cols)")

    # ── First 20 rows ────────────────────────────────────────────────
    show_cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "importance",
        "elo_home", "elo_away", "elo_diff",
        "home_goals_scored_last5", "away_goals_scored_last5",
        "home_win_rate_last5", "away_win_rate_last5",
        "h2h_total", "h2h_home_wins", "outcome",
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 18)
    print("\n" + "=" * 100)
    print("FIRST 20 ROWS  (selected columns)")
    print("=" * 100)
    print(feat[show_cols].head(20).to_string())

    # ── Elo snapshots ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ELO SNAPSHOTS — sanity check")
    print("=" * 100)
    snap_df = pd.DataFrame(snapshots).T
    snap_df.index.name = "date"
    print(snap_df.to_string())

    # ── Final computed Elo vs. provided current Elo ──────────────────
    print("\n" + "=" * 100)
    print("FINAL COMPUTED ELO  (top 25)")
    print("=" * 100)
    final = pd.Series(final_elos).sort_values(ascending=False).head(25)
    print(final.round(0).astype(int).to_string())

    # ── Compare with elo_ratings.csv ─────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPUTED vs PROVIDED Elo  (fixture teams)")
    print("=" * 100)
    provided = pd.read_csv("data/raw/elo_ratings.csv")
    fixture_teams = [
        "Australia", "Egypt", "Argentina", "Cape Verde", "Colombia",
        "Ghana", "Canada", "Morocco", "Paraguay", "France",
        "Brazil", "Norway", "Mexico", "England", "Portugal",
        "Spain", "United States", "Belgium",
    ]
    comp = provided[provided["team"].isin(fixture_teams)][["team", "elo_rating"]].copy()
    comp["computed_elo"] = comp["team"].map(final_elos).round(0).astype(int)
    comp["diff"] = comp["computed_elo"] - comp["elo_rating"]
    comp.sort_values("elo_rating", ascending=False, inplace=True)
    print(comp.to_string(index=False))
