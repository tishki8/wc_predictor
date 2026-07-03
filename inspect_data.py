"""
Data quality inspection script for WC Predictor.
Loads all CSV files and reports schema, stats, and quality issues.
"""

import pandas as pd
import numpy as np
import sys

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 160)

# ── 1. matches_full.csv ─────────────────────────────────────────────
print("=" * 80)
print("1. MATCHES_FULL.CSV")
print("=" * 80)

df = pd.read_csv("data/processed/matches_full.csv", parse_dates=["date"])

print(f"\nShape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")
print(f"\nUnique tournaments ({df['tournament'].nunique()}):")
print(df['tournament'].value_counts().head(20))
print(f"\nUnique teams (home): {df['home_team'].nunique()}")
print(f"Unique teams (away): {df['away_team'].nunique()}")
all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
print(f"Unique teams (total): {len(all_teams)}")

# Score distributions
print(f"\nhome_score stats:\n{df['home_score'].describe()}")
print(f"\naway_score stats:\n{df['away_score'].describe()}")

# Check for negative scores
neg_scores = df[(df['home_score'] < 0) | (df['away_score'] < 0)]
if len(neg_scores) > 0:
    print(f"\n⚠️  NEGATIVE SCORES found: {len(neg_scores)} rows")
    print(neg_scores.head())

# Check for extremely high scores (>20)
extreme_scores = df[(df['home_score'] > 20) | (df['away_score'] > 20)]
if len(extreme_scores) > 0:
    print(f"\n⚠️  EXTREME SCORES (>20) found: {len(extreme_scores)} rows")
    print(extreme_scores[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']])

# Check for duplicate matches
dupes = df.duplicated(subset=['date', 'home_team', 'away_team'], keep=False)
if dupes.sum() > 0:
    print(f"\n⚠️  DUPLICATE MATCHES (same date/home/away): {dupes.sum()} rows")
    print(df[dupes].sort_values(['date', 'home_team']).head(10))
else:
    print("\n✅ No duplicate matches found")

# neutral field stats
print(f"\nNeutral venue distribution:\n{df['neutral'].value_counts()}")

# extra_time stats
print(f"\nExtra time distribution:\n{df['extra_time'].value_counts()}")

# Elo coverage
elo_missing = df[['home_elo', 'away_elo']].isnull().sum()
print(f"\nElo missing values:\n{elo_missing}")
elo_present = df[['home_elo', 'away_elo']].notna().all(axis=1).sum()
print(f"Rows with both Elo values: {elo_present} / {len(df)} ({100*elo_present/len(df):.1f}%)")

# Elo range
print(f"\nhome_elo range: {df['home_elo'].min()} – {df['home_elo'].max()}")
print(f"away_elo range: {df['away_elo'].min()} – {df['away_elo'].max()}")

# Shootout data
print(f"\nShootout matches: {df['went_to_shootout'].sum()}")
print(f"Shootout winner populated: {df['shootout_winner'].notna().sum()}")

# Outcome distribution
df['outcome'] = np.where(df['home_score'] > df['away_score'], 'H',
                 np.where(df['home_score'] < df['away_score'], 'A', 'D'))
print(f"\nOutcome distribution:\n{df['outcome'].value_counts(normalize=True).round(3)}")

# Matches by decade
df['decade'] = (df['date'].dt.year // 10) * 10
print(f"\nMatches by decade:\n{df.groupby('decade').size()}")

# ── 2. future_fixtures.csv ──────────────────────────────────────────
print("\n" + "=" * 80)
print("2. FUTURE_FIXTURES.CSV")
print("=" * 80)

ff = pd.read_csv("data/processed/future_fixtures.csv", parse_dates=["date"])
print(f"\nShape: {ff.shape}")
print(f"\nColumn dtypes:\n{ff.dtypes}")
print(f"\nNull counts:\n{ff.isnull().sum()}")
print(f"\nAll fixture data:\n{ff.to_string()}")

# Check if future fixture teams exist in historical data
future_teams = set(ff['home_team'].unique()) | set(ff['away_team'].unique())
missing_teams = future_teams - all_teams
if missing_teams:
    print(f"\n⚠️  Teams in fixtures NOT in historical data: {missing_teams}")
else:
    print(f"\n✅ All fixture teams ({len(future_teams)}) found in historical data")

# Schema compatibility check
common_cols = set(df.columns) & set(ff.columns)
only_in_matches = set(df.columns) - set(ff.columns) - {'outcome', 'decade'}
only_in_fixtures = set(ff.columns) - set(df.columns)
print(f"\nCommon columns: {common_cols}")
print(f"Only in matches_full: {only_in_matches}")
print(f"Only in fixtures: {only_in_fixtures}")

# ── 3. elo_ratings.csv ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. ELO_RATINGS.CSV")
print("=" * 80)

elo = pd.read_csv("data/raw/elo_ratings.csv")
print(f"\nShape: {elo.shape}")
print(f"\nColumn dtypes:\n{elo.dtypes}")
print(f"\nElo rating range: {elo['elo_rating'].min()} – {elo['elo_rating'].max()}")
print(f"Teams covered: {elo['team'].nunique()}")

# Check if future teams have elo ratings
elo_teams = set(elo['team'].unique())
fixture_teams_missing_elo = future_teams - elo_teams
if fixture_teams_missing_elo:
    print(f"\n⚠️  Fixture teams missing from Elo: {fixture_teams_missing_elo}")
else:
    print(f"\n✅ All fixture teams have Elo ratings")

# Check for duplicate elo rank
dup_rank = elo[elo.duplicated(subset=['elo_rank'], keep=False)]
if len(dup_rank) > 0:
    print(f"\nTied Elo ranks: {dup_rank['elo_rank'].unique()}")

# ── 4. shootouts.csv ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. SHOOTOUTS.CSV")
print("=" * 80)

so = pd.read_csv("data/raw/shootouts.csv", parse_dates=["date"])
print(f"\nShape: {so.shape}")
print(f"\nColumn dtypes:\n{so.dtypes}")
print(f"\nNull counts:\n{so.isnull().sum()}")
print(f"Date range: {so['date'].min()} → {so['date'].max()}")

# Check if winner is always one of home/away
bad_winners = so[~so['winner'].isin(so['home_team']) & ~so['winner'].isin(so['away_team'])]
winner_valid = so.apply(lambda r: r['winner'] in [r['home_team'], r['away_team']], axis=1)
invalid_winners = (~winner_valid).sum()
if invalid_winners > 0:
    print(f"\n⚠️  Shootout winner not home/away team: {invalid_winners} rows")
    print(so[~winner_valid].head())
else:
    print(f"\n✅ All shootout winners are either home or away team")

# ── 5. fifa_match_importance_weights.csv ─────────────────────────────
print("\n" + "=" * 80)
print("5. FIFA_MATCH_IMPORTANCE_WEIGHTS.CSV")
print("=" * 80)

fw = pd.read_csv("data/raw/fifa_match_importance_weights.csv")
print(f"\nShape: {fw.shape}")
print(f"\nFull table:\n{fw.to_string()}")

# Can we map tournament names to these categories?
print(f"\n--- Tournament names in matches_full ---")
tournaments = df['tournament'].unique()
print(f"Total unique tournaments: {len(tournaments)}")
wc_matches = df[df['tournament'].str.contains('World Cup', case=False, na=False)]
print(f"World Cup matches: {len(wc_matches)}")
print(f"  Tournaments matching 'World Cup': {wc_matches['tournament'].unique()}")

friendly_matches = df[df['tournament'].str.contains('Friendly', case=False, na=False)]
print(f"Friendly matches: {len(friendly_matches)}")

qual_matches = df[df['tournament'].str.contains('qualif', case=False, na=False)]
print(f"Qualifier matches: {len(qual_matches)}")
print(f"  Tournaments matching 'qualif': {qual_matches['tournament'].unique()[:10]}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
