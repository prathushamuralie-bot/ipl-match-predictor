"""
generate_data.py
Creates a synthetic-but-realistic IPL match history dataset.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

TEAMS = [
    {"short": "CSK",  "base_strength": 0.62, "home": "Chepauk, Chennai"},
    {"short": "MI",   "base_strength": 0.65, "home": "Wankhede, Mumbai"},
    {"short": "RCB",  "base_strength": 0.66, "home": "Chinnaswamy, Bengaluru"},
    {"short": "KKR",  "base_strength": 0.60, "home": "Eden Gardens, Kolkata"},
    {"short": "DC",   "base_strength": 0.46, "home": "Arun Jaitley, Delhi"},
    {"short": "PBKS", "base_strength": 0.55, "home": "Mullanpur, Punjab"},
    {"short": "RR",   "base_strength": 0.45, "home": "Sawai Mansingh, Jaipur"},
    {"short": "SRH",  "base_strength": 0.52, "home": "Uppal, Hyderabad"},
    {"short": "GT",   "base_strength": 0.64, "home": "Narendra Modi Stadium, Ahmedabad"},
    {"short": "LSG",  "base_strength": 0.50, "home": "Ekana, Lucknow"},
]
team_lookup = {t["short"]: t for t in TEAMS}
shorts = [t["short"] for t in TEAMS]

N_SEASONS = 6
MATCHES_PER_SEASON = 70

rows = []
for season in range(N_SEASONS):
    wins = {s: 0 for s in shorts}
    losses = {s: 0 for s in shorts}
    last5 = {s: [] for s in shorts}
    nrr = {s: 0.0 for s in shorts}
    h2h_wins = {}

    for m in range(MATCHES_PER_SEASON):
        a, b = np.random.choice(shorts, size=2, replace=False)
        venue_team = np.random.choice([a, b, np.random.choice(shorts)])
        toss_winner = np.random.choice([a, b])

        played_a = wins[a] + losses[a]
        played_b = wins[b] + losses[b]
        wr_a = wins[a] / played_a if played_a else 0.5
        wr_b = wins[b] / played_b if played_b else 0.5

        form_a = (sum(last5[a]) / len(last5[a])) if last5[a] else 0.5
        form_b = (sum(last5[b]) / len(last5[b])) if last5[b] else 0.5

        key = tuple(sorted([a, b]))
        hh = h2h_wins.get(key, {a: 0, b: 0})
        h2h_total = hh.get(a, 0) + hh.get(b, 0)
        h2h_a = hh.get(a, 0) / h2h_total if h2h_total else 0.5

        home_a = 1 if venue_team == a else 0
        home_b = 1 if venue_team == b else 0
        toss_a = 1 if toss_winner == a else 0

        strength_gap = team_lookup[a]["base_strength"] - team_lookup[b]["base_strength"]
        true_p_a = 0.5 + 0.55 * strength_gap \
                   + 0.18 * (wr_a - wr_b) \
                   + 0.15 * (form_a - form_b) \
                   + 0.10 * (h2h_a - 0.5) \
                   + 0.06 * (home_a - home_b) \
                   + 0.03 * (toss_a - 0.5) * 2
        true_p_a = np.clip(true_p_a, 0.05, 0.95)

        a_wins = np.random.rand() < true_p_a
        winner = a if a_wins else b
        loser = b if a_wins else a

        rows.append({
            "season": season + 1,
            "team_a": a, "team_b": b,
            "win_rate_a": round(wr_a, 3), "win_rate_b": round(wr_b, 3),
            "form_a": round(form_a, 3), "form_b": round(form_b, 3),
            "h2h_a": round(h2h_a, 3),
            "nrr_a": round(nrr[a], 3), "nrr_b": round(nrr[b], 3),
            "home_a": home_a, "home_b": home_b,
            "toss_a": toss_a,
            "winner_is_a": int(a_wins),
        })

        wins[winner] += 1
        losses[loser] += 1
        last5[winner].append(1); last5[winner] = last5[winner][-5:]
        last5[loser].append(0); last5[loser] = last5[loser][-5:]
        swing = 0.04 + np.random.rand() * 0.05
        nrr[winner] = round(nrr[winner] + swing, 3)
        nrr[loser] = round(nrr[loser] - swing * 0.7, 3)
        hh[winner] = hh.get(winner, 0) + 1
        h2h_wins[key] = hh

df = pd.DataFrame(rows)
df.to_csv("ipl_matches.csv", index=False)
print(f"Generated {len(df)} matches -> ipl_matches.csv")
print(df.head())