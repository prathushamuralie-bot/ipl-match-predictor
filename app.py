"""
app.py — IPL Match Predictor (Streamlit + Logistic Regression)
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

MODEL = joblib.load("ipl_model.joblib")
SCALER = joblib.load("ipl_scaler.joblib")

FEATURES = [
    "win_rate_a", "win_rate_b",
    "form_a", "form_b",
    "h2h_a",
    "nrr_a", "nrr_b",
    "home_a", "home_b",
    "toss_a",
]

TEAMS = [
    {"short": "CSK",  "name": "Chennai Super Kings",         "home": "Chepauk, Chennai"},
    {"short": "MI",   "name": "Mumbai Indians",               "home": "Wankhede, Mumbai"},
    {"short": "RCB",  "name": "Royal Challengers Bengaluru",  "home": "Chinnaswamy, Bengaluru"},
    {"short": "KKR",  "name": "Kolkata Knight Riders",        "home": "Eden Gardens, Kolkata"},
    {"short": "DC",   "name": "Delhi Capitals",               "home": "Arun Jaitley, Delhi"},
    {"short": "PBKS", "name": "Punjab Kings",                 "home": "Mullanpur, Punjab"},
    {"short": "RR",   "name": "Rajasthan Royals",             "home": "Sawai Mansingh, Jaipur"},
    {"short": "SRH",  "name": "Sunrisers Hyderabad",          "home": "Uppal, Hyderabad"},
    {"short": "GT",   "name": "Gujarat Titans",               "home": "Narendra Modi Stadium, Ahmedabad"},
    {"short": "LSG",  "name": "Lucknow Super Giants",         "home": "Ekana, Lucknow"},
]
SHORT_TO_TEAM = {t["short"]: t for t in TEAMS}
SHORTS = [t["short"] for t in TEAMS]

if "stats" not in st.session_state:
    seed_wl = {
        "CSK": (7, 6), "MI": (8, 5), "RCB": (9, 5), "KKR": (8, 6), "DC": (6, 8),
        "PBKS": (8, 6), "RR": (5, 9), "SRH": (7, 7), "GT": (9, 5), "LSG": (6, 7),
    }
    seed_nrr = {
        "CSK": 0.18, "MI": 0.31, "RCB": 0.42, "KKR": 0.22, "DC": -0.21,
        "PBKS": 0.15, "RR": -0.34, "SRH": 0.05, "GT": 0.38, "LSG": -0.09,
    }
    stats = {}
    for s in SHORTS:
        w, l = seed_wl[s]
        wr = w / (w + l)
        form = [1 if np.random.rand() < wr else 0 for _ in range(5)]
        stats[s] = {"wins": w, "losses": l, "nrr": seed_nrr[s], "form": form}
    st.session_state.stats = stats
    st.session_state.h2h = {}
    st.session_state.feed = []


def h2h_key(a, b):
    return tuple(sorted([a, b]))


def get_h2h_a_rate(a, b):
    key = h2h_key(a, b)
    rec = st.session_state.h2h.get(key, {})
    total = rec.get(a, 0) + rec.get(b, 0)
    return rec.get(a, 0) / total if total else 0.5


def build_features(a, b, venue_short, toss_short):
    s = st.session_state.stats
    wr_a = s[a]["wins"] / (s[a]["wins"] + s[a]["losses"])
    wr_b = s[b]["wins"] / (s[b]["wins"] + s[b]["losses"])
    form_a = np.mean(s[a]["form"])
    form_b = np.mean(s[b]["form"])
    h2h_a = get_h2h_a_rate(a, b)
    return pd.DataFrame([{
        "win_rate_a": wr_a, "win_rate_b": wr_b,
        "form_a": form_a, "form_b": form_b,
        "h2h_a": h2h_a,
        "nrr_a": s[a]["nrr"], "nrr_b": s[b]["nrr"],
        "home_a": 1 if venue_short == a else 0,
        "home_b": 1 if venue_short == b else 0,
        "toss_a": 1 if toss_short == a else 0,
    }])[FEATURES]


def predict_proba_a(a, b, venue_short, toss_short):
    X = build_features(a, b, venue_short, toss_short)
    X_s = SCALER.transform(X)
    p_a = MODEL.predict_proba(X_s)[0, 1]
    return float(np.clip(p_a, 0.03, 0.97)), X


def simulate_match(a, b, venue_short, toss_short):
    p_a, _ = predict_proba_a(a, b, venue_short, toss_short)
    a_wins = np.random.rand() < p_a
    winner, loser = (a, b) if a_wins else (b, a)

    s = st.session_state.stats
    s[winner]["wins"] += 1
    s[loser]["losses"] += 1
    s[winner]["form"] = (s[winner]["form"] + [1])[-5:]
    s[loser]["form"] = (s[loser]["form"] + [0])[-5:]
    swing = round(0.04 + np.random.rand() * 0.05, 3)
    s[winner]["nrr"] = round(s[winner]["nrr"] + swing, 3)
    s[loser]["nrr"] = round(s[loser]["nrr"] - swing * 0.7, 3)

    key = h2h_key(a, b)
    st.session_state.h2h.setdefault(key, {})
    st.session_state.h2h[key][winner] = st.session_state.h2h[key].get(winner, 0) + 1

    st.session_state.feed.insert(0, {
        "winner": winner, "loser": loser,
        "venue": SHORT_TO_TEAM[venue_short]["home"],
    })
    return winner, loser, p_a
# ---------------------- UI ----------------------
st.markdown("""
<style>
.block-container{padding-top:1.5rem;}
h1{font-family:'Trebuchet MS', sans-serif;}
.metric-card{background:#11182b;border:1px solid #23304d;border-radius:10px;padding:14px;}
</style>
""", unsafe_allow_html=True)

st.title("🏏 IPL Match Predictor")
st.caption("Logistic Regression model trained on historical-style match data · live stats update with every simulated match")

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Predict a matchup")
    name_options = [f"{t['name']} ({t['short']})" for t in TEAMS]
    a_idx = st.selectbox("Team A", range(len(TEAMS)), format_func=lambda i: name_options[i], index=1)
    b_idx = st.selectbox("Team B", range(len(TEAMS)), format_func=lambda i: name_options[i], index=2)
    a_short, b_short = TEAMS[a_idx]["short"], TEAMS[b_idx]["short"]

    venue_idx = st.selectbox("Venue", range(len(TEAMS)), format_func=lambda i: f"{TEAMS[i]['home']}", index=a_idx)
    venue_short = TEAMS[venue_idx]["short"]

    toss_choice = st.radio("Toss winner", ["No toss yet", TEAMS[a_idx]["short"], TEAMS[b_idx]["short"]], horizontal=True)
    toss_short = toss_choice if toss_choice in (a_short, b_short) else None

    if a_short == b_short:
        st.warning("Pick two different teams.")
    else:
        p_a, X = predict_proba_a(a_short, b_short, venue_short, toss_short)
        p_b = 1 - p_a

        st.markdown(f"#### {a_short}: **{p_a*100:.1f}%** &nbsp;&nbsp;vs&nbsp;&nbsp; {b_short}: **{p_b*100:.1f}%**")
        st.progress(p_a)

        with st.expander("Show model input features"):
            st.dataframe(X, hide_index=True)

        if st.button("▶ Simulate this match", type="primary", use_container_width=True):
            winner, loser, p_at_sim = simulate_match(a_short, b_short, venue_short, toss_short)
            st.success(f"**{winner}** won the simulated match against {loser} (model gave {a_short} {p_a*100:.1f}% before kickoff).")
            st.rerun()

with col2:
    st.subheader("Live standings")
    rows = []
    for t in TEAMS:
        s = st.session_state.stats[t["short"]]
        played = s["wins"] + s["losses"]
        wr = round(100 * s["wins"] / played, 1) if played else 0
        rows.append({
            "Team": t["short"], "P": played, "W": s["wins"], "L": s["losses"],
            "NRR": s["nrr"], "Win%": wr,
            "Last 5": "".join("W" if f else "L" for f in s["form"]),
        })
    standings_df = pd.DataFrame(rows).sort_values("Win%", ascending=False).reset_index(drop=True)
    st.dataframe(standings_df, hide_index=True, use_container_width=True)

    st.subheader("Match feed")
    if not st.session_state.feed:
        st.caption("No matches simulated yet this session.")
    else:
        for item in st.session_state.feed[:10]:
            st.markdown(f"- **{item['winner']}** beat {item['loser']} · venue: {item['venue']}")

st.divider()
with st.expander("How the model works"):
    st.markdown("""
    A **Logistic Regression** model (scikit-learn) was trained on 420 simulated
    historical-style matches with these pre-match features:
    season win-rate, last-5 form, head-to-head rate, net run rate, home
    advantage, and toss outcome — for both teams.

    The model outputs a probability that Team A wins, which is what drives
    the percentage bar above. Every time you simulate a match, the live
    stats (wins/losses/NRR/form/head-to-head) shift, so the *next* prediction
    for any matchup reflects the updated season state.
    """)