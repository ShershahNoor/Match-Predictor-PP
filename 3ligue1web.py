import os
import json
import time
import random
from typing import Dict, List

import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

API_KEY = "a245808884b2b7335d24204a22535922"
BASE_URL = "https://v3.football.api-sports.io"
PREFERRED_SEASON = 2024
CACHE_DIR = "api_cache"
CACHE_EXPIRY = 86400

LEAGUES = {
    "Ligue 1": 61,
}

class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
    def forward(self, x):
        return self.net(x)

@st.cache_data
def api_get(endpoint: str, params: Dict) -> Dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = endpoint.replace("/", "_") + "_" + "_".join(f"{k}-{v}" for k, v in params.items())
    path = os.path.join(CACHE_DIR, key + ".json")

    if os.path.exists(path) and os.path.getsize(path) > 0:
        if time.time() - os.path.getmtime(path) < CACHE_EXPIRY:
            try:
                with open(path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                os.remove(path)

    headers = {"x-apisports-key": API_KEY}
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("response") is not None:
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        else:
            st.error(f"API Error: {data.get('errors')}")
            return {}
            
    except Exception as e:
        st.error(f"Request failed: {e}")
        return {}

def fetch_teams(league_id, season):
    data = api_get("teams", {"league": league_id, "season": season})
    return {t["team"]["name"]: t["team"]["id"] for t in data.get("response", [])}

def extract_features(stats: Dict) -> np.ndarray:
    f, g = stats.get("fixtures", {}), stats.get("goals", {})
    played, wins, draws, losses = f.get("played",{}).get("total",0), f.get("wins",{}).get("total",0), f.get("draws",{}).get("total",0), f.get("loses",{}).get("total",0)
    gf, ga = g.get("for",{}).get("total",{}).get("total",0), g.get("against",{}).get("total",{}).get("total",0)
    avg_pts = (wins * 3 + draws) / played if played else 0
    return np.array([played, wins, draws, losses, gf, ga, gf - ga, avg_pts], dtype=np.float32)

def match_features(h: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.concatenate([h - a, h, a])

@st.cache_resource
def get_trained_model_and_stats(_league_id, _season):
    teams = fetch_teams(_league_id, _season)
    if not teams:
        st.error(f"NO TEAMS IN LEAGUE {_league_id}. Check API Key.")
        return None, {}, {}

    fixtures_data = api_get("fixtures", {"league": _league_id, "season": _season, "status": "FT"})
    fixtures = fixtures_data.get("response", [])
    
    stats_map = {}
    for t_name, tid in teams.items():
        stats_resp = api_get("teams/statistics", {"league": _league_id, "season": _season, "team": tid})
        stats = stats_resp.get("response")
        if stats: 
            stats_map[tid] = extract_features(stats)

    X, y = [], []
    for f in fixtures:
        h_id = f["teams"]["home"]["id"]
        a_id = f["teams"]["away"]["id"]
        
        if h_id in stats_map and a_id in stats_map:
            gh, ga = f["goals"]["home"], f["goals"]["away"]
            if gh is not None and ga is not None:
                label = 0 if gh > ga else 1 if gh == ga else 2
                X.append(match_features(stats_map[h_id], stats_map[a_id]))
                y.append(label)

    st.sidebar.write(f"**Debug Log ({_league_id}):**")
    st.sidebar.write(f"- Teams found: {len(teams)}")
    st.sidebar.write(f"- Stats collected: {len(stats_map)}")
    st.sidebar.write(f"- Completed matches: {len(fixtures)}")
    st.sidebar.write(f"- Training samples created: {len(X)}")

    if len(X) == 0:
        st.warning(f"No training data for {_season}. Try changing PREFERRED_SEASON to 2023.")
        return None, stats_map, teams

    X_np = np.array(X)
    if X_np.ndim < 2:
        st.error("Data format error: X array is not 2D.")
        return None, stats_map, teams

    X_train = torch.tensor(X_np)
    y_train = torch.tensor(np.array(y))
    
    model = MLP(X_train.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        
    return model, stats_map, teams

st.set_page_config(page_title="Aura Blud Football Predictor", layout="wide")
st.title("Ligue 1 Match Predictor 🇫🇷")
st.write("Data gathered from API-Sports, Deep Learning Personal Project")

tabs = st.tabs(list(LEAGUES.keys()))

for i, (name, L_ID) in enumerate(LEAGUES.items()):
    with tabs[i]:
        st.subheader(f"{name} Analysis")
        
        with st.spinner("Analyzing league data and training model..."):
            model, stats_map, team_dict = get_trained_model_and_stats(L_ID, PREFERRED_SEASON)
        
        col1, col2 = st.columns(2)
        team_names = sorted(list(team_dict.keys()))
        
        with col1:
            home = st.selectbox("Select Home Team 🥰", team_names, key=f"h_{L_ID}")
        with col2:
            away = st.selectbox("Select Away Team 😡", team_names, key=f"a_{L_ID}")

        if st.button(f"Predict {name} Matchup", key=f"btn_{L_ID}"):
            if home == away:
                st.error("Teams must be different!")
            else:
                home_id = team_dict[home]
                away_id = team_dict[away]

                if home_id not in stats_map or away_id not in stats_map:
                    missing = home if home_id not in stats_map else away
                    st.error(f"⚠️ Missing data for **{missing}**. This team might not have enough stats for the {PREFERRED_SEASON} season yet.")
                    st.info("Try switching to a previous season (like 2023 or 2024) at the top of the script to ensure all teams have complete data.")
                else:
                    h_feat = stats_map[home_id]
                    a_feat = stats_map[away_id]
                    x = match_features(h_feat, a_feat)
                    
                    with torch.no_grad():
                        probs = torch.softmax(model(torch.tensor(x)), dim=0).numpy()
                
                labels = ["Home Win!", "Draw", "Away Win!"]
                idx = int(np.argmax(probs))
                confidence = probs[idx] * 100

                st.divider()
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.metric("Predicted Outcome", labels[idx])
                    
                with res_col2:
                    st.metric("How confident am I?", f"{confidence:.1f}%", delta="High" if confidence > 60 else "Low")

                st.write("Match Probability Gauge:")
                st.progress(float(probs[idx])) 

                if idx == 0:
                    st.success(f"🤩 ShershahAI uses futuresight to predict **{home}** wins at home!")
                elif idx == 2:
                    st.success(f"😭 ShershahAI uses future-sight to predict an away win for **{away}**!")
                else:
                    st.warning(f"😅 A tough matchup. ShershahAI sees that a Draw is the most likely outcome.")