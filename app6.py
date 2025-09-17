import streamlit as st
import pandas as pd
import numpy as np
import re
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics import ndcg_score
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("API KEY LOADED:", os.getenv("AZURE_OPENAI_API_KEY"))


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="NBA Playoff Game Recommender", layout="wide")
st.title("🏀 NBA Playoff Game Recommender with Gen AI")


# -----------------------------
# Helpers: Normalization
# -----------------------------
def normalize_matchup(name: str) -> str:
    """
    Normalize a matchup into 'Team A @ Team B' with alphabetical ordering.
    Works for both validation ('A @ B') and schedule ('A & B').
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # Split on @ or &
    if "@" in name:
        parts = [p.strip() for p in name.split("@")]
    elif "&" in name:
        parts = [p.strip() for p in name.split("&")]
    else:
        return name
    if len(parts) != 2:
        return name
    team1, team2 = sorted(parts)
    return f"{team1} @ {team2}"


# -----------------------------
# Feature Dictionaries
# -----------------------------
top_players = {
    "Denver Nuggets": ["Nikola Jokic"],
    "Boston Celtics": ["Jayson Tatum"],
    "Golden State Warriors": ["Stephen Curry"],
    "Los Angeles Lakers": ["LeBron James", "Anthony Davis"],
    "New York Knicks": ["Jalen Brunson"],
    "Cleveland Cavaliers": ["Donovan Mitchell"],
    "Minnesota Timberwolves": ["Anthony Edwards"],
    "Indiana Pacers": ["Tyrese Haliburton"],
    "Oklahoma City Thunder": ["Shai Gilgeous-Alexander"]
}

team_regions = {
    "Denver Nuggets": "Colorado",
    "Boston Celtics": "Massachusetts",
    "Golden State Warriors": "California",
    "Los Angeles Lakers": "California",
    "New York Knicks": "New York",
    "Cleveland Cavaliers": "Ohio",
    "Minnesota Timberwolves": "Minnesota",
    "Indiana Pacers": "Indiana",
    "Oklahoma City Thunder": "Oklahoma"
}

big_market_teams = {"New York Knicks", "Los Angeles Lakers", "Boston Celtics",
                    "Chicago Bulls", "Golden State Warriors"}

rivalries = [
    ("Boston Celtics", "New York Knicks"),
    ("Los Angeles Lakers", "Los Angeles Clippers"),
    ("Cleveland Cavaliers", "Golden State Warriors")
]


# -----------------------------
# Tagging Functions
# -----------------------------
def tag_star_players(desc): return [p for t, ps in top_players.items() if t.lower() in desc.lower() for p in ps]
def tag_teams(desc): return [t for t in top_players if t.lower() in desc.lower()]
def tag_regions(teams): return [team_regions.get(t, "Unknown") for t in teams]
def tag_market(teams): return ["Market_Big" if t in big_market_teams else "Market_Value" for t in teams]
def tag_rivalry(teams): return ["Rivalry"] if any(a in teams and b in teams for a, b in rivalries) else []
def tag_superstar(players): return ["Has_Superstar"] if players else []


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df_train = pd.read_excel("df_train.xlsx")
    df_val = pd.read_excel("df_validation.xlsx")
    schedule = pd.read_csv("playoff_schedule.csv")

    for df in [df_train, df_val]:
        df.columns = df.columns.str.lower()
        df["event_norm"] = df["event_description"].apply(normalize_matchup)
        df["star_players"] = df["event_description"].apply(tag_star_players)
        df["teams"] = df["event_description"].apply(tag_teams)
        df["regions"] = df["teams"].apply(tag_regions)
        df["game_norm"] = df["event_description"].apply(normalize_matchup)  # 👈 ensure this exists


    schedule.columns = schedule.columns.str.lower()
    schedule["game_norm"] = schedule["game"].apply(normalize_matchup)
    schedule["date"] = pd.to_datetime(schedule["date"], dayfirst=True)

    # Round mapping
    matchup_to_round = {
    normalize_matchup("Boston Celtics & Orlando Magic"): "R8",
    normalize_matchup("Cleveland Cavaliers & Miami Heat"): "R8",
    normalize_matchup("Denver Nuggets & Los Angeles Clippers"): "R8",
    normalize_matchup("Detroit Pistons & New York Knicks"): "R8",
    normalize_matchup("Golden State Warriors & Houston Rockets"): "R8",
    normalize_matchup("Indiana Pacers & Milwaukee Bucks"): "R8",
    normalize_matchup("Los Angeles Lakers & Minnesota Timberwolves"): "R8",
    normalize_matchup("Memphis Grizzlies & Oklahoma City Thunder"): "R8",
    normalize_matchup("Boston Celtics & New York Knicks"): "Quarterfinals",
    normalize_matchup("Cleveland Cavaliers & Indiana Pacers"): "Quarterfinals",
    normalize_matchup("Denver Nuggets & Oklahoma City Thunder"): "Quarterfinals",
    normalize_matchup("Golden State Warriors & Minnesota Timberwolves"): "Quarterfinals",
    normalize_matchup("Indiana Pacers & New York Knicks"): "Semifinals",
    normalize_matchup("Minnesota Timberwolves & Oklahoma City Thunder"): "Semifinals",
    normalize_matchup("Indiana Pacers & Oklahoma City Thunder"): "Finals"
}

    schedule["round"] = schedule["game_norm"].map(matchup_to_round)
    return df_train, df_val, schedule


df_train, df_validation, playoff_schedule = load_data()


# -----------------------------
# Favourite Team Detector
# -----------------------------
@st.cache_data
def get_fav_teams(df_train, playoff_schedule):
    favs = {}
    for uid in df_train["mask_id"].unique():
        user_games = df_train[df_train["mask_id"] == uid].explode("teams")

        if user_games["teams"].dropna().empty:   # 👈 skip if no valid teams
            continue

        fav_team = user_games["teams"].value_counts().idxmax()
        favs[uid] = fav_team
    return favs



fav_teams = get_fav_teams(df_train, playoff_schedule)


# -----------------------------
# Feature Builders
# -----------------------------
def build_item_features(df):
    features = []
    for row in df.itertuples():
        feats = []
        feats.extend(row.teams)
        feats.extend(row.regions)
        feats.extend(tag_market(row.teams))
        feats.extend(tag_rivalry(row.teams))
        feats.extend(tag_superstar(row.star_players))
        features.append((row.event_norm, feats))
    return features


def build_user_features(df):
    features = []
    for uid in df["mask_id"].unique():
        user_games = df[df["mask_id"] == uid].explode("teams")
        familiar_teams = user_games["teams"].value_counts().head(2).index.tolist()

        feats = [f"Familiar_Team_{t}" for t in familiar_teams]
        if uid in fav_teams:
            feats.append(f"Fav_Team_{fav_teams[uid]}")
        features.append((uid, feats))
    return features


def segment_players(df_train):
    """Segment players based on how often they bet (activity frequency)."""
    activity = df_train.groupby("mask_id")["event_norm"].count()
    
    # Simple segmentation: Low / Medium / High
    thresholds = activity.quantile([0.33, 0.66])
    segments = {}
    
    for uid, count in activity.items():
        if count <= thresholds.iloc[0]:
            segments[uid] = "Low_Activity"
        elif count <= thresholds.iloc[1]:
            segments[uid] = "Medium_Activity"
        else:
            segments[uid] = "High_Activity"
    
    return segments


# Precompute once
player_segments = segment_players(df_train)

def build_user_features(df):
    features = []
    for uid in df["mask_id"].unique():
        user_games = df[df["mask_id"] == uid].explode("teams")
        familiar_teams = user_games["teams"].value_counts().head(2).index.tolist()

        feats = []
        feats.extend([f"Familiar_Team_{t}" for t in familiar_teams])

        # Favorite team
        if uid in fav_teams:
            feats.append(f"Fav_Team_{fav_teams[uid]}")

        # Activity segment
        seg = player_segments.get(uid, "Unknown")
        feats.append(seg)
        if seg == "Low_Activity":
            feats.append("Prefers_Big_Market")

        # 🆕 Regional Loyalty
        regions_played = user_games["teams"].map(lambda t: team_regions.get(t, "Unknown")).dropna()
        if not regions_played.empty:
            top_region = regions_played.value_counts().idxmax()
            ratio = regions_played.value_counts(normalize=True).iloc[0]
            if ratio > 0.7:  # strong concentration
                feats.append(f"Region_Loyalty_{top_region}")

        features.append((uid, feats))
    return features


def build_item_features(df):
    features = []
    for row in df.itertuples():
        feats = []
        feats.extend(row.teams)  # Teams
        feats.extend(row.regions)  # Regions already extracted
        feats.extend(tag_market(row.teams))
        feats.extend(tag_rivalry(row.teams))
        feats.extend(tag_superstar(row.star_players))

        # 🆕 Add region feature explicitly
        for reg in row.regions:
            feats.append(f"Region_{reg}")

        features.append((row.event_norm, feats))
    return features




# -----------------------------
# Simulation: Series-level
# -----------------------------
def simulate_series(df_val, model, mapping, round_sched, item_feats, user_feats):
    mask_map, _, game_map, game_rev = (
        mapping[0], mapping[1], mapping[2], {v: k for k, v in mapping[2].items()}
    )

    results = []
    norm_to_pretty = dict(zip(round_sched["game_norm"], round_sched["game"]))

    for game_norm in round_sched["game_norm"].unique():
        if game_norm not in game_map:
            continue

        game_idx = game_map[game_norm]

        # 🎯 Actual players who bet on this series (any date, any bet)
        actual_players = set(df_val.loc[df_val["game_norm"] == game_norm, "mask_id"].unique())

        # 🤖 Predicted players (model thinks will bet on this series)
        predicted_players = set()
        for uid in df_val["mask_id"].unique():
            if uid not in mask_map:
                continue
            uidx = mask_map[uid]
            score = model.predict(
                uidx, [game_idx],
                item_features=item_feats,
                user_features=user_feats
            )[0]
            if score > 0:  # threshold
                predicted_players.add(uid)

        results.append({
            "series": norm_to_pretty.get(game_norm, game_norm),
            "actual_players": len(actual_players),
            "predicted_players": len(predicted_players)
        })

    return pd.DataFrame(results)



# -----------------------------
# Train Model
# -----------------------------
def train_model(df, loss="warp", comps=64, epochs=30):
    # ✅ Include all playoff matchups (train + schedule)
    all_items = pd.concat([
        df["event_norm"],
        playoff_schedule["game_norm"]
    ]).unique()

    dataset = Dataset()
    dataset.fit(
        users=df["mask_id"].unique(),
        items=all_items,  # 👈 ensures playoff games are in the mapping
        item_features=[f for _, feats in build_item_features(df) for f in feats],
        user_features=[f for _, feats in build_user_features(df) for f in feats]
    )

    # Interactions only from training
    interactions, weights = dataset.build_interactions(
        (r.mask_id, r.event_norm, r.wager_amount) for r in df.itertuples()
    )

    # Features
    item_feats = dataset.build_item_features(build_item_features(df))
    user_feats = dataset.build_user_features(build_user_features(df))

    # Train model
    model = LightFM(loss=loss, no_components=comps)
    model.fit(
        interactions,
        item_features=item_feats,
        user_features=user_feats,
        sample_weight=weights,
        epochs=epochs,
        num_threads=4
    )

    mapping = dataset.mapping()
    return model, mapping, item_feats, user_feats



# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, df_val, mapping, round_sched, item_feats, user_feats):
    mask_map, _, game_map, game_rev = (
        mapping[0], mapping[1], mapping[2], {v: k for k, v in mapping[2].items()}
    )
    games = round_sched["game_norm"].unique()
    active_players = df_val[df_val["game_norm"].isin(games)]["mask_id"].unique()

    hits, total, precisions, ndcgs = 0, 0, [], []
    for uid in active_players:
        if uid not in mask_map: 
            continue
        user_idx = mask_map[uid]
        gidx = [game_map[g] for g in games if g in game_map]
        if not gidx: 
            continue

        scores = model.predict(user_idx, gidx, item_features=item_feats, user_features=user_feats)
        recs = sorted([(game_rev[i], s) for i, s in zip(gidx, scores)],
                      key=lambda x: x[1], reverse=True)[:3]
        rec_games = [g for g, _ in recs]
        actual = set(df_val.loc[df_val["mask_id"] == uid, "game_norm"]) & set(games)

        if actual & set(rec_games):
            hits += 1
        total += 1

        y_true = [1 if g in actual else 0 for g, _ in recs]
        y_score = [s for _, s in recs]
        if sum(y_true) > 0:
            precisions.append(sum(y_true) / len(recs))
            ndcgs.append(ndcg_score([y_true], [y_score]))

    return hits / total if total else 0, np.mean(precisions) if precisions else 0, np.mean(ndcgs) if ndcgs else 0


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    round_choice = st.selectbox(
        "Select Round", 
        ["R8", "Quarterfinals", "Semifinals", "Finals"]
    )
    st.markdown(f"Filtering recommendations for **{round_choice}** games only.")

    st.header("📌 Tab Overview")
    st.markdown("""
    This app recommends NBA games using a **hybrid recommender system**:
    - Train & Evaluate - This section lets you train and evaluate the recommender model.
You can experiment with different model settings and instantly see how they affect recommendation quality.
    - Player Explorer  
    - Simulation  
    - Dashboard  
    - Chatbot  
    """)
    
# Filter playoff schedule based on round
round_sched = playoff_schedule[playoff_schedule["round"] == round_choice]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Train & Evaluate", "🎯 Player Explorer", "📅 Simulation",
    "📊 Dashboard"
])

with tab1:
    st.subheader("⚙️ Train Model")
    st.markdown("""
#### About this Tab
This section lets you **train and evaluate** the recommender model.  
You can experiment with different model settings and instantly see how they affect recommendation quality.

- Adjust **model hyperparameters** (loss, latent dimensions, epochs).  
- Apply **boost factors** (Star Boost, BigFan Boost).  

Evaluation is restricted to the **selected playoff round** (e.g., R8, Quarterfinals).  
Metrics reported:
- **Hit@3** → At least one of the player's actual bets appears in the Top-3 recs  
- **Precision@3** → Fraction of the Top-3 recs that are correct  
- **NDCG** → Ranking quality (higher = better ordering)  
""")

    # --- Hyperparameter controls ---
    loss = st.selectbox("Loss Function", ["warp", "bpr"], index=0)
    components = st.selectbox("Latent Dimensions", [32, 64, 128], index=0)

    alpha = st.slider("Alpha (CF vs Popularity)", 0.5, 0.9, 0.7)
    st.caption("📌 Balance between collaborative filtering and global popularity")

    star_boost = st.slider("Star Boost", 0.0, 0.5, 0.2)
    st.caption("📌 Extra weight for games with marquee players")

    fan_boost = st.slider("BigFan Boost", 0.0, 0.5, 0.2)
    st.caption("📌 Extra weight for bettors who follow many games")

    epochs = st.slider("Epochs", 10, 100, 30, step=10)
    st.caption("📌 More epochs = better learning, but longer training")

    # --- Train button ---
    if st.button("Train Model"):
        with st.spinner("Training LightFM model... ⏳"):
            try:
                model, mapping, item_feats, user_feats = train_model(
                    df_train, loss=loss, comps=components, epochs=epochs
                )

                # Save to session state
                st.session_state.update({
                    "model": model,
                    "mapping": mapping,
                    "item_feats": item_feats,
                    "user_feats": user_feats,
                    "alpha": alpha,
                    "star_boost": star_boost,
                    "fan_boost": fan_boost
                })

                # Evaluate immediately
                hit, precision, ndcg = evaluate(
                    model, df_validation, mapping, round_sched, item_feats, user_feats
                )
                st.success(
                    f"✅ Trained Model → Hit@3={hit:.2%}, "
                    f"Precision@3={precision:.2%}, NDCG={ndcg:.2f}"
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    # If no model yet
    if "model" not in st.session_state:
        st.info("⚠️ Train a model first to unlock recommendations and analytics.")

    # --- Hyperparameter tuning ---
    st.markdown("---")
    st.subheader("🔎 Hyperparameter Tuning")

    if st.button("Run Hyperparameter Tuning"):
        search_space = {
            "loss": ["warp", "bpr"],
            "components": [32, 64],
            "epochs": [20, 40],
            "star_boost": [0.0, 0.2, 0.4],
            "fan_boost": [0.0, 0.2, 0.4]
        }

        results = []
        try:
            for l in search_space["loss"]:
                for comp in search_space["components"]:
                    for ep in search_space["epochs"]:
                        model, mapping, item_feats, user_feats = train_model(
                            df_train, loss=l, comps=comp, epochs=ep
                        )
                        for sb in search_space["star_boost"]:
                            for fb in search_space["fan_boost"]:
                                hit, prec, ndcg = evaluate(
                                    model, df_validation, mapping, round_sched,
                                    item_feats, user_feats
                                )
                                results.append({
                                    "Loss": l,
                                    "Dims": comp,
                                    "Epochs": ep,
                                    "StarBoost": sb,
                                    "FanBoost": fb,
                                    "Hit@3": hit,
                                    "Precision@3": prec,
                                    "NDCG": ndcg
                                })

            results_df = pd.DataFrame(results).sort_values("NDCG", ascending=False)
            st.dataframe(results_df)

            best = results_df.iloc[0]
            st.success(
                f"🏆 Best Config → Loss={best['Loss']}, Dims={best['Dims']}, "
                f"Epochs={best['Epochs']}, StarBoost={best['StarBoost']}, "
                f"FanBoost={best['FanBoost']} → "
                f"Hit@3={best['Hit@3']:.2%}, Precision@3={best['Precision@3']:.2%}, "
                f"NDCG={best['NDCG']:.2f}"
            )
        except Exception as e:
            st.error(f"Tuning failed: {e}")


# --- Tab 2: Player Explorer ---
with tab2:
    st.subheader("🎯 Player Explorer")
    st.markdown("""
### 🔎 About this Tab
Explore **personalized recommendations** for individual players (`mask_id`).  

- Select a player from the dropdown.  
- Click **Run Recommendations** to generate the **Top 3 suggested games**.  
- Recommendations are restricted to the **selected playoff round**.  
- Compare recommendations against the player's **actual bets**.  
- View **historical betting behavior** and **team loyalty**.  
- Ask the **AI Explainer** why these games were recommended.
""")

    # --- Require model ---
    if "model" not in st.session_state:
        st.warning("⚠️ Train a model first in the 'Train & Evaluate' tab.")
    else:
        uid = st.selectbox("Select mask_id", df_validation["mask_id"].unique())

        # --- Run Recommendations ---
        if st.button("Run Recommendations"):
            try:
                mask_map, _, game_map, game_rev = (
                    st.session_state["mapping"][0],
                    st.session_state["mapping"][1],
                    st.session_state["mapping"][2],
                    {v: k for k, v in st.session_state["mapping"][2].items()}
                )

                if uid in mask_map:
                    uidx = mask_map[uid]
                    round_games = round_sched["game_norm"].unique()
                    gidx = [game_map[g] for g in round_games if g in game_map]

                    if gidx:
                        scores = st.session_state["model"].predict(
                            uidx, gidx,
                            item_features=st.session_state["item_feats"],
                            user_features=st.session_state["user_feats"]
                        )
                        recs = sorted(
                            [(game_rev[i], s) for i, s in zip(gidx, scores)],
                            key=lambda x: x[1],
                            reverse=True
                        )[:3]

                        # Map back to human-readable names
                        norm_to_pretty = dict(zip(
                            playoff_schedule["game_norm"],
                            playoff_schedule["game"]
                        ))
                        rec_games = [norm_to_pretty.get(g, g) for g, _ in recs]

                        # Actual bets
                        actual_games = df_validation.loc[
                            df_validation["mask_id"] == uid, "game_norm"
                        ].unique()
                        actual_pretty = [norm_to_pretty.get(g, g) for g in actual_games if g in norm_to_pretty]

                        # Player segment + regional loyalty
                        segment = player_segments.get(uid, "Unknown")
                        user_feats_dict = dict(build_user_features(df_train))
                        loyalty = [f for f in user_feats_dict.get(uid, []) if f.startswith("Region_Loyalty_")]
                        region_loyalty = loyalty[0] if loyalty else "None"

                        # Save in session
                        st.session_state.update({
                            "last_uid": uid,
                            "last_recs": rec_games,
                            "last_actual": actual_pretty,
                            "last_segment": segment,
                            "last_region_loyalty": region_loyalty
                        })
            except Exception as e:
                st.error(f"Recommendation failed: {e}")

        # --- Show Recommendations ---
        if "last_recs" in st.session_state:
            st.write(f"**Top {round_choice} Recommendations for mask_id {st.session_state['last_uid']}:**")
            for g in st.session_state["last_recs"]:
                st.write(f"👉 {g}")

        # --- Show Actual Bets ---
        if "last_actual" in st.session_state:
            st.write(f"**Actual Bets in Validation for mask_id {st.session_state['last_uid']}:**")
            if st.session_state["last_actual"]:
                for g in st.session_state["last_actual"]:
                    st.markdown(f"- ✅ **{g}**")
            else:
                st.write("❌ No bets recorded in validation for this player.")

        # --- Historical Betting Behaviour ---
        if "last_uid" in st.session_state:
            player_history = df_train.loc[df_train["mask_id"] == st.session_state["last_uid"]]

            if not player_history.empty:
                st.markdown("### 📈 Historical Betting Behaviour (Training Data)")

                # Detect possible date column
                date_col = next(
                    (c for c in ["betdate", "purchasedate", "date", "transaction_date", "event_date"]
                     if c in player_history.columns),
                    None
                )

                if date_col:
                    player_history[date_col] = pd.to_datetime(player_history[date_col], errors="coerce")
                    history_counts = (
                        player_history.groupby(player_history[date_col].dt.to_period("M"))
                        .size()
                        .reset_index(name="bets")
                    )
                    history_counts[date_col] = history_counts[date_col].astype(str)
                    st.line_chart(history_counts.set_index(date_col)["bets"])

                # Top 3 favourite teams
                norm_to_pretty = dict(zip(playoff_schedule["game_norm"], playoff_schedule["game"]))
                team_counts = []
                for g in player_history["game_norm"]:
                    pretty = norm_to_pretty.get(g, g)
                    for team in top_players.keys():
                        if team.lower() in pretty.lower():
                            team_counts.append(team)

                if team_counts:
                    st.markdown("### 🏀 Top 3 Teams This Player Bets On")
                    top3 = pd.Series(team_counts).value_counts().head(3)
                    for team, count in top3.items():
                        st.markdown(f"- **{team}**: {count} bets")

                    # Monthly stacked trend
                    team_history = []
                    for g, d in zip(player_history["game_norm"], player_history[date_col]):
                        pretty = norm_to_pretty.get(g, g)
                        for team in top3.index:
                            if team.lower() in pretty.lower():
                                team_history.append({
                                    "month": d.to_period("M").strftime("%Y-%m"),
                                    "team": team
                                })

                    if team_history:
                        df_team_history = pd.DataFrame(team_history)
                        team_trend = (
                            df_team_history.groupby(["month", "team"])
                            .size()
                            .reset_index(name="bets")
                        )
                        team_trend_pivot = team_trend.pivot(
                            index="month", columns="team", values="bets"
                        ).fillna(0)

                        # Normalize to % shares
                        team_trend_norm = team_trend_pivot.div(team_trend_pivot.sum(axis=1), axis=0)
                        st.area_chart(team_trend_norm)
                else:
                    st.info("No team associations found in this player's betting history.")
            else:
                st.info("No historical bets found for this player in training data.")

        # --- AI Explanation ---
        if "last_recs" in st.session_state and st.button("🤖 Explain with AI"):
            fav_team = fav_teams.get(st.session_state["last_uid"], "None")
            rec_games = st.session_state["last_recs"]
            actual_games = st.session_state.get("last_actual", [])
            segment = st.session_state.get("last_segment", "Unknown")
            region_loyalty = st.session_state.get("last_region_loyalty", "None")

            prompt = f"""
            Player {st.session_state['last_uid']} is segmented as a **{segment}** bettor.
            They were recommended: {", ".join(rec_games)}.
            Their favorite team: {fav_team}.
            Regional loyalty: {region_loyalty}.
            Actual bets: {", ".join(actual_games) if actual_games else "None"}.

            Explain why these games are relevant based on:
            - Favorite & familiar teams
            - Rivalries
            - Market size
            - Regional loyalty
            - Typical behavior of {segment} bettors
            Compare recommendations vs. actual bets.
            """

            try:
                resp = client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": "You are an NBA betting recommender explainer bot."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400
                )
                msg = resp.choices[0].message
                content = msg["content"] if isinstance(msg, dict) else msg.content
                st.session_state["last_explanation"] = content
            except Exception as e:
                st.error(f"AI explanation failed: {e}")

        if "last_explanation" in st.session_state:
            st.markdown("### 🤖 AI Explanation")
            st.write(st.session_state["last_explanation"])










# --- Tab 3: Simulation ---
with tab3:
    st.subheader("📅 Simulation: Series-Level Performance")
    st.markdown("""
### 🔎 About this Tab
Simulate **series-level betting performance** by comparing:  

- ✅ **Actual players** who bet on each series in the validation set  
- 🤖 **Predicted players** based on Top-5 recommendations  

This simulation is **restricted to the selected playoff round** (e.g., R8, Quarterfinals).  
It helps validate whether the model captures **collective betting behavior** at the series level.
""")

    if "model" not in st.session_state:
        st.warning("⚠️ Train a model first in the 'Train & Evaluate' tab.")
    else:
        try:
            # --- Prepare mappings ---
            mask_map, _, game_map, game_rev = (
                st.session_state["mapping"][0],
                st.session_state["mapping"][1],
                st.session_state["mapping"][2],
                {v: k for k, v in st.session_state["mapping"][2].items()}
            )

            norm_to_pretty = dict(zip(
                playoff_schedule["game_norm"],
                playoff_schedule["game"]
            ))

            # --- Restrict to round games only ---
            round_games = round_sched["game_norm"].unique()
            round_games = [g for g in round_games if g in game_map]

            pred_records = []

            # --- Build Top-5 recommendations per player ---
            for uid in df_validation["mask_id"].unique():
                if uid not in mask_map:
                    continue

                uidx = mask_map[uid]
                gidx = [game_map[g] for g in round_games if g in game_map]

                if not gidx:
                    continue

                scores = st.session_state["model"].predict(
                    uidx, gidx,
                    item_features=st.session_state["item_feats"],
                    user_features=st.session_state["user_feats"]
                )
                recs = sorted(
                    [(game_rev[i], s) for i, s in zip(gidx, scores)],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for g, _ in recs:
                    pretty = norm_to_pretty.get(g, g)
                    pred_records.append({"mask_id": uid, "series": pretty})

            df_pred = pd.DataFrame(pred_records)

            # --- Actual bets for this round ---
            df_actual = df_validation.copy()
            df_actual = df_actual[df_actual["game_norm"].isin(round_games)]
            df_actual["series"] = df_actual["game_norm"].map(norm_to_pretty)
            actual_counts = (
                df_actual.groupby("series")["mask_id"]
                .nunique()
                .reset_index(name="actual_players")
            )

            # --- Predicted counts ---
            pred_counts = (
                df_pred.groupby("series")["mask_id"]
                .nunique()
                .reset_index(name="predicted_players")
            )

            # --- Combine ---
            sim = pd.merge(
                actual_counts, pred_counts,
                on="series", how="outer"
            ).fillna(0)

            sim = sim.sort_values("actual_players", ascending=False)

            st.markdown(f"""
            ✅ Simulation for **{round_choice}** round only.  
            Compare actual vs. predicted bettors per series.  
            """)

            # --- Show Data ---
            st.dataframe(sim)

            # --- Visualization ---
            st.bar_chart(sim.set_index("series")[["actual_players", "predicted_players"]])

        except Exception as e:
            st.error(f"Simulation failed: {e}")





# --- Tab 4: Marketing Analytics ---
# --- Tab 4: Marketing Analytics ---
with tab4:
    st.subheader("📊 Marketing Analytics")
    st.markdown("""
### 🔎 About this Tab
Analyze betting behavior from different marketing perspectives:  

- 💰 **Top Wagered Teams & Players**  
- 👥 **Segment Preferences** (Low, Medium, High activity)  
- 🔀 **Pre-Playoff → Playoff Betting Flow** (team loyalty transitions)  
- 🔥 **Most Anticipated Game** for R8  
- 🤖 **AI Marketing Chatbot** for ad-hoc insights  
""")

    # --- Top charts ---
    st.markdown("### 💰 Top Wagered Teams & Star Players")
    try:
        team_wagers = (
            df_train.explode("teams")
            .groupby("teams")["wager_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(team_wagers)

        player_wagers = (
            df_train.explode("star_players")
            .dropna(subset=["star_players"])
            .groupby("star_players")["wager_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(player_wagers)
    except Exception as e:
        st.error(f"Failed to compute top wagers: {e}")

    # --- Segment preferences ---
    st.markdown("---")
    st.subheader("👥 Segment Preferences")
    try:
        seg_df = df_train.merge(
            pd.Series(player_segments, name="segment"),
            left_on="mask_id",
            right_index=True,
            how="left"
        )
        seg_prefs = (
            seg_df.explode("teams")
            .groupby(["segment", "teams"])
            .size()
            .unstack(fill_value=0)
        )

        st.write("### Activity Segments (Low / Medium / High)")
        st.bar_chart(seg_prefs.loc[:, seg_prefs.columns.isin(big_market_teams)].T)
    except Exception as e:
        st.error(f"Segment analysis failed: {e}")

    # --- Pre-Playoff → Playoff Betting Flow ---
    st.markdown("---")
    st.subheader("🔀 Pre-Playoff → Playoff Betting Flow")
    try:
        playoff_teams = set()
        for g in playoff_schedule["game"]:
            playoff_teams.update([t.strip() for t in g.split("&")])

        # Pre-playoff bets
        pre = df_train[["mask_id", "teams"]].explode("teams")

        # Playoff bets
        post = df_validation[["mask_id", "teams"]].explode("teams")
        post = post[post["teams"].isin(playoff_teams)]

        # Merge and count transitions
        merged = pre.merge(post, on="mask_id", suffixes=("_pre", "_playoff"))
        trans_counts = (
            merged.groupby(["teams_pre", "teams_playoff"])
            .size()
            .reset_index(name="count")
        )

        # Pivot to heatmap
        pivot = (
            trans_counts.pivot(index="teams_pre", columns="teams_playoff", values="count")
            .fillna(0)
        )

        likelihood = pivot.div(pivot.sum(axis=1), axis=0)

        st.write("### Transition Likelihoods (Pre-Playoff → Playoff)")
        st.dataframe(likelihood.style.background_gradient(cmap="Blues"))

        # Heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(likelihood, annot=False, cmap="Blues", ax=ax)
        ax.set_title("Pre-Playoff → Playoff Betting Likelihoods")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Pre-Playoff → Playoff flow failed: {e}")

    # --- Most Anticipated Game (R8) ---
    st.markdown("---")
    st.subheader("🔥 Most Anticipated Game (R8)")
    try:
        r8_sched = playoff_schedule[playoff_schedule["round"] == "R8"]

        if not r8_sched.empty and "model" in st.session_state:
            norm_to_pretty = dict(zip(r8_sched["game_norm"], r8_sched["game"]))
            results = []

            mask_map, _, game_map, _ = (
                st.session_state["mapping"][0],
                st.session_state["mapping"][1],
                st.session_state["mapping"][2],
                {v: k for k, v in st.session_state["mapping"][2].items()}
            )

            for g in r8_sched["game_norm"].unique():
                if g not in game_map:
                    continue
                game_idx = game_map[g]

                # Actual bettors
                actual_players = set(
                    df_validation.loc[df_validation["game_norm"] == g, "mask_id"].unique()
                )

                # Predicted bettors
                predicted_players = set()
                for uid in df_validation["mask_id"].unique():
                    if uid not in mask_map:
                        continue
                    uidx = mask_map[uid]
                    score = st.session_state["model"].predict(
                        uidx, [game_idx],
                        item_features=st.session_state["item_feats"],
                        user_features=st.session_state["user_feats"]
                    )[0]
                    if score > 0:
                        predicted_players.add(uid)

                results.append({
                    "game": norm_to_pretty.get(g, g),
                    "actual_bettors": len(actual_players),
                    "predicted_bettors": len(predicted_players),
                    "gap": len(predicted_players) - len(actual_players)
                })

            results_df = pd.DataFrame(results).sort_values("actual_bettors", ascending=False)

            if not results_df.empty:
                top_game = results_df.iloc[0]
                st.success(
                    f"🔥 The most anticipated R8 series is **{top_game['game']}** "
                    f"with {top_game['actual_bettors']} bettors (validation)."
                )

                st.write("### Actual vs Predicted Bettors per R8 Series")
                st.bar_chart(results_df.set_index("game")[["actual_bettors", "predicted_bettors"]])

                st.write("### Prediction Gaps (Predicted - Actual)")
                st.dataframe(results_df[["game", "actual_bettors", "predicted_bettors", "gap"]])
            else:
                st.info("No valid R8 results to display.")
        else:
            st.info("No R8 series found in schedule or model not trained.")
    except Exception as e:
        st.error(f"R8 analysis failed: {e}")

    # --- Marketing Chatbot ---
    st.markdown("---")
    st.subheader("🤖 Marketing Analytics Chatbot")
    query = st.text_input("Ask a marketing question (e.g., Which team has the most loyal bettors?)")

    if query and "model" in st.session_state:
        try:
            resp = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing analytics assistant for NBA betting insights."},
                    {"role": "user", "content": query}
                ],
                max_tokens=400
            )
            st.markdown("### 📝 AI Marketing Insight")
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Chatbot failed: {e}")

