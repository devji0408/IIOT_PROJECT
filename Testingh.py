import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

# ------------------ LOAD CSS ------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------------ LOAD MODEL ------------------
with open("model/model.pkl","rb") as f:
    model = pickle.load(f)

with open("model/encoders.pkl","rb") as f:
    encoders = pickle.load(f)

# ------------------ NAVBAR (FIXED) ------------------
menu = st.radio(
    "",
    ["🏠 Home","📊 Prediction","🏆 Champions","🔎 Player Stats","🖼 Gallery"],
    horizontal=True
)

# ------------------ TEAMS ------------------
teams = [
"Royal Challengers Bengaluru","Chennai Super Kings","Mumbai Indians",
"Kolkata Knight Riders","Delhi Capitals","Rajasthan Royals",
"Sunrisers Hyderabad","Punjab Kings","Gujarat Titans","Lucknow Super Giants"
]

team_venues = {
"Royal Challengers Bengaluru":"Bengaluru","Chennai Super Kings":"Chennai",
"Mumbai Indians":"Mumbai","Kolkata Knight Riders":"Kolkata",
"Delhi Capitals":"Delhi","Rajasthan Royals":"Jaipur",
"Sunrisers Hyderabad":"Hyderabad","Punjab Kings":"Chandigarh",
"Gujarat Titans":"Ahmedabad","Lucknow Super Giants":"Lucknow"
}

match_types = ["League","Qualifier","Eliminator","Final"]

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/ipl_data.csv")
    df = df[df['season'] >= 2022]

    team_mapping = {
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru"
    }

    for col in ['team1','team2','toss_winner','winner']:
        df[col] = df[col].replace(team_mapping)

    return df

df = load_data()

# ------------------ SAFE ENCODER ------------------
def safe_encode(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

# ================== HOME ==================
if menu == "🏠 Home":

    # 🔥 HERO SECTION
    st.markdown("""
    <div class="card">
    <h1>🏏 IPL AI Dashboard</h1>
    <p style="text-align:center; font-size:18px;">
    🔥 Match Prediction • Player Stats • IPL History
    </p>
    </div>
    """, unsafe_allow_html=True)


    # 🔥 QUICK STATS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🏏 Matches", "1150+")
    col2.metric("👥 Teams", "15")
    col3.metric("📅 Seasons", "2008–2025")
    col4.metric("🤖 Accuracy", "75%")


    # 🔥 FEATURES SECTION
    st.markdown("<h2 style='text-align:center;'>🚀 Features</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>📊 Match Prediction</h3>
        <p>Predict match winners using ML model</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h3>🔎 Player Stats</h3>
        <p>Search any player and view performance</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🏆 IPL Winners</h3>
        <p>Explore winners from 2008 to 2025</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h3>🖼️ Gallery</h3>
        <p>Stadiums, moments & highlights</p>
        </div>
        """, unsafe_allow_html=True)


    # 🔥 CHART SECTION
    st.markdown("<h2 style='text-align:center;'>📈 Top Performance from 2022</h2>", unsafe_allow_html=True)

    if 'winner' in df.columns:
        chart_data = df['winner'].value_counts().head(10)
        st.bar_chart(chart_data)


    # 🔥 FUN FACT
    
    st.markdown("""
    <div class="card">
    <h3>🔥 Did You Know?</h3>
    <p>Mumbai Indians & Chennai Super Kings are the most successful IPL teams 🏆</p> 
    <p><b>RCB</b> has one of the biggest fanbases in the IPL.</p>
 

    <p><b>Virat Kohli</b> has scored the most runs in IPL history.</p>
    <p><b>Virat Kohli</b> has the most centuries in IPL.</p>
    <p><b>Chris Gayle</b> has hit the most sixes in IPL.</p>
    <p><b>Shikhar Dhawan</b> has hit the most fours in IPL.</p>
    <p><b>David Warner</b> has scored the most fifties in IPL.</p>

    <p><b>Yuzvendra Chahal</b> has taken the most wickets in IPL.</p>
    <p><b>Alzarri Joseph</b> holds the best bowling figures (6/12) in IPL.</p>
    <p><b>Amit Mishra</b> has taken the most hat-tricks in IPL.</p>
    <p><b>Sunil Narine</b> has one of the best economy rates in IPL history.</p>
    <p><b>Bhuvneshwar Kumar</b> has bowled the most dot balls in IPL.</p>

    <p><b>Mumbai Indians</b> have won the most IPL titles.</p>
    <p><b>Chennai Super Kings</b> have played the most IPL finals.</p>
    <p><b>Mumbai Indians</b> have the most wins in IPL history.</p>
    <p><b>Sunrisers Hyderabad</b> holds the highest team score (287/3).</p>
    <p><b>Royal Challengers Bangalore</b> won the IPL 2025 title.</p>
     <p><b>RCB</b> has a strong brand value in world cricket.</p>

    <p><b>Chris Gayle</b> scored the fastest century in IPL (30 balls).</p>
    <p><b>Yashasvi Jaiswal</b> scored the fastest fifty in IPL (13 balls).</p>
    <p><b>MS Dhoni</b> has played the most matches in IPL.</p>
    <p><b>Suresh Raina</b> has taken the most catches in IPL.</p>
    <p><b>AB de Villiers</b> has won the most Man of the Match awards in IPL.</p>

    <p><b>Rajasthan Royals</b> won the first IPL season in 2008.</p>
    <p><b>Gujarat Titans</b> won the title in their debut season (2022).</p>
    <p><b>Chennai Super Kings</b> have qualified for the playoffs the most times.</p>
    <p><b>Mumbai Indians</b> have recorded the most successful title wins in finals.</p>
                    
    <p><b>Kolkata Knight Riders</b> have won multiple IPL titles with strong performances.</p>
    <p><b>Sunrisers Hyderabad</b> are known for having one of the best bowling units.</p>
    <p><b>Lucknow Super Giants</b> have shown strong performances despite being a new team.</p> 
    <h3>😎 Player Spotlight</h3>
    <p><b>Virat Kohli</b> is popularly known as “King Kohli 👑”.</p>
    <p><b>MS Dhoni</b> is called “Captain Cool 🧊”.</p>
    <p><b>Rohit Sharma</b> is known as “Hitman 💥”.</p>
    <p><b>Suresh Raina</b> is called “Mr. IPL🔥”.</p>
    <p><b>AB de Villiers</b> is famous as “Mr. 360 🔄”.</p>

    <p><b>Chris Gayle</b> is known as “Universe Boss 🌌”.</p>
    <p><b>Jasprit Bumrah</b> is called “Boom Boom Bumrah💣”.</p>
    <p><b>Hardik Pandya</b> is known as “Kung Fu Pandya🚀”.</p>  
    </div>
                
    """, unsafe_allow_html=True)


    # 🔥 CTA
    st.success("👉 Go to Prediction tab and predict your next match winner!")

# ================== PREDICTION PAGE ==================
elif menu == "📊 Prediction":

    import streamlit as st
    import numpy as np
    import pandas as pd
    import os

    st.title("📊 IPL 2026 Match Prediction")

    # ---------- BASE PATH ----------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ---------- TEAM LOGOS ----------
    team_logos = {
        "Mumbai Indians": os.path.join(BASE_DIR, "assets", "logos", "mi.png"),
        "Chennai Super Kings": os.path.join(BASE_DIR, "assets", "logos", "csk.png"),
        "Royal Challengers Bengaluru": os.path.join(BASE_DIR, "assets", "logos", "rcb.png"),
        "Kolkata Knight Riders": os.path.join(BASE_DIR, "assets", "logos", "kkr.png"),
        "Rajasthan Royals": os.path.join(BASE_DIR, "assets", "logos", "rr.png"),
        "Sunrisers Hyderabad": os.path.join(BASE_DIR, "assets", "logos", "srh.png"),
        "Delhi Capitals": os.path.join(BASE_DIR, "assets", "logos", "dc.png"),
        "Punjab Kings": os.path.join(BASE_DIR, "assets", "logos", "pbks.png"),
        "Gujarat Titans": os.path.join(BASE_DIR, "assets", "logos", "gt.png"),
        "Lucknow Super Giants": os.path.join(BASE_DIR, "assets", "logos", "lsg.png")
    }

    # ---------- SAFE IMAGE FUNCTION ----------
    def show_logo(team, width=80):
        path = team_logos.get(team, "")
        if os.path.exists(path):
            st.image(path, width=width)
        else:
            st.warning(f"Logo missing: {team}")

    # --- Inputs ---
    team1 = st.selectbox("Team 1", teams)
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    match_type = st.selectbox("Match Type", match_types)

    # --- Venue ---
    venue = team_venues.get(team1, "Unknown")
    st.write("📍 Venue:", venue)

    # ---------- TEAM VS UI ----------
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        show_logo(team1)
        st.write(team1)
    with col2:
        st.markdown("### 🆚")
    with col3:
        show_logo(team2)
        st.write(team2)

    # ================= PREDICTION =================
    if st.button("Predict Winner"):

        # --- Encoding ---
        team1_enc = safe_encode(encoders['team1'], team1)
        team2_enc = safe_encode(encoders['team2'], team2)
        toss_enc = safe_encode(encoders['toss_winner'], toss_winner)
        venue_enc = safe_encode(encoders['venue'], venue)
        match_enc = safe_encode(encoders['match_type'], match_type)

        toss_adv_team1 = 1 if toss_winner == team1 else 0
        toss_adv_team2 = 1 if toss_winner == team2 else 0
        is_final = 1 if match_type == "Final" else 0

        # --- Strength ---
        team_win_counts = df['winner'].value_counts().to_dict()
        team_total_matches = {}
        for t in set(df['team1']).union(set(df['team2'])):
            team_total_matches[t] = ((df['team1']==t).sum() + (df['team2']==t).sum())

        team1_strength = team_win_counts.get(team1,0)/team_total_matches.get(team1,1)
        team2_strength = team_win_counts.get(team2,0)/team_total_matches.get(team2,1)

        # --- Home Advantage ---
        home_venues = {
            "Royal Challengers Bengaluru":"Bengaluru","Chennai Super Kings":"Chennai",
            "Mumbai Indians":"Mumbai","Kolkata Knight Riders":"Kolkata",
            "Delhi Capitals":"Delhi","Rajasthan Royals":"Jaipur",
            "Sunrisers Hyderabad":"Hyderabad","Punjab Kings":"Chandigarh",
            "Gujarat Titans":"Ahmedabad","Lucknow Super Giants":"Lucknow"
        }

        home_adv_team1 = 1 if home_venues.get(team1,'')==venue else 0
        home_adv_team2 = 1 if home_venues.get(team2,'')==venue else 0

        # --- Head-to-Head ---
        h2h_team1 = ((df['team1']==team1) & (df['team2']==team2) & (df['winner']==team1)).sum() + \
                    ((df['team1']==team2) & (df['team2']==team1) & (df['winner']==team1)).sum()
        h2h_team2 = ((df['team1']==team1) & (df['team2']==team2) & (df['winner']==team2)).sum() + \
                    ((df['team1']==team2) & (df['team2']==team1) & (df['winner']==team2)).sum()

        total_h2h = h2h_team1 + h2h_team2
        if total_h2h > 0:
            h2h_team1_norm = h2h_team1 / total_h2h
            h2h_team2_norm = h2h_team2 / total_h2h
        else:
            h2h_team1_norm = 0.5
            h2h_team2_norm = 0.5

        # --- Model Input (FIXED) ---
        data = np.array([[team1_enc, team2_enc, toss_enc, venue_enc, match_enc,
                          toss_adv_team1, is_final,
                          team1_strength, team2_strength,
                          home_adv_team1, home_adv_team2]])

        # --- Model Prediction ---
        prob = model.predict_proba(data)[0]
        classes = model.classes_

        team1_class = safe_encode(encoders['winner'], team1)
        team2_class = safe_encode(encoders['winner'], team2)

        team1_prob = prob[list(classes).index(team1_class)] * 100
        team2_prob = prob[list(classes).index(team2_class)] * 100

        # --- Improved Logic ---
        logic_team1 = (
            0.25 * (1 if toss_winner == team1 else 0) +
            0.10 * home_adv_team1 +
            0.25 * h2h_team1_norm +
            0.30 * team1_strength
        ) * 100

        logic_team2 = (
            0.25 * (1 if toss_winner == team2 else 0) +
            0.10 * home_adv_team2 +
            0.25 * h2h_team2_norm +
            0.30 * team2_strength
        ) * 100

        # --- Final Score ---
        team1_final = 0.5 * logic_team1 + 0.5 * team1_prob
        team2_final = 0.5 * logic_team2 + 0.5 * team2_prob

        # --- Winner ---
        if team1_final > team2_final:
            winner_text = team1
        elif team2_final > team1_final:
            winner_text = team2
        else:
            winner_text = "Tie likely"

        # ---------- DISPLAY ----------
        st.markdown(f"""
        <div style="text-align:center;">
            <h2 style="color:#22c55e;">🏆 Winner: {winner_text}</h2>
        </div>
        """, unsafe_allow_html=True)

        if winner_text != "Tie likely":
            show_logo(winner_text, width=120)

        # Scores
        st.write(f"{team1} Final Score: {team1_final:.2f}")
        st.write(f"{team2} Final Score: {team2_final:.2f}")

        # Chart
        chart_data = pd.DataFrame({
            'Team': [team1, team2],
            'Score': [team1_final, team2_final]
        })
        st.bar_chart(chart_data.set_index('Team'))

        # Probability
        st.subheader("📊 Win Probability")
        st.progress(int(team1_final))
        st.write(f"{team1}: {team1_final:.1f}%")

        st.progress(int(team2_final))
        st.write(f"{team2}: {team2_final:.1f}%")

        # Head-to-Head
        h2h = df[
            ((df["team1"] == team1) & (df["team2"] == team2)) |
            ((df["team1"] == team2) & (df["team2"] == team1))
        ]

        st.subheader("📈 Head to Head")
        st.write(f"{team1} wins: {len(h2h[h2h['winner']==team1])}")
        st.write(f"{team2} wins: {len(h2h[h2h['winner']==team2])}")



# ================== WINNERS ==================
elif menu == "🏆 Champions":

    st.title("🏆 IPL Winners")

    ipl_winners = [
    ("2008","Rajasthan Royals","Chennai Super Kings"),
    ("2009","Deccan Chargers","Royal Challengers Bengaluru"),
    ("2010","Chennai Super Kings","Mumbai Indians"),
    ("2011","Chennai Super Kings","Royal Challengers Bengaluru"),
    ("2012","Kolkata Knight Riders","Chennai Super Kings"),
    ("2013","Mumbai Indians","Chennai Super Kings"),
    ("2014","Kolkata Knight Riders","Punjab Kings"),
    ("2015","Mumbai Indians","Chennai Super Kings"),
    ("2016","Sunrisers Hyderabad","Royal Challengers Bengaluru"),
    ("2017","Mumbai Indians","Rising Pune Supergiant"),
    ("2018","Chennai Super Kings","Sunrisers Hyderabad"),
    ("2019","Mumbai Indians","Chennai Super Kings"),
    ("2020","Mumbai Indians","Delhi Capitals"),
    ("2021","Chennai Super Kings","Kolkata Knight Riders"),
    ("2022","Gujarat Titans","Rajasthan Royals"),
    ("2023","Chennai Super Kings","Gujarat Titans"),
    ("2024","Kolkata Knight Riders","Sunrisers Hyderabad"),
    ("2025","Royal Challengers Bengaluru","Punjab Kings")
    ]

    for y,w,r in ipl_winners:
        st.markdown(f"""
        <div class="card">
        <h3>{y}</h3>
        🥇 {w}<br>
        🥈 {r}
        </div>
        """, unsafe_allow_html=True)






# ================== PLAYER ==================
elif menu == "🔎 Player Stats":

    st.title("🔎 Player Stats")

    # Player Data
    players_data = {
        "Virat Kohli": {"role": "Batter", "runs": 8004, "wickets": 4, "matches": 252, "fours": 705, "sixes": 265},
        "Rohit Sharma": {"role": "Batter", "runs": 6628, "wickets": 15, "matches": 257, "fours": 590, "sixes": 280},
        "MS Dhoni": {"role": "Wicketkeeper", "runs": 5243, "wickets": 0, "matches": 264, "fours": 360, "sixes": 252},
        "Shikhar Dhawan": {"role": "Batter", "runs": 6769, "wickets": 4, "matches": 222, "fours": 768, "sixes": 152},
        "Chris Gayle": {"role": "Batter", "runs": 4965, "wickets": 18, "matches": 142, "fours": 404, "sixes": 357},
        "David Warner": {"role": "Batter", "runs": 6565, "wickets": 0, "matches": 184, "fours": 663, "sixes": 236},
        "AB de Villiers": {"role": "Batter", "runs": 5162, "wickets": 2, "matches": 184, "fours": 413, "sixes": 251},
        "Hardik Pandya": {"role": "Allrounder", "runs": 2525, "wickets": 64, "matches": 137, "fours": 180, "sixes": 136},
        "Ravindra Jadeja": {"role": "Allrounder", "runs": 2959, "wickets": 160, "matches": 240, "fours": 210, "sixes": 110},
        "Jasprit Bumrah": {"role": "Bowler", "runs": 70, "wickets": 165, "matches": 138, "fours": 5, "sixes": 2},
        "Andre Russell": {"role": "Allrounder", "runs": 2484, "wickets": 110, "matches": 124, "fours": 170, "sixes": 193}
    }

    # 🔽 Player Select Dropdown
    player = st.selectbox("Select Player", list(players_data.keys()))

    if player:
        data = players_data[player]

        st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background:#0f172a;
            color:white;
            box-shadow:0 0 10px rgba(0,0,0,0.5);
        ">
        <h2>{player}</h2>
        <hr>
        <b>Role:</b> {data['role']}<br>
        <b>Matches:</b> {data['matches']}<br>
        <b>Runs:</b> {data['runs']}<br>
        <b>Wickets:</b> {data['wickets']}<br>
        <b>Fours:</b> {data['fours']}<br>
        <b>Sixes:</b> {data['sixes']}
        </div>
        """, unsafe_allow_html=True)

        # Chart
        st.subheader("📊 Performance Overview")
        st.bar_chart({
            "Runs": [data['runs']],
            "Fours": [data['fours']],
            "Sixes": [data['sixes']],
            "Wickets": [data['wickets']]
        })

# ================== IMAGE GALLERY ==================
elif menu == "🖼 Gallery":

    st.title("🖼️ IPL Gallery (Season wise)")

    base_path = "images"

    # List seasons (subfolders)
    seasons = [s for s in sorted(os.listdir(base_path)) if os.path.isdir(os.path.join(base_path, s))]
    selected_season = st.selectbox("Select Season", seasons)

    # Path to selected season
    folder_path = os.path.join(base_path, selected_season)

    # List only image files
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

    if not images:
        st.info("No images in this season yet!")
    else:
        cols = st.columns(3)
        for i, img_file in enumerate(images):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            with cols[i % 3]:
                st.image(img, use_container_width=True, caption=img_file)