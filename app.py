import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import base64

# with open("assets/logos/ab.jpeg", "rb") as f:
#     data = base64.b64encode(f.read()).decode()

# st.markdown(f"""
# <style>
# .stApp {{
#     background-image: url("data:image/jpg;base64,{data}");
#     background-size: cover;
#     background-position: center;
#     background-attachment: fixed;
# }}
# </style>
# """, unsafe_allow_html=True)
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

# ------------------ NAVBAR ------------------
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

# ---------- PRECOMPUTE ----------
team_win_counts = df['winner'].value_counts().to_dict()
team_total_matches = {
    t: ((df['team1']==t).sum() + (df['team2']==t).sum())
    for t in set(df['team1']).union(set(df['team2']))
}

# ------------------ SAFE ENCODER ------------------
def safe_encode(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

# ================== HOME ==================
if menu == "🏠 Home":

    st.markdown("""
    <div class="card">
    <h1>🏏 IPL AI Dashboard</h1>
    <p style="text-align:center;">🔥 Match Prediction • Player Stats • IPL History</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏏 Matches", "1150+")
    col2.metric("👥 Teams", "15")
    col3.metric("📅 Seasons", "2008–2025")
    col4.metric("🤖 Accuracy", "75%")

    st.markdown("<h2>📈 Top Teams (2022+)</h2>", unsafe_allow_html=True)
    st.bar_chart(df['winner'].value_counts().head(10))

    # 🔥 FULL FACTS BACK
    st.markdown("""
   <p><b>Mumbai Indians</b> & <b>Chennai Super Kings</b> are the most successful IPL teams 🏆</p> 
<p><b>Royal Challengers Bengaluru (RCB)</b> has one of the biggest fanbases in IPL.</p>

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
<p><b>Sunrisers Hyderabad</b> holds one of the highest team scores (287/3).</p>
<p><b>Royal Challengers Bengaluru</b> won the IPL 2025 title.</p>

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
     😎 Player Spotlight</h3>
    <p><b>Virat Kohli</b> – “King Kohli 👑”</p>
    <p><b>MS Dhoni</b> – “Captain Cool 🧊”</p>
    <p><b>Rohit Sharma</b> – “Hitman 💥”</p>
    <p><b>Suresh Raina</b> – “Mr. IPL 🔥”</p>
    <p><b>AB de Villiers</b> – “Mr. 360 🔄”</p>
    <p><b>Chris Gayle</b> – “Universe Boss 🌌”</p>
    <p><b>Jasprit Bumrah</b> – “Boom Boom Bumrah 💣”</p>
    <p><b>Hardik Pandya</b> – “Kung Fu Pandya 🚀”</p>  
    """, unsafe_allow_html=True)
    

    st.success("👉 Go to Prediction tab!")

# ================== PREDICTION ==================
elif menu == "📊 Prediction":

    st.title("📊 IPL 2026 Match Prediction")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    team_logos = {
        "Mumbai Indians": os.path.join(BASE_DIR,"assets","logos","mi.png"),
        "Chennai Super Kings": os.path.join(BASE_DIR,"assets","logos","csk.png"),
        "Royal Challengers Bengaluru": os.path.join(BASE_DIR,"assets","logos","rcb.png"),
        "Kolkata Knight Riders": os.path.join(BASE_DIR,"assets","logos","kkr.png"),
        "Rajasthan Royals": os.path.join(BASE_DIR,"assets","logos","rr.png"),
        "Sunrisers Hyderabad": os.path.join(BASE_DIR,"assets","logos","srh.png"),
        "Delhi Capitals": os.path.join(BASE_DIR,"assets","logos","dc.png"),
        "Punjab Kings": os.path.join(BASE_DIR,"assets","logos","pbks.png"),
        "Gujarat Titans": os.path.join(BASE_DIR,"assets","logos","gt.png"),
        "Lucknow Super Giants": os.path.join(BASE_DIR,"assets","logos","lsg.png")
    }

    def show_logo(team,w=80):
        path = team_logos.get(team,"")
        if os.path.exists(path):
            st.image(path,width=w)

    team1 = st.selectbox("Team 1", teams)
    team2 = st.selectbox("Team 2", [t for t in teams if t!=team1])
    toss_winner = st.selectbox("Toss Winner",[team1,team2])
    match_type = st.selectbox("Match Type",match_types)

    venue = team_venues.get(team1,"Unknown")
    st.write("📍 Venue:",venue)

    c1,c2,c3 = st.columns(3)
    with c1:
        show_logo(team1); st.write(team1)
    with c2:
        st.markdown("### 🆚")
    with c3:
        show_logo(team2); st.write(team2)

    if st.button("Predict Winner"):

        team1_enc = safe_encode(encoders['team1'], team1)
        team2_enc = safe_encode(encoders['team2'], team2)
        toss_enc = safe_encode(encoders['toss_winner'], toss_winner)
        venue_enc = safe_encode(encoders['venue'], venue)
        match_enc = safe_encode(encoders['match_type'], match_type)

        toss_adv_team1 = 1 if toss_winner==team1 else 0
        is_final = 1 if match_type=="Final" else 0

        team1_strength = team_win_counts.get(team1,0)/team_total_matches.get(team1,1)
        team2_strength = team_win_counts.get(team2,0)/team_total_matches.get(team2,1)

        home_adv_team1 = 1 if team_venues.get(team1)==venue else 0
        home_adv_team2 = 1 if team_venues.get(team2)==venue else 0

        data = np.array([[team1_enc,team2_enc,toss_enc,venue_enc,match_enc,
                          toss_adv_team1,is_final,
                          team1_strength,team2_strength,
                          home_adv_team1,home_adv_team2]])

        prob = model.predict_proba(data)[0]
        classes = model.classes_

        t1c = safe_encode(encoders['winner'], team1)
        t2c = safe_encode(encoders['winner'], team2)

        team1_prob = prob[list(classes).index(t1c)]*100
        team2_prob = prob[list(classes).index(t2c)]*100

        team1_final = 0.6*team1_prob + 0.4*team1_strength*100
        team2_final = 0.6*team2_prob + 0.4*team2_strength*100

        winner = team1 if team1_final>team2_final else team2

        st.markdown(f"<h2 style='text-align:center;'>🏆 {winner}</h2>",unsafe_allow_html=True)
        show_logo(winner,120)
        st.balloons()

        st.write(team1,team1_final)
        st.write(team2,team2_final)

        st.bar_chart(pd.DataFrame({"Team":[team1,team2],"Score":[team1_final,team2_final]}).set_index("Team"))

        st.write("🔍 Confidence:",abs(team1_final-team2_final))
        
     # Head-to-Head
        h2h = df[
            ((df["team1"] == team1) & (df["team2"] == team2)) |
            ((df["team1"] == team2) & (df["team2"] == team1))
        ]

        st.subheader("📈 Head to Head")
        st.write(f"{team1} wins: {len(h2h[h2h['winner']==team1])}")
        st.write(f"{team2} wins: {len(h2h[h2h['winner']==team2])}")

# ================== CHAMPIONS ==================
elif menu == "🏆 Champions":

    st.title("🏆 IPL Winners")

    # ---------- IPL WINNERS ----------
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

    st.subheader("🏆 Champions List")
    for y,w,r in ipl_winners:
        st.markdown(f"""
        <div class="card">
        <h3>{y}</h3>
        🥇 {w}<br>
        🥈 {r}
        </div>
        """, unsafe_allow_html=True)


    # ---------- ORANGE CAP ----------
    st.subheader("🟠 Orange Cap Winners (Top Run Scorer)")

    orange_cap = [
        ("2008","Shaun Marsh"),
        ("2009","Matthew Hayden"),
        ("2010","Sachin Tendulkar"),
        ("2011","Chris Gayle"),
        ("2012","Chris Gayle"),
        ("2013","Michael Hussey"),
        ("2014","Robin Uthappa"),
        ("2015","David Warner"),
        ("2016","Virat Kohli"),
        ("2017","David Warner"),
        ("2018","Kane Williamson"),
        ("2019","David Warner"),
        ("2020","KL Rahul"),
        ("2021","Ruturaj Gaikwad"),
        ("2022","Jos Buttler"),
        ("2023","Shubman Gill"),
        ("2024","Virat Kohli"),
        ("2025","Dev")
    ]

    for y,p in orange_cap:
        st.write(f"{y} - 🟠 {p}")


    # ---------- PURPLE CAP ----------
    st.subheader("🟣 Purple Cap Winners (Top Wicket Taker)")

    purple_cap = [
        ("2008","Sohail Tanvir"),
        ("2009","RP Singh"),
        ("2010","Pragyan Ojha"),
        ("2011","Lasith Malinga"),
        ("2012","Morne Morkel"),
        ("2013","Dwayne Bravo"),
        ("2014","Mohit Sharma"),
        ("2015","Dwayne Bravo"),
        ("2016","Bhuvneshwar Kumar"),
        ("2017","Bhuvneshwar Kumar"),
        ("2018","Andrew Tye"),
        ("2019","Imran Tahir"),
        ("2020","Kagiso Rabada"),
        ("2021","Harshal Patel"),
        ("2022","Yuzvendra Chahal"),
        ("2023","Mohammed Shami"),
        ("2024","Harshal Patel"),
        ("2025","Prasidh Krishna")
    ]

    for y,p in purple_cap:
        st.write(f"{y} - 🟣 {p}")

# ================== PLAYER ==================
elif menu == "🔎 Player Stats":

    st.title("🔎 Player Stats")

    # 🔥 FULL PLAYER DATA (RESTORED)
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

        # 🔥 CARD UI (same as tera original)
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

        # 📊 Chart
        st.subheader("📊 Performance Overview")
        st.bar_chart({
            "Runs": [data['runs']],
            "Fours": [data['fours']],
            "Sixes": [data['sixes']],
            "Wickets": [data['wickets']]
        })

# ================== GALLERY ==================
elif menu == "🖼 Gallery":

    base_path="images"

    if not os.path.exists(base_path):
        st.error("Images folder not found")
        st.stop()

    seasons=os.listdir(base_path)
    s=st.selectbox("Season",seasons)
    folder=os.path.join(base_path,s)

    imgs=[f for f in os.listdir(folder) if f.endswith(("png","jpg","jpeg"))]

    cols=st.columns(3)
    for i,img in enumerate(imgs):
        with cols[i%3]:
            st.image(os.path.join(folder,img))
