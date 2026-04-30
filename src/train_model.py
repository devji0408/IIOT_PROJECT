import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/ipl_data.csv")

# ---------------- FILTER RECENT DATA ----------------
df = df[df['season'] >= 2022]

# ---------------- CLEAN TEAM NAMES ----------------
team_mapping = {
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru"
}

for col in ['team1','team2','toss_winner','winner']:
    df[col] = df[col].replace(team_mapping)

# ---------------- SELECT REQUIRED COLUMNS ----------------
df = df[['team1','team2','toss_winner','venue','match_type','winner']]
df = df.dropna()

# ---------------- CLEAN VENUE (CITY FORMAT) ----------------
df['venue'] = df['venue'].str.strip()

# ---------------- FEATURE ENGINEERING ----------------

# 1️⃣ Toss Advantage (team1 = 1 if toss won)
df['toss_adv_team1'] = (df['toss_winner'] == df['team1']).astype(int)
df['toss_adv_team2'] = (df['toss_winner'] == df['team2']).astype(int)

# 2️⃣ Final Match Flag (pressure effect)
df['is_final'] = (df['match_type'] == "Final").astype(int)

# 3️⃣ Team Strength (past wins ratio)
team_win_counts = df['winner'].value_counts().to_dict()
team_total_matches = {}

for team in set(df['team1']).union(set(df['team2'])):
    team_total_matches[team] = ((df['team1'] == team).sum() + (df['team2'] == team).sum())

df['team1_strength'] = df['team1'].map(lambda x: team_win_counts.get(x,0)/team_total_matches.get(x,1))
df['team2_strength'] = df['team2'].map(lambda x: team_win_counts.get(x,0)/team_total_matches.get(x,1))

# 4️⃣ Home Advantage (if team1 is playing in its usual home venue)
# Example: define home venues for teams
home_venues = {
"Royal Challengers Bengaluru":"Bengaluru","Chennai Super Kings":"Chennai",
"Mumbai Indians":"Mumbai","Kolkata Knight Riders":"Kolkata",
"Delhi Capitals":"Delhi","Rajasthan Royals":"Jaipur",
"Sunrisers Hyderabad":"Hyderabad","Punjab Kings":"Chandigarh",
"Gujarat Titans":"Ahmedabad","Lucknow Super Giants":"Lucknow"
}

df['home_adv_team1'] = df.apply(lambda row: 1 if home_venues.get(row['team1'],'')==row['venue'] else 0, axis=1)

# Fill missing values
df = df.fillna(0)

# ---------------- LABEL ENCODING ----------------
encoders = {}
for col in ['team1','team2','toss_winner','venue','match_type','winner']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- FEATURES ----------------
X = df.drop("winner", axis=1)
y = df["winner"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_score = cross_val_score(model, X, y, cv=5).mean()

print(f"✅ Accuracy: {acc:.4f}")
print(f"🔥 Cross Validation Score: {cv_score:.4f}")

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ Model + Encoders saved successfully")