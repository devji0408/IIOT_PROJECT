import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df[['team1','team2','toss_winner','venue','match_type','winner']]

    df = df.dropna()

    le = LabelEncoder()

    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('winner', axis=1)
    y = df['winner']

    return X, y