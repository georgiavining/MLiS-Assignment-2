import pandas as pd


def create_df(type: str):
    df = pd.read_csv("../data/N_seaice_extent_daily_v4.0.csv", skiprows=1) #skipping 1st row

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "YYYY": "Year",
        "MM": "Month",
        "DD": "Day",
        "10^6 sq km": "Extent",
        "10^6 sq km.1": "Missing"
    })

    df = df.drop(columns=[col for col in df.columns if "Source data" in col])

    if type == "yearly":
        df = df.groupby(['Year'])['Extent'].mean().reset_index()

    elif type == "monthly":
        df = df.groupby(['Year', 'Month'])['Extent'].mean().reset_index()

    return df


def train_test_split(df, year_split: int):
    train = df[df["Year"] <= year_split]
    test = df[df["Year"] > year_split]

    X_train = train.drop(columns=["Extent"]).values
    y_train = train["Extent"].values

    X_test = test.drop(columns=["Extent"]).values
    y_test = test["Extent"].values

    return X_train, y_train, X_test, y_test

def create_lagged_features(df, lags: list):
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'lag_{lag}'] = df_lagged['Extent'].shift(lag)
    df_lagged = df_lagged.dropna()
    return df_lagged


