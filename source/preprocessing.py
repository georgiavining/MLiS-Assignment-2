import pandas as pd
import numpy as np

def create_SIE_df(type: str):
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
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    return df_lagged

def merge_temperature_data(SIE_df, temp_df, temp_columns: list, temp_column_names: list = None):
    
    for col in temp_columns:
        temp_df[col] = pd.to_numeric(temp_df[col].replace('***', np.nan), errors='coerce')

    merge_keys = ["Year"]
    if "Month" in SIE_df.columns:
        merge_keys.append("Month")

    merged_df = SIE_df.merge(
        temp_df[merge_keys + temp_columns], 
        on = merge_keys,
        how="inner"
    )

    if temp_column_names is not None:
        merged_df = merged_df.rename(columns=dict(zip(temp_columns, temp_column_names)))
    
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    return merged_df

def merge_co2_data(SIE_df, co2_df, co2_column_name="CO2 Concentration"):
    merge_keys = ["Year"]

    co2_df = co2_df.rename(columns={
        "year": "Year",
        "month": "Month",   
        "average": co2_column_name,
        "mean": co2_column_name
    })

    merge_keys = ["Year"]
    if "Month" in co2_df.columns and "Month" in SIE_df.columns:
        merge_keys.append("Month")

    merged_df = SIE_df.merge(
        co2_df[merge_keys + [co2_column_name]],
        on=merge_keys,
        how="inner"
    )

    merged_df = merged_df.dropna().reset_index(drop=True)
    return merged_df

