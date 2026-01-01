from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy import sqrt
import pandas as pd
from pathlib import Path

def save_results_to_csv(y_test, y_pred, model_name: str, features: str, target: str, train_period: str, test_period: str):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"mae = {mae}, rmse={rmse}, r2= {r2}")

    results = {
        "model": model_name,
        "features": features,
        "target": target,
        "train_period": train_period,
        "test_period": test_period,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


    #adding results dictionary to a new data frame so that i can easily compare models throughout
    results_df = pd.DataFrame([results])

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "regression_model_results.csv"

    #converting it into a csv file
    results_df.to_csv(
        results_path,
        mode = "a",
        header=not results_path.exists(),
        index=False
    )