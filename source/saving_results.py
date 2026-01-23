from numpy import sqrt
import pandas as pd
from pathlib import Path

def save_results_to_csv(y_test, y_pred, model_name: str, features: str, target: str, train_period: str, test_period: str):
    mae =  1/len(y_test) * sum(abs(y_test - y_pred))
    mse = 1/len(y_test) * sum((y_test - y_pred)**2)
    rmse = sqrt(mse)
    r2 = 1 - (sum((y_test - y_pred)**2) / sum((y_test - y_test.mean())**2))

    print(f"mse = {mse}, mae = {mae}, rmse={rmse}, r2= {r2}")

    results = {
        "model": model_name,
        "features": features,
        "target": target,
        "train_period": train_period,
        "test_period": test_period,
        "MSE": mse,
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