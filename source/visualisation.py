import matplotlib.pyplot as plt
import numpy as np

def plot_xy(
    x,
    y,
    plot_type: str= "line",
    title: str  = "",
    xlabel: str = "",
    ylabel: str = "",
    regression_line: bool = False
):
    
    plt.figure(figsize=(10, 5))

    if plot_type == "line":
        plt.plot(x,y)

    elif plot_type == "scatter":
        plt.scatter(x,y)
        if regression_line:
            x = np.array(x).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)
            x_regression = np.hstack((np.ones((x.shape[0], 1)), x))
            beta = np.linalg.inv(x_regression.T @ x_regression) @ x_regression.T @ y
            y_pred = x_regression @ beta
            plt.plot(x, y_pred, color='red', linewidth=2)

    elif plot_type == "bar":
        plt.bar(x,y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_type != "bar":
        plt.grid(True)
    plt.show()
