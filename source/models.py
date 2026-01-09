import numpy as np

def linear_regression(X_train, y_train, X_test, y_test):
    
    #design matrix X
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    #target vector y
    y = y_train.reshape(-1, 1)

    #closed-form solution (normal equation) for parameter vector beta
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    #predictions on test set
    X_test_design = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_pred = X_test_design @ beta

    return y_pred.flatten()


def decision_tree(X_train, y_train, X_test, y_test):
    pass