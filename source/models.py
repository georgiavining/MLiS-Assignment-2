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
    

def finding_best_split(X,y):
    n_samples, n_features = X.shape

    #setting initial values to be overridden during loop
    best_mse = float('inf')
    best_feature, best_threshold = None, None

    #checking all features and thresholds for the best split

    for feature in range(n_features):
        #getting unique values in each feature array to use as thresholds
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left = X[:, feature] <= threshold
            right = X[:, feature] > threshold

            #moving on if one side of the split is empty
            if len(y[left]) == 0 or len(y[right]) == 0:
                continue

            #calculating weighted mse for the split
            weighted_mse = ((len(y[left]) / n_samples) * np.var(y[left])) + ((len(y[right]) / n_samples) * np.var(y[right]))

            if weighted_mse < best_mse:
                best_mse = weighted_mse
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_mse


#building a regression tree recursively

def build_tree_regressor(X, y, depth=0, max_depth=5):
    n_samples, n_features = X.shape

    # recursion base case (leaf node) --> return mean value
    if depth >= max_depth or n_samples <= 1:
        return np.mean(y)

    feature, threshold, mse = finding_best_split(X, y)

    if feature is None:
        return np.mean(y)

    left = X[:, feature] <= threshold
    right = X[:, feature] > threshold

    # recursive calls for building left and right subtrees
    left_subtree = build_tree_regressor(X[left], y[left], depth + 1, max_depth)
    right_subtree = build_tree_regressor(X[right], y[right], depth + 1, max_depth)

    #returning decision node as a tuple
    return (feature, threshold, left_subtree, right_subtree)

def predict_tree_regressor(tree, X):
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples)

    for i in range(n_samples): #iterating through each sample
        node = tree

        #traversing the tree until reaching a leaf node
        while isinstance(node, tuple):
            feature, threshold, left_subtree, right_subtree = node
            if X[i, feature] <= threshold:
                node = left_subtree
            else:
                node = right_subtree
        y_pred[i] = node  # leaf node value

    return y_pred


def ridge_regression(X_train, y_train, X_test, y_test, lambda_):
    
    #design matrix X
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    #target vector y
    y = y_train.reshape(-1, 1)

    #closed-form solution for ridge regression 
    n_features = X.shape[1]
    I = np.eye(n_features) # identity matrix
    I[0, 0] = 0  # excluding intercept term from regularisation
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y

    #predictions on test set
    X_test_design = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_pred = X_test_design @ beta

    return y_pred.flatten()

def find_best_regularisation_parameter(X_train, y_train, X_val, y_val, lambda_values):
    best_lambda = None
    best_mse = float('inf')

    for lambda_ in lambda_values:
        y_val_pred = ridge_regression(X_train, y_train, X_val, y_val, lambda_)
        mse = 1/len(y_val) * sum((y_val - y_val_pred)**2)

        if mse < best_mse:
            best_mse = mse
            best_lambda = lambda_

    return best_lambda