import numpy as np
import pandas as pd


#list of only training files
training_files = ['train-100-10.csv', 'train-100-100.csv', 'train-1000-100.csv', 'train-50(1000)-100.csv', 'train-100(1000)-100.csv', 'train-150(1000)-100.csv']

# calculate mean squared error (MSE)
def mse(prediction, actual):
    # get average of squared differences between predicted & actual
    err = (1 / len(prediction)) * np.sum(np.square(actual - prediction), axis=0)

    # return as float, in case it's a dataframe
    if isinstance(err, (float, int)):  # avoid KeyType errors
        return float(err)
    else:
        return float(err.iloc[0])  # handle dataframe output


# Calc L2 weights (Ridge Regression)
def L2_weights(x, y, lmbda):
    # x-transpose * x
    xdot = x.T @ x

    # identity matrix for reg term
    I = np.identity(xdot.shape[0])

    # ridge regression: (X^T * X + λ * I)^-1 * X^T * y
    first = np.linalg.inv(xdot + (lmbda * I))
    second = x.T @ y
    return first @ second  # return weights


# extract X and y from CSV
def get_xy(file):
    raw_data = pd.read_csv(file)  # read from csv
    x_without_ones = raw_data.drop('y', axis=1)  # drop target 'y'

    # add ones column for bias term
    ones_column = np.ones((x_without_ones.shape[0], 1))
    x = np.hstack([ones_column, x_without_ones])  # concatenate ones column w/ features

    y = raw_data['y']
    y_df = y.to_frame()

    return x, y_df


best_lambda_per_file = {}  # stores best lambda for each file
average_mse = {}  # stores average MSE for each file

# loop through each training file
for file in training_files:
    x, y = get_xy(file)
    data_per_fold = len(x) // 10  # divide data into 10 folds
    lambdas = {}

    k_train = []
    k_test = []
    # create training & test sets for 10 folds
    for k in range(10):
        start = k * data_per_fold
        end = start + data_per_fold
        x_train = np.concatenate([x[:start], x[end:]])  # get train data
        y_train = np.concatenate([y[:start], y[end:]])  # get train target

        x_test = x[start:end]  # get test data
        y_test = y[start:end]  # get test target

        k_train.append([x_train, y_train])
        k_test.append([x_test, y_test])

    # loop through 0-150 lambdas
    for l in range(0, 151):
        performance = []
        for index, pair in enumerate(k_train):
            x_train = pair[0]  # training data for fold
            y_train = pair[1]  # training target
            x_test = k_test[index][0]  # test data for fold
            y_test = k_test[index][1]  # test target

            # calc weights with current lambda
            weights = L2_weights(x_train, y_train, l)
            # calc test MSE for this fold
            testing_error = mse(x_test @ weights, y_test)
            performance.append(testing_error)  # store fold MSE

        # store avg MSE for current lambda
        lambdas[l] = round(sum(performance) / len(performance), 4)

    # find best lambda (min avg MSE) for curr file
    best_lambda_per_file[file] = min(lambdas, key=lambdas.get)
    average_mse[file] = min(lambdas.values())
    print(f'The lambda with the lowest avg MSE for {file} was {min(lambdas,key=lambdas.get)}, with an MSE of {min(lambdas.values())}')