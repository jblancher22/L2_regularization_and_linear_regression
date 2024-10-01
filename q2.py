import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### creates smaller datasets from original
# data = pd.read_csv('train-1000-100.csv')
#
# # first 50 rows to new CSV
# data50 = data.iloc[0:50]
# data50.to_csv('train-50(1000)-100.csv', index=False)
#
# # first 100 rows to new CSV
# data100 = data.iloc[0:100]
# data100.to_csv('train-100(1000)-100.csv', index=False)
#
# # first 150 rows to new CSV
# data150 = data.iloc[0:150]
# data150.to_csv('train-150(1000)-100.csv', index=False)

# list of file pairs with training & test sets
files = [['train-100-10.csv', 'test-100-10.csv'],
         ['train-100-100.csv', 'test-100-100.csv'],
         ['train-1000-100.csv', 'test-1000-100.csv'],
         ['train-50(1000)-100.csv', 'test-1000-100.csv'],
         ['train-100(1000)-100.csv', 'test-1000-100.csv'],
         ['train-150(1000)-100.csv', 'test-1000-100.csv']]


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


# loop through each train-test pair
for file_set in files:
    lambda_list = list(range(0,151))
    y_train_list = []
    y_test_list = []

    for l in range(0, 151):

        # get X and y for training set
        x_train, y_train = get_xy(file_set[0])
        # calc weights for current lambda
        weights = L2_weights(x_train, y_train, l)
        # get training MSE
        training_error = mse(x_train @ weights, y_train)
        y_train_list.append(training_error)

        # get X and y for test set
        x_test, y_test = get_xy(file_set[1])
        # get test MSE
        testing_error = mse(x_test @ weights, y_test)
        y_test_list.append(testing_error)

    # print best lambda and min MSE for test set
    print(f'Lambda to minimize test set MSE for {file_set[0]}: {y_test_list.index(min(y_test_list))}')
    print(f'Min MSE: {round(min(y_test_list), 4)}')
    print()

    # plot MSE vs Lambda for training & test
    plt.plot(lambda_list, y_train_list, color='r', label='Training Data')
    plt.plot(lambda_list, y_test_list, color='b', label='Testing Data')
    plt.title(f'{file_set[0]} vs {file_set[1]} in Lambda Range 0-150')  # title based on file names
    plt.xlabel('Lambda', size=12)  # X-axis label
    plt.ylabel('MSE', size=12)  # Y-axis label
    plt.legend()  # show legend for lines
    plt.show()  # display the plot

    #creates additional 1-150 lambda graph for certain sets
    if file_set[0] in ['train-50(1000)-100.csv', 'train-100(1000)-100.csv', 'train-150(1000)-100.csv']:
        # plot MSE vs Lambda for training & test
        plt.plot(lambda_list[1:], y_train_list[1:], color='r', label='Training Data') #omits 0 lambda data
        plt.plot(lambda_list[1:], y_test_list[1:], color='b', label='Testing Data') #omits 0 lambda data
        plt.title(f'{file_set[0]} vs {file_set[1]} in Lambda Range 1-150')  # title based on file names
        plt.xlabel('Lambda', size=12)  # X-axis label
        plt.ylabel('MSE', size=12)  # Y-axis label
        plt.legend()  # show legend for lines
        plt.show()  # display the plot


