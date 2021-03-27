import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def main():
    from just4funml.utils.helper import train_cv_test_split
    from just4funml.learning_algorithms.supervised import linear_regression

    # dataframe = pd.read_csv("../data/weatherAUS.csv")
    # dataframe = dataframe[['MinTemp', 'MaxTemp']]
    # dataframe = dataframe.dropna()
    # dataframe.to_csv("../data/cleanedData.csv", index=False)
    dataframe = pd.read_csv("../data/cleanedData.csv")

    # Get some insight from the data
    print('=================================')
    print('First 5 examples in the dataset')
    print('=================================')
    print(dataframe.head(), '\n')

    print('=============================================')
    print('Get some statistical description of the data')
    print('=============================================')
    print(dataframe.describe(), '\n')

    print('=============================================')
    print('List the available column of the data')
    print('=============================================')
    print(dataframe.columns, '\n')

    print('=============================================')
    print('Print the overall shape of the data')
    print('=============================================')
    print('Shape of the data:', dataframe.shape, '\n')

    # # Plot the data
    # # print('=============================================')
    # # print('Visualize the available data')
    # # print('=============================================')
    # # plt.plot(dataframe['MinTemp'], dataframe['MaxTemp'], 'bx', markersize=1)
    # # plt.xlabel('Minimum Temperature')
    # # plt.ylabel('Maximum Temperature')
    # # plt.title('Maximum vs Minimum Temperature')
    # # plt.axis([-10, 50, -10, 50])
    # # plt.show()

    dataframe_train, dataframe_cv, dataframe_test = train_cv_test_split(dataframe)
    # print(dataframe_train.shape)
    # print(dataframe_cv.shape)
    # print(dataframe_test.shape)

    features = dataframe_train.iloc[:, 0]
    labels = dataframe_train.iloc[:, 1]
    model = linear_regression.LinearRegression(features, labels)
    cost = model.cost_function()
    print(cost)


if __name__ == '__main__':
    # Add the path of '../Machine Learning Algorithm' to sys.path to ensure module import
    # from other packages successful
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    main()

