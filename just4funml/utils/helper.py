import numpy as np


def train_cv_test_split(dataset):
    """
    return training, cross validation, and test set which are obtained by randomly by splitting
    the passed in dataset into 60% training, 20% cross validation, 20% test set
    """
    # Randomize the dataset
    dataset = dataset.sample(frac=1)

    # Split the dataset into train, cv, test set (60%, 20%, 20%)
    dataset_train, dataset_cv, dataset_test = np.split(dataset, [int(0.6 * len(dataset)), int(0.8 * len(dataset))])

    # Reset the index for all 3 dataset and drop the previous index
    dataset_train = dataset_train.reset_index(drop=True)
    dataset_cv = dataset_cv.reset_index(drop=True)
    dataset_test = dataset_test.reset_index(drop=True)

    return dataset_train, dataset_cv, dataset_test
