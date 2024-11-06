# Math libraries
import numpy as np
import pandas as pd

## IMPUTES MISSING VALUES FOR TRAINING AND TESTING DATAFRAMES
## INPUTS:
### train_df: training dataframe
### test_df: testing dataframe
def impute_train_test(train_df, test_df):
    # Training and testing datasets have same number of columns
    for col in train_df.columns:
        if train_df[col].dtypes == 'O':
            # Find most frequent value of current column of train_df
            f_hat = train_df[col].value_counts().index[0]
            if train_df[col].isna().sum() != 0:
                # Find row indices which contain NaN values of current column of train_df
                idx = train_df[train_df[col].isna()].index
                # Replace those rows with f_hat
                train_df.loc[idx, col] = f_hat
            if test_df[col].isna().sum() != 0:
                # Find row indices which contain NaN values of current column of test_df
                idx = test_df[test_df[col].isna()].index
                # Replace those rows with f_hat
                test_df.loc[idx, col] = f_hat

        elif train_df[col].dtypes == 'float64':
            # Find mean of current column of train_df
            train_mean = train_df[col].mean()
            if train_df[col].isna().sum() != 0:
                # Find row indices which contain NaN values of current column of train_df
                idx = train_df[train_df[col].isna()].index
                # Replace those rows with train_mean
                train_df.loc[idx, col] = train_mean
            if test_df[col].isna().sum() != 0:
                # Find row indices which contain NaN values of current column of test_df
                idx = test_df[test_df[col].isna()].index
                # Replace those rows with train_mean
                test_df.loc[idx, col] = train_mean

## EXTRACTS HEADER NAMES OF CATEGORICAL AND NUMERICAL COLUMNS IN DATAFRAME
## INPUTS:
### df: dataframe
## OUTPUTS:
### obj_col: array of header names for the categorical columns
### num_col: array of header names for the numerical columns
def get_categorical_numerical_headers(df):
    # Identify categorical columns (these will be type 'O')
    obj_bool = (df.dtypes == 'O')
    # Identify numerical columns (these will be 'float64' or 'int64')
    num_bool = (df.dtypes != 'O')
    # Find index values of categorical columns
    obj_idx = np.where(obj_bool)[0]
    # Find index values of numerical columns
    num_idx = np.where(num_bool)[0]
    # Initialise array of strings to store categorical and numerical header names
    obj_col = [''] * obj_idx.shape[0]
    num_col = [''] * num_idx.shape[0]
    for i in np.arange(obj_idx.shape[0]):
        # Generate headers with string arithmetic and type conversion
        obj_col[i] = 'A' + str(obj_idx[i] + 1)
    for i in np.arange(num_idx.shape[0]):
        # Generate headers with string arithmetic and type conversion
        num_col[i] = 'A' + str(num_idx[i] + 1)

    return obj_col, num_col
