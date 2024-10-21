# Predicting Credit Card Approvals using Machine Learning
Credit card approvals comprise the many types of applications received by modern commerical banking institutions. A variety of metrics are used to determine an individual's successful credit card approval, inluding their age, income and credit score. With the growing number of these applications, their manual analysis is often time-consuming and can be subject to error. Machine learning methods provide an effective solution for automating this process, which ultimately involves a classification task, i.e. the application is either accepted or denied. For this project, we will build a predictor which automates the credit card approval process via the machine learning methods of Logistic Regression, K-nearest neighbours (KNN) and Random Forest Models.

## 0. Importing Packages
We begin by importing the necessary Python libraries.
```python
# Math and Plotting libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

## 1. Importing the dataset
This project utilises the [credit card approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) imported from the UCI Machine Learning Repository which has been anonymised for confidentiality. We import the dataset as follows:
```python
# UC Irvine Machine Learning Repository
from ucimlrepo import fetch_ucirepo
# fetch dataset
credit_approval = fetch_ucirepo(id=27)
```
Inspecting the type of ```credit_approval```, we see that it consists of 3 keys: 'features', 'targets', and 'original'.
```python
credit_approval.data.features.head()
```
```python
   A15    A14 A13 A12  A11 A10 A9    A8 A7 A6 A5 A4     A3     A2 A1
0    0  202.0   g   f    1   t  t  1.25  v  w  g  u  0.000  30.83  b
1  560   43.0   g   f    6   t  t  3.04  h  q  g  u  4.460  58.67  a
2  824  280.0   g   f    0   f  t  1.50  h  q  g  u  0.500  24.50  a
3    3  100.0   g   t    5   t  t  3.75  v  w  g  u  1.540  27.83  b
4    0  120.0   s   f    0   f  t  1.71  v  w  g  u  5.625  20.17  b
```
```python
credit_approval.data.targets.head()
```
```python
  A16
0   +
1   +
2   +
3   +
4   +
```
```python
credit_approval.data.original.head()
```
```python
  A1     A2     A3 A4 A5 A6 A7    A8 A9 A10  A11 A12 A13    A14  A15 A16
0  b  30.83  0.000  u  g  w  v  1.25  t   t    1   f   g  202.0    0   +
1  a  58.67  4.460  u  g  q  h  3.04  t   t    6   f   g   43.0  560   +
2  a  24.50  0.500  u  g  q  h  1.50  t   f    0   f   g  280.0  824   +
3  b  27.83  1.540  u  g  w  v  3.75  t   t    5   t   g  100.0    3   +
4  b  20.17  5.625  u  g  w  v  1.71  t   f    0   f   s  120.0    0   +
```
The dataset contains 690 rows with some missing values, with the 'original' key being composed of the columns from the 'features' and 'targets' keys. Upon further inspection of the ```credit_approval.data.original``` dataframe, we identify its attributes to be:

- A1: a, b
- A2: continuous
- A3: continuous
- A4: u, y, l, t
- A5: g, p, gg
- A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff
- A7: v, h, bb, j, n, z, dd, ff, o
- A8: continuous
- A9: t, f
- A10: t, f
- A11: continuous
- A12: t, f
- A13: g, p, s
- A14: continuous
- A15: continuous
- A16: +, -

These anonymised attributes reveal very little about the nature of the features. However, [this resource](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) provides insight to those characteristics typically employed by banking institutions when considering credit card applications. Following from this, we can determine with good confidence that the features of the ```credit_approval.data.original``` dataframe must map to the following:
  
- A1: Gender
- A2: Age
- A3: Debt
- A4: Marital status
- A5: Bank customer type
- A6: Education level
- A7: Ethnicity
- A8: Years of Employment
- A9: Prior default
- A10: Employment status
- A11: Credit score
- A12: Drivers license type
- A13: Citizenship status
- A14: Zipcode
- A15: Income
- A16: Approval status

This mapping will help during the later stages of this project when more informed decisions will be made with the data.

## 2. Data Exploration
Beginning our data exploration, we check the fields for those variables comprising the `credit_approval` dataframe.
```python
credit_approval.variables
```
```python
   name     role         type demographic description units missing_values
0   A16   Target  Categorical        None        None  None             no
1   A15  Feature   Continuous        None        None  None             no
2   A14  Feature   Continuous        None        None  None            yes
3   A13  Feature  Categorical        None        None  None             no
4   A12  Feature  Categorical        None        None  None             no
5   A11  Feature   Continuous        None        None  None             no
6   A10  Feature  Categorical        None        None  None             no
7    A9  Feature  Categorical        None        None  None             no
8    A8  Feature   Continuous        None        None  None             no
9    A7  Feature  Categorical        None        None  None            yes
10   A6  Feature  Categorical        None        None  None            yes
11   A5  Feature  Categorical        None        None  None            yes
12   A4  Feature  Categorical        None        None  None            yes
13   A3  Feature   Continuous        None        None  None             no
14   A2  Feature   Continuous        None        None  None            yes
15   A1  Feature  Categorical        None        None  None            yes
```
Checking the information and summary statistics (and making the assignment ```credit_df = credit_approval.data.original```):
```python
# Assign '.original' dataframe to credit_df
credit_df = credit_approval.data.original
# Check information
credit_df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 690 entries, 0 to 689
Data columns (total 16 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   A1      678 non-null    object 
 1   A2      678 non-null    float64
 2   A3      690 non-null    float64
 3   A4      684 non-null    object 
 4   A5      684 non-null    object 
 5   A6      681 non-null    object 
 6   A7      681 non-null    object 
 7   A8      690 non-null    float64
 8   A9      690 non-null    object 
 9   A10     690 non-null    object 
 10  A11     690 non-null    int64  
 11  A12     690 non-null    object 
 12  A13     690 non-null    object 
 13  A14     677 non-null    float64
 14  A15     690 non-null    int64  
 15  A16     690 non-null    object 
dtypes: float64(4), int64(2), object(10)
memory usage: 86.4+ KB
```
```python
# Check summary statistics
credit_df.describe()
```
```python
               A2          A3          A8        A11          A14            A15
count  678.000000  690.000000  690.000000  690.00000   677.000000     690.000000  
mean    31.568171    4.758725    2.223406    2.40000   184.014771    1017.385507  
std     11.957862    4.978163    3.346513    4.86294   173.806768    5210.102598  
min     13.750000    0.000000    0.000000    0.00000     0.000000       0.000000  
25%     22.602500    1.000000    0.165000    0.00000    75.000000       0.000000  
50%     28.460000    2.750000    1.000000    0.00000   160.000000       5.000000  
75%     38.230000    7.207500    2.625000    3.00000   276.000000     395.500000  
max     80.250000   28.000000   28.500000   67.00000  2000.000000  100000.000000
```
The above summary statistics have been automatically limited to those columns pertaining to numerical features. Furthermore, the ```count``` field reveals that columns A2 and A14 do not include all 690 rows of the dataset. As indicated by the ```missing_values``` column in the table returned by ```credit_approval.variables``` , this means that some entries from features A2 and A14 are missing (along with columns A1, A4, A5, A6 and A7) and will require imputation. We also see how the values in column A15 are several orders of magnitude greater than those of the other numerical columns, suggesting we apply feature scaling techniques. This will be addressed in further detail later on in the project.

Before preprocessing the data to impute any missing values and perform feature scaling, we conclude our initial exploration by examining the histograms of these numerical features. While this is normally performed after preprocessing to obtain a more accurate interpretation, an early examination helps build familiarity with the data and can provide general insights to their distributions. Moreover, this allows us to make early inferences about the existence of any outliers. We first extract the 'targets' and 'features' variables from ```credit_df```:
```python
# Extract targets
y = credit_df['A16']
# Extract features
Xfeatures = credit_df.drop(['A16'], axis = 1)
```
Now, we generate the corresponding histograms for those numerical features comprising the ```Xfeatures``` dataframe. To generalise our implementation for scalability, we create a module called ```CCASubs``` and within it, develop a function to automatically extract the column header names for those categorical and numerical features.
```python
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
```
Importing the ```CCASubs``` module containing the ```get_categorical_numerical_headers()``` function, we extract the header names for the categorical and numerical features from ```Xfeatures```.
```python
# Import module containing the impute_train_test function
import CCASubs

# Get header names for the categorical and numerical columns
cat_cols, num_cols = CCASubs.get_categorical_numerical_headers(Xfeatures)
```
Generate the histograms for those numerical features using ```num_cols```.
```python
## GENERATE HISTOGRAM OF NUMERICAL FEATURES BEFORE SPLITTING AND IMPUTING
# Configure backend for interactive mode
matplotlib.use('TkAgg')
# Initialise (2 x 3) axes for subplot of histograms (there are 6 numerical features)
sp_row = 2
sp_col = 3
# Reshape num_cols for convenient indexing later on
num_col_reshape = np.reshape(num_cols, (sp_row, sp_col))
# Set number of bins to be the square root of the total number of samples
num_samples = Xfeatures.shape[0]
num_bins = int(np.ceil(np.sqrt(num_samples)))
# Instantiate subplot from matplotlib.pyplot
fig, axs = plt.subplots(sp_row, sp_col)
# Iterate over rows of subplot
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot histogram for corresponding column index of Xfeature dataframe
        axs[i,j].hist(Xfeatures[num_col_reshape[i,j]], bins = num_bins, edgecolor = 'black', linewidth = 1.2)
        axs[i,j].set_xlabel(num_col_reshape[i,j], fontsize = 10)
        axs[i,j].set_ylabel('Frequency')
```

![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/hist_numerical_features_before_imputing.png)

The distribution of the data for these numerical features are heavy-tailed and skewed to the right, meaning that their medians are less than their means, and suggests the presence of outliers. At this final stage of our exploration, these simple visualisations give an early overview of the statistics underlying the numerical features. While any further analysis requires we first preprocess the data, this initial exploration has successfully developed our familiarity with the overall dataset and how it should be processed prior to implementing the ML models.

## 3. Preprocessing the data
Following our initial exploration, we will now preprocess the data before building our ML models. The preprocessing sequence can be broken down into the following tasks:

- Splitting the dataset into training and testing sets.
- Imputing the missing data.
- Converting non-numerical data.
- Scaling the feature values.

Before continuing, we make two remarks: first, it is important to impute any missing information in both the training and testing datasets. Ignoring any missed values can negatively affect the performance of the predictive model, with some ML algorithms requiring a complete dataset for their successful operation (such as Logistic Regression). Second, we note how it is ideal to first split the data into the training and testing datasets prior to imputing. This is due to what the training and testing datasets attempt to replicate in practice: the training dataset comprises historical information that is used to first build the predictive model, while the testing dataset serves as future unknown information which the model ingests to predict an outcome. In essence, the training dataset represents the past, with the testing dataset representing the future. Since the ML model is constructed solely from the training dataset, any preprocessing of the data, such as imputing missing values, should be performed on the training dataset alone. In other words, any imputation method applied to the training dataset should consider only those statistics of the training dataset. Then, if the testing dataset also requires imputing, its imputation method should follow that which was applied to the training dataset. This procedure ensures that no information from the testing data is used to preprocess the training data, which would bias the construction of the ML model and its ensuing results. This concept is referred to as 'data leakage', and will be handled accordingly in this project prior to constructing our ML models later on. Our justification for opting to impute the missing values instead of omitting those samples entirely will also be covered.

### 3.1 Splitting the dataset into training and testing sets
In Section 1, our characterisation of the anonymised features gave further insight to the nature of the dataset. Now with this contextual knowledge, we identify how _feature selection_ could be exercised before splitting our data into its training and testing sets. For instance, columns A12 and A14 could be dropped prior to splitting, as drivers license type and zipcode could be declared as relatively minor factors compared to the other characteristics when deciding the approval of a credit card application. This small-scale example of feature selection supports how good feature selection practices can facilitate ML modelling by reducing the number of overall features and the introduction of noise to the dataset; improving the performance, efficiency, and interpretability of the ML model. In practice however, a more robust method for determining which features are relevant for inclusion would involve measuring the statistical correlation between the available features along with the targets. Since this is beyond the scope of the project, we ignore this process and leave all features within the dataset intact. We then split the features and targets variables into their training and testing sets:
```python
# Split Xfeatures into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(Xfeatures, y, test_size = 0.33, random_state = 42)
```

### 3.2 Imputing the missing data
Before discussing methods of imputing missing values in the training and testing datasets, we elaborate on our aforementioned choice of imputing the missing values over omitting those samples as a whole. As the total number of samples (690) comprising the original dataset is already small, then omitting any entirely could detract from the performance of the ensuing ML model. On the other hand, as only approximately 0.61% of the available values are missing, one could argue on reasonable grounds for their omission. However, conventional practice is to provide the predictive model with as much training data as possible so as to maximise support during the learning stage. Additionally, any remaining features from those samples containing missing values would still contribute towards fine-tuning the underlying ML algorithm and improving its robustness. 

With these justifications in mind, we are now ready to perform data imputation on ```Xtrain``` and ```Xtest```. As ```nan``` values are used to indicate any missing data, we first inspect the distribution of ```nan``` values  across all columns of ```Xtrain``` and ```Xtest```.
```python
Xtrain.isna().sum()
Xtest.isna().sum()
```
```python
A1      8
A2      5
A3      0
A4      6
A5      6
A6      7
A7      7
A8      0
A9      0
A10     0
A11     0
A12     0
A13     0
A14    12
A15     0
dtype: int64
A1     4
A2     7
A3     0
A4     0
A5     0
A6     2
A7     2
A8     0
A9     0
A10    0
A11    0
A12    0
A13    0
A14    1
A15    0
dtype: int64
```
As expected, the missing entries in ```Xtrain``` and ```Xtest``` occur in columns A1, A2, A4, A5, A6, A7 and A14, which assume the following datatypes (as obtained in Section 2):

- A1: Categorical
- A2: Continuous
- A4: Categorical
- A5: Categorical
- A6: Categorical
- A7: Categorical
- A14: Continuous

Since the missing data is either categorical or continuous in nature, we must implement separate imputation methods for them. Adopting conventional practice, we will apply the methods of mean imputation to the continuous data (type ```float64```) and mode imputation to the categorical data (type ```O```). To facilitate this, we build the following function which takes a training and testing dataframe (in that order) and imputes them according to their data types.
```python
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

```
Input ```Xtrain``` and ```Xtest``` to this function (contained in ```CCASubs.py```) and verify that they contain no missing values.
```python
# Impute Xtrain and Xtest using the impute_train_test function
CCASubs.impute_train_test(Xtrain, Xtest)
# Count number of nan entries in Xtrain
Xtrain.isna().sum()
# Count number of nan entries in Xtest
Xtest.isna().sum()
```
```python
A1     0
A2     0
A3     0
A4     0
A5     0
A6     0
A7     0
A8     0
A9     0
A10    0
A11    0
A12    0
A13    0
A14    0
A15    0
dtype: int64
A1     0
A2     0
A3     0
A4     0
A5     0
A6     0
A7     0
A8     0
A9     0
A10    0
A11    0
A12    0
A13    0
A14    0
A15    0
dtype: int64
```

### 3.3 Converting non-numerical data
We now proceed to convert the categorical feature variables into numerical data types. This task is necessary as many ML models require the data to be strictly numerical in type, with this conversion also resulting in faster computation. To achieve this, we utilise the ```OrdinalEncoder()``` routine from the ```sklearn.preprocessing``` library to encode the categorical features into integer arrays. We note how similarly to before, this task is executed after the data has been split into its training and testing sets so as to circumvent data leakage. If the data were to be encoded before splitting, the model would be effectively informed a priori of the labels composing the future data and would lead to it overfitting during the testing phase. Additionally, the encoding model should first be fitted using only the data from the training set before being applied to both the training and testing sets to effect a proper transformation.

For scalability, we write a function to automate the process of extracting the header names for the categorical features.
```python
## EXTRACTS HEADER NAMES OF CATEGORICAL COLUMNS IN DATAFRAME
## INPUTS:
### df: dataframe
## OUTPUTS:
### obj_col: array of names of categorical columns
def get_categorical_col_names(df):
    # Identify columns of data type object ('O')
    obj_bool = (df.dtypes == 'O')
    # Find index values of categorical columns
    obj_idx = np.where(obj_bool)[0]
    # Initialise empty array of strings to store header names
    obj_col = [''] * obj_idx.shape[0]
    for i in np.arange(obj_idx.shape[0]):
        # Generate headers with string arithmetic and type conversion
        obj_col[i] = 'A' + str(obj_idx[i]+1)

    return obj_col
```
Check that the function correctly returns all categorical features and list them.
```python
# Get header names for the categorical columns
cat_cols = CCASubs.get_categorical_col_names(Xfeatures)
# Check categorical columns for Xtrain and Xtest
print('Xtrain (categorical columns):\n')
print(Xtrain[cat_cols])
print('\n')
print('Xtest (categorical columns):\n')
print(Xtest[cat_cols])
```
```python
Xtrain (categorical columns):
    A1 A4 A5  A6  A7 A9 A10 A12 A13
382  a  y  p   i  bb  f   f   f   g
137  b  u  g   m   v  t   t   f   g
346  b  u  g   c   v  f   f   t   g
326  b  y  p   c   v  f   f   f   g
33   a  u  g   e   v  t   f   t   g
..  .. .. ..  ..  .. ..  ..  ..  ..
71   b  u  g   d  bb  t   f   t   g
106  b  u  g   k   v  t   f   f   s
270  b  u  g   c   v  f   f   f   p
435  b  y  p  ff  ff  f   t   f   g
102  b  u  g   q   v  t   t   f   g
[462 rows x 9 columns]

Xtest (categorical columns):
    A1 A4 A5  A6  A7 A9 A10 A12 A13
286  a  u  g  ff  ff  f   t   t   g
511  a  u  g   j   j  t   f   f   g
257  b  u  g   d   v  f   f   f   g
336  b  u  g   c   v  f   f   t   g
318  b  y  p   m  bb  f   f   t   s
..  .. .. ..  ..  .. ..  ..  ..  ..
375  a  y  p   e  dd  f   f   f   g
234  a  u  g   i  bb  t   t   f   g
644  b  y  p   w   v  f   f   t   g
271  b  u  g   c   v  f   f   t   g
311  b  y  p   c   v  f   f   t   g
[228 rows x 9 columns]
```
Verify the results of the ordinal encoder.
```python
# Verify results of the ordinal encoder
print('Xtrain (encoded):\n')
print(Xtrain)
print('\n')
print('Xtest (encoded):\n')
print(Xtest)
```
```python
Xtrain (encoded):
      A1     A2     A3   A4   A5    A6  ...  A10  A11  A12  A13         A14   A15
382  0.0  24.33  2.500  2.0  2.0   6.0  ...  0.0    0  0.0  0.0  200.000000   456
137  1.0  33.58  2.750  1.0  0.0   9.0  ...  1.0    6  0.0  0.0  204.000000     0
346  1.0  32.25  1.500  1.0  0.0   1.0  ...  0.0    0  1.0  0.0  372.000000   122
326  1.0  30.17  1.085  2.0  2.0   1.0  ...  0.0    0  0.0  0.0  170.000000   179
33   0.0  36.75  5.125  1.0  0.0   4.0  ...  0.0    0  1.0  0.0    0.000000  4000
..   ...    ...    ...  ...  ...   ...  ...  ...  ...  ...  ...         ...   ...
71   1.0  34.83  4.000  1.0  0.0   3.0  ...  0.0    0  1.0  0.0  177.275556     0
106  1.0  28.75  1.165  1.0  0.0   8.0  ...  0.0    0  0.0  2.0  280.000000     0
270  1.0  37.58  0.000  1.0  0.0   1.0  ...  0.0    0  0.0  1.0  177.275556     0
435  1.0  19.00  0.000  2.0  2.0   5.0  ...  1.0    4  0.0  0.0   45.000000     1
102  1.0  18.67  5.000  1.0  0.0  10.0  ...  1.0    2  0.0  0.0    0.000000    38
[462 rows x 15 columns]

Xtest (encoded):
      A1         A2     A3   A4   A5    A6  ...  A10  A11  A12  A13    A14   A15
286  0.0  31.635755   1.50  1.0  0.0   5.0  ...  1.0    2  1.0  0.0  200.0   105
511  0.0  46.000000   4.00  1.0  0.0   7.0  ...  0.0    0  0.0  0.0  100.0   960
257  1.0  20.000000   0.00  1.0  0.0   3.0  ...  0.0    0  0.0  0.0  144.0     0
336  1.0  47.330000   6.50  1.0  0.0   1.0  ...  0.0    0  1.0  0.0    0.0   228
318  1.0  19.170000   0.00  2.0  2.0   9.0  ...  0.0    0  1.0  2.0  500.0     1
..   ...        ...    ...  ...  ...   ...  ...  ...  ...  ...  ...    ...   ...
375  0.0  20.830000   0.50  2.0  2.0   4.0  ...  0.0    0  0.0  0.0  260.0     0
234  0.0  58.420000  21.00  1.0  0.0   6.0  ...  1.0   13  0.0  0.0    0.0  6700
644  1.0  36.170000   0.42  2.0  2.0  12.0  ...  0.0    0  1.0  0.0  309.0     2
271  1.0  32.330000   2.50  1.0  0.0   1.0  ...  0.0    0  1.0  0.0  280.0     0
311  1.0  19.000000   1.75  2.0  2.0   1.0  ...  0.0    0  1.0  0.0  112.0     6
[228 rows x 15 columns]
```

### 3.4 Scaling the feature values
- In Section 2, we saw how the values of column A15 are several orders of magnitude greater than the other numerical values. While this suggests implementing feature scaling techniques, we first acknowledge how its suitability depends on the ML methods applied to the dataset.
- For this project, we will be using the ML methods of Logistic Regression, KNN and Random Forest. All three methods fall under the umbrella of classification models, for which distance metrics (such as Euclidean, Manhattan, Minkowski or Hamming) can be used to improve performance.


GENERATE PLOTS
Although a number of techniques could be implemented to more rigorously determine which distribution each feature conforms to, we may invoke the Central Limit Theorem (CLT) to sensibly assume that all features are normally distributed. This suggests that during preprocessing, we could apply the standard approach of identifying outliers to be those data points situated further than 3 standard deviations from the mean.

THEN GIVE DISCUSSION ABOUT OUTLIERS:

We quickly discuss the two main approaches towards the management of outliers and their underlying caveats. The first approach involves removing any outliers before splitting the data into its training and testing sets. This ensures consistency throughout the entire dataset as their removal would adjust the means and variances of the numerical features, thereby affecting any imputation methods applied later on. While their exclusion would positively influence the robustness of the ML model, we can no longer assess its performance with anomalous values that would simulate fringe cases in practice. Meanwhile, the second approach involves removing any outliers after splitting the data. In this case, outliers are removed only from the training set in order to reduce any skewed analyses or inaccuracies in the model, while those within the testing set are preserved to give better insight to its performance. Consequently, the means and variances corresponding to the training set may not accurately reflect what could otherwise be considered their 'true' values, and would influence the statistics used to train the model. Thus, the decision to remove outliers from any dataset, whether before or after splitting, is typically context dependent and generally left to the analyst's discretion. In our case, we will inspect the numerical features for any outliers before moving to the data preprocessing stage. Furthermore, given how the features of this dataset most likely reflect those characteristics outlined in Section 3.1 (gender, age, debt, marital status, etc), we can intuit that any 'outliers' would be "true" outliers; not being representative of any measurement or processing errors, data entry or poor sampling. Thus, we deem it appropriate to preserve any outliers and for their inclusion during the training and testing stages of the ML models.
