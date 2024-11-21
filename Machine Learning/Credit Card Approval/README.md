# Predicting Credit Card Application Approvals using Machine Learning
Credit card approvals comprise the many types of applications received by modern commerical banking institutions. A variety of metrics are used to determine an individual's successful credit card approval, inluding their age, income and credit score. With the growing number of these applications, their manual analysis is often time-consuming and can be subject to error. Machine learning methods provide an effective solution for automating this process, which ultimately involves a classification task, i.e. the application is either accepted or denied. For this project, we build a predictor which automates the credit card approval process via machine learning methods using the following three classification algorithms: Logistic Regression (LR), k-Nearest Neighbours (KNN) and Random Forest (RF). We organise this document according to those steps comprising the general data science pipeline:
- Section 1: [Import Packages and Dataset](#1-import-packages-and-dataset)
- Section 2: [Data Exploration](#2-data-exploration)
- Section 3: [Preprocessing the data](#3-preprocessing-the-data)
- Section 4: [Classification using Machine Learning](#4-classification-using-machine-learning)
- Section 5: [Results and Evaluation](#5-results-and-evaluation)
- Section 6: [Conclusion and Future Work](#6-conclusion-and-future-work)

## 1. Import packages and dataset 
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

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

# Personal libraries
import MySubs
```

This project utilises the [credit card approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) imported from the UCI Machine Learning Repository which has been anonymised for confidentiality. We import the dataset as follows:
```python
# Import UC Irvine Machine Learning Repository
from ucimlrepo import fetch_ucirepo
# Fetch dataset
credit_approval = fetch_ucirepo(id=27)
```
Inspecting the type of ```credit_approval```, we see that it consists of 3 keys: 'features', 'targets', and 'original'.
```python
print('credit_approval.data.features:\n')
print(credit_approval.data.features)
print('\n')
print('credit_approval.data.targets:\n')
print(credit_approval.data.targets)
print('\n')
print('credit_approval.data.original:\n')
print(credit_approval.data.original)
```
```python
credit_approval.data.features:
     A15    A14 A13 A12  A11 A10 A9    A8  A7  A6 A5 A4      A3     A2 A1
0      0  202.0   g   f    1   t  t  1.25   v   w  g  u   0.000  30.83  b
1    560   43.0   g   f    6   t  t  3.04   h   q  g  u   4.460  58.67  a
2    824  280.0   g   f    0   f  t  1.50   h   q  g  u   0.500  24.50  a
3      3  100.0   g   t    5   t  t  3.75   v   w  g  u   1.540  27.83  b
4      0  120.0   s   f    0   f  t  1.71   v   w  g  u   5.625  20.17  b
..   ...    ...  ..  ..  ...  .. ..   ...  ..  .. .. ..     ...    ... ..
685    0  260.0   g   f    0   f  f  1.25   h   e  p  y  10.085  21.08  b
686  394  200.0   g   t    2   t  f  2.00   v   c  g  u   0.750  22.67  a
687    1  200.0   g   t    1   t  f  2.00  ff  ff  p  y  13.500  25.25  a
688  750  280.0   g   f    0   f  f  0.04   v  aa  g  u   0.205  17.92  b
689    0    0.0   g   t    0   f  f  8.29   h   c  g  u   3.375  35.00  b
[690 rows x 15 columns]

credit_approval.data.targets:
    A16
0     +
1     +
2     +
3     +
4     +
..   ..
685   -
686   -
687   -
688   -
689   -
[690 rows x 1 columns]

credit_approval.data.original:
    A1     A2      A3 A4 A5  A6  A7    A8 A9 A10  A11 A12 A13    A14  A15 A16
0    b  30.83   0.000  u  g   w   v  1.25  t   t    1   f   g  202.0    0   +
1    a  58.67   4.460  u  g   q   h  3.04  t   t    6   f   g   43.0  560   +
2    a  24.50   0.500  u  g   q   h  1.50  t   f    0   f   g  280.0  824   +
3    b  27.83   1.540  u  g   w   v  3.75  t   t    5   t   g  100.0    3   +
4    b  20.17   5.625  u  g   w   v  1.71  t   f    0   f   s  120.0    0   +
..  ..    ...     ... .. ..  ..  ..   ... ..  ..  ...  ..  ..    ...  ...  ..
685  b  21.08  10.085  y  p   e   h  1.25  f   f    0   f   g  260.0    0   -
686  a  22.67   0.750  u  g   c   v  2.00  f   t    2   t   g  200.0  394   -
687  a  25.25  13.500  y  p  ff  ff  2.00  f   t    1   t   g  200.0    1   -
688  b  17.92   0.205  u  g  aa   v  0.04  f   f    0   f   g  280.0  750   -
689  b  35.00   3.375  u  g   c   h  8.29  f   f    0   t   g    0.0    0   -
[690 rows x 16 columns]
```
The dataset contains 690 rows with some missing values, with the 'original' key being composed of the columns from the 'features' and 'targets' keys. Upon further inspection of the ```credit_approval.data.original``` dataframe, we identify its attributes to be:

| Feature | Attributes |
|---------| -----------|
| A1 | a, b |
| A2 | continuous |
| A3 | continuous |
| A4 | u, y, l, t |
| A5 | g, p, gg |
| A6 | c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff |
| A7 | v, h, bb, j, n, z, dd, ff, o |
| A8 | continuous |
| A9 | t, f |
| A10 | t, f |
| A11 | continuous |
| A12 | t, f |
| A13 | g, p, s |
| A14 | continuous |
| A15 | continuous |
| A16 | +, - |

These anonymised attributes reveal very little about the nature of the features. However, [this resource](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) provides insight to those characteristics typically employed by banking institutions when considering credit card applications. Following from this, we can determine with good confidence that the features of the ```credit_approval.data.original``` dataframe must map to the following:

| Feature | Proposed description |
|---------| -----------|
| A1 | Gender |
| A2 | Age |
| A3 | Debt |
| A4 | Marital status |
| A5 | Bank customer type |
| A6 | Education level |
| A7 | Ethnicity |
| A8 | Years of Employment |
| A9 | Prior default |
| A10 | Employment status |
| A11 | Credit score |
| A12 | Drivers license type |
| A13 | Citizenship status |
| A14 | Zipcode |
| A15 | Income |
| A16 | Approval status |

This mapping will help during the later stages of this project when more informed decisions will be made with the data.

## 2. Data Exploration
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

Beginning our data exploration, we check the fields for those variables comprising the `credit_approval` dataframe.
```python
# Check fields comprising variables in credit_approval
print('credit_approval.variables:\n')
print(credit_approval.variables)
```
```python
credit_approval.variables:
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
Assign ```credit_df = credit_approval.data.original``` and check the information and summary statistics.
```python
# Reassign credit_approval.data.original to credit_df
credit_df = credit_approval.data.original
# Check information
print('credit_df.info():\n')
print(credit_df.info())
```
```python
credit_df.info():
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
None
```
```python
# Check summary statistics
print('credit_df.describe():\n')
print(credit_df.describe())
```
```python
credit_df.describe():
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
X_features = credit_df.drop(['A16'], axis = 1)
```
Now, we generate the corresponding histograms for those numerical features comprising the ```X_features``` dataframe. To generalise our implementation for scalability, we create a module called ```MySubs``` and within it, develop a function to automatically extract the column header names for those categorical and numerical features.
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
Importing the ```MySubs``` module containing the ```get_categorical_numerical_headers()``` function, we extract the header names for the categorical and numerical features from ```X_features```.
```python
# Import module containing the impute_train_test function
import MySubs

# Get header names for the categorical and numerical columns
cat_cols, num_cols = MySubs.get_categorical_numerical_headers(X_features)
```
Generate the histograms for those numerical features using ```num_cols```.
```python
# Configure backend for interactive mode
matplotlib.use('TkAgg')
# Initialise (2 x 3) axes for subplots of histograms (there are 6 numerical features)
sp_row = 2
sp_col = 3
# Reshape num_cols for convenient indexing later on
num_col_reshape = np.reshape(num_cols, (sp_row, sp_col))
# Set number of bins to be the square root of the total number of samples
num_samples = X_features.shape[0]
num_bins = int(np.ceil(np.sqrt(num_samples)))
# Get header indices corresponding to numerical features
num_idx = np.where(X_features.dtypes != 'O')[0]
# Reshape num_idx for easier indexing when labelling
num_idx_reshape = np.reshape(num_idx, (sp_row, sp_col))
# Array of characteristics
feature_character = ['Gender', 'Age', 'Debt', 'Marital status', 'Bank customer type', 'Education level', 'Ethnicity', 'Years of Employment', 'Prior default', 'Employment status', 'Credit score', 'Drivers license type', 'Citizenship status', 'Zipcode', 'Income']
feature_character_num = np.array(feature_character)[num_idx]
# Reshape characteristics array for easier indexing when labelling
feature_character_num_reshape = np.reshape(feature_character_num, (sp_row, sp_col))
# Instantiate subplot from matplotlib.pyplot
fig, axs = plt.subplots(sp_row, sp_col)
# Iterate over rows of subplot
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot histogram for corresponding column index of Xfeature dataframe
        axs[i,j].hist(X_features[num_col_reshape[i,j]], bins = num_bins, edgecolor = 'black', linewidth = 1.2)
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_num_reshape[i,j], fontsize = 10)
        axs[i,j].set_ylabel('Frequency')
        plt.tight_layout
```

![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/hist_numerical_features_before_preprocessing.png)

The distribution of the data for these numerical features are heavy-tailed and skewed to the right, meaning that their medians are less than their means, and suggests the presence of outliers. As an additional check, we also visualise the target variable within the ```credit_df``` dataframe to compare the number of approved and declined applications.
```python
# Compare the number of approved and declined applications
fig, ax = plt.subplots(1, 1, figsize = (7, 5), sharex = True)
sns.countplot(data = credit_df, x = 'A16', edgecolor = "black", palette = "viridis", order = credit_df['A16'].value_counts().index)
total = credit_df['A16'].value_counts().sum()
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('Application status', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()
```

![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/approved_declined.png)

Out of 690 samples, we see 383 (55.5%) applications were denied ('-') while 307 (44.5%) applications were approved ('+'). This implies that the dataset has an approximately equal representation of both outcomes.

At this final stage of our exploration, these simple visualisations give an early overview of the statistics underlying the numerical features. While any further analysis requires we first preprocess the data, this initial exploration has successfully developed our familiarity with the overall dataset and how it should be processed prior to implementing the ML models.

## 3. Preprocessing the data
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

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
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size = 0.2, random_state = 42, stratify = y)
```
We quickly note that our above splitting of the data conforms to the popular "two-way holdout" method. Our decision for partitioning the data this way over the other popular "three-way holdout" method will be discussed further in Section 4. Also, note that we apply stratification when splitting the data. While a balanced dataset may render stratification unnecessary (especially for an already small and balanced dataset like ours), its application in our case statistically does no harm. Additionally, since our data is not a time series, stratification is sensible in this context.

### 3.2 Imputing the missing data
Before discussing methods of imputing missing values in the training and testing datasets, we elaborate on our aforementioned choice of imputing the missing values over omitting those samples as a whole. As the total number of samples (690) comprising the original dataset is already small, then omitting any entirely could detract from the performance of the ensuing ML model. On the other hand, as only approximately 0.61% of the available values are missing, one could argue on reasonable grounds for their omission. However, conventional practice is to provide the predictive model with as much training data as possible so as to maximise support during the learning stage. Additionally, any remaining features from those samples containing missing values would still contribute towards fine-tuning the underlying ML algorithm and improving its robustness. 

With these justifications in mind, we are now ready to perform data imputation on ```Xtrain``` and ```Xtest```. As ```nan``` values are used to indicate any missing data, we first inspect the distribution of ```nan``` values  across all columns of ```Xtrain``` and ```Xtest```.
```python
# Count number of nan entries in Xtrain before imputing
print('Number of nan entries in Xtrain before imputing:\n')
print(Xtrain.isna().sum())
# Count number of nan entries in Xtest before imputing
print('Number of nan entries in Xtest before imputing:\n')
print(Xtest.isna().sum())
```
```python
Number of nan entries in Xtrain before imputing:
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

Number of nan entries in Xtest before imputing:
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
Input ```Xtrain``` and ```Xtest``` to this function (contained in ```MySubs.py```) and verify that they contain no missing values.
```python
# Impute Xtrain and Xtest using the impute_train_test function
MySubs.impute_train_test(Xtrain, Xtest)

# Count number of nan entries in Xtrain after imputing
print('Number of nan entries in Xtrain after imputing:\n')
Xtrain.isna().sum()
# Count number of nan entries in Xtest after imputing
print('Number of nan entries in Xtest after imputing:\n')
Xtest.isna().sum()
```
```python
Number of nan entries in Xtrain after imputing:
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

Number of nan entries in Xtest after imputing:
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

Print the categorical features of ```Xtrain``` and ```Xtest``` before encoding.
```python
# Print categorical columns for Xtrain and Xtest
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
Instantiate the ordinal encoder and fit it to the categorical columns of ```Xtrain```, then apply the transformation to both ```Xtrain``` and ```Xtest```.
```python
# Instantiate OrdinalEncoder() function
ordinal_encoder = OrdinalEncoder()
# Fit and transform the encoder to the training data
Xtrain[cat_cols] = ordinal_encoder.fit_transform(Xtrain[cat_cols])
# Transform the testing data using the encoder previously fitted to the training data
Xtest[cat_cols] = ordinal_encoder.transform(Xtest[cat_cols])
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
Visualise the now encoded ```Xtrain``` data using histograms.
```python
# Initialise (5 x 3) axes for subplots of histograms (there are 15 numerical features after encoding)
sp_row = 5
sp_col = 3
# Reshape num_cols for convenient indexing later on
num_col_reshape = np.reshape(Xtrain.columns, (sp_row, sp_col))
# Set number of bins to be the square root of the total number of samples
num_samples = Xtrain.shape[0]
num_bins = int(np.ceil(np.sqrt(num_samples)))
# Instantiate subplot from matplotlib.pyplot
fig, axs = plt.subplots(sp_row, sp_col)
# Array of characteristics
feature_character = ['Gender', 'Age', 'Debt', 'Marital status', 'Bank customer type', 'Education level', 'Ethnicity', 'Years of Employment', 'Prior default', 'Employment status', 'Credit score', 'Drivers license type', 'Citizenship status', 'Zipcode', 'Income']
feature_character_reshape = np.reshape(feature_character, (sp_row, sp_col))
# Iterate over rows of subplot
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot histogram for corresponding column index of Xtrain dataframe
        axs[i,j].hist(Xtrain[num_col_reshape[i,j]], bins = num_bins, edgecolor = 'black', linewidth = 1.2)
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_reshape[i,j], fontsize = 10)
        axs[i,j].set_ylabel('Frequency')
```
![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/hist_encoded_numerical_features_after_preprocessing.png)

### 3.4 Scaling the feature values
In Section 2, we saw how the values of feature A15 (income) are several orders of magnitude greater than the other numerical features. For datasets like this, significant variations in feature values can bias the performance of the ensuing ML model. To circumvent this, feature scaling techniques are applied to the training and testing data, with the most common being the standard scaler and min-max scaler. The former technique standardises the data by scaling it to have a mean of 0 and a standard deviation of 1, and the latter technique normalises the data by scaling it between 0 and 1. While there are other techniques, each with their own caveats, the decision on applying feature scaling to the data ultimately depends on the ML algorithm being used.

For this project, we will be using the ML methods of Logistic Regression, KNN and Random Forest, which all fall under the umbrella of classification models. For Logistic Regression, the underlying algorithm employs the principles of gradient descent to effect the optimisation technique. For these types of ML models, feature scaling can assist the gradient descent to converge more quickly towards the minima. For KNN, distance-based algorithms are normally employed, whereby the distance between data points is used to determine their similarity and thus their classifications. Given this dependency on distance-based algorithms, ML models which utilise this method, such as KNN, are highly sensitive to feature scaling. On the other hand, the algorithms underlying decision tree-based ML models, such as Random Forest, are generally invariant to feature scaling since the decision tree only splits a node based on the available features. While there is little consequence of applying feature scaling to the data building decision tree-based ML models, hyperparameters specific to these models typically require further tuning to improve their predictive capabilities.

In light of this, we employ the min-max technique to scale our data. We instantiate the ```MinMaxScaler()``` function from the imported ```sklearn.preprocessing``` library, then first fit and transform the scaler to the training data before applying the same transformation to the testing data.
```python
# Instantiate MinMaxScaler() function
scaler = MinMaxScaler(feature_range = (0, 1))
# Fit and transform the scaler to the training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform the testing data using the scaler previously fitted to the training data
X_test_scaled = scaler.transform(X_test)
```
Now that the training and testing data have been successfully imputed, encoded and scaled, we proceed with regenerating the histograms of the 15 total features comprising the ```X_train_scaled```.
```python
# Convert X_train_scaled to a dataframe for easier plotting
X_train_scaled_df = pd.DataFrame(X_train_scaled)
# Initialise (5 x 3) axes for subplots of histograms (there are 15 numerical features after encoding)
sp_row = 5
sp_col = 3
# Reshape num_cols for convenient indexing later on
num_rescaled_col_reshape = np.reshape(X_train_scaled_df.columns, (sp_row, sp_col))
# Set number of bins to be the square root of the total number of samples
num_samples = X_train_scaled_df.shape[0]
num_bins = int(np.ceil(np.sqrt(num_samples)))
# Instantiate subplot from matplotlib.pyplot
fig, axs = plt.subplots(sp_row, sp_col)
# Iterate over rows of subplot
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot histogram for corresponding column index of X_train_scaled dataframe
        axs[i,j].hist(X_train_scaled_df[num_rescaled_col_reshape[i,j]], bins = num_bins, edgecolor = 'black', linewidth = 1.2)
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_reshape[i,j], fontsize = 10)
        axs[i,j].set_ylabel('Frequency')
```
![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/hist_scaled_numerical_features_after_preprocessing.png)

With the data now normalised, we confirm our previous mentioning of how the distributions for the numerical features A2, A3, A8, A11, A14 and A15 are heavy-tailed and skewed to the right. Currently, further statistical methods could be employed to more rigorously analyse the dataset, whose information would assist with interpreting those predictions returned by the ML models. For example, descriptive statistical methods could be used to provide insight into the central tendency, spread, and shape of these distributions, along with their means, medians and modes. We could also apply statistical tests, such as the Shapiro-Wilk test or the Kolmogorov-Smirnov test, to assess whether the data is truly normal. However, to remain within the scope of this project, we simplify the statistical analyses by invoking the Central Limit Theorem (CLT) to assume that all numerical features are normally distributed. Following this assumption, we could apply the standard approach of identifying outliers to be those data points situated further than 3 standard deviations from the mean. To visualise this, we generate the boxplots for features A2, A3, A8, A11, A14 and A15.
```python
# Initialise (2 x 3) axes for subplots of histograms (there are 6 numerical features)
sp_row = 2
sp_col = 3
# Array of numerical characteristics
num_idx = np.where(X_features.dtypes != 'O')[0]
num_idx_reshape = np.reshape(num_idx, (sp_row, sp_col))
feature_character_num = np.array(feature_character)[num_idx]
feature_character_num_reshape = np.reshape(feature_character_num, (sp_row, sp_col))
# Reshape num_cols for convenient indexing later on
num_col_reshape = np.reshape(num_cols, (sp_row, sp_col))
fig, axs = plt.subplots(sp_row, sp_col)
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot boxplot for corresponding column index of Xfeature dataframe
        axs[i,j].boxplot(X_train_scaled_df[num_idx_reshape[i,j]], vert = 0, sym = 'rs', positions=[0], widths=[0.3])
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_num_reshape[i,j], fontsize = 10)
        axs[i,j].get_yaxis().set_ticks([])
        plt.tight_layout()
```
![image](https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Credit%20Card%20Approval/images/box_scaled_numerical_features.png)

As expected, all numerical features contain outliers, with a large number comprising features A2 (age), A3 (debt), A8 (years of employment), A11 (credit score) and A15 (income). While this preliminary examination suggests we remove these outliers from the training data before building our ML models, this may not always work to our benefit. To clarify, we quickly discuss the two main approaches overarching the management of outliers and their underlying caveats. The first approach involves removing any outliers before training the ML model under the hypothesis that this would increase the statistical power, improve model accuracy and reduce underfitting. However, these extreme values may instead indicate anomalies within the data, not necessarily 'outliers' in the traditional sense. Then without first discriminating whether these values are truly outliers, their removal could prevent us from testing the model's performance using anomalies representative of what could naturally occur in practice. Meanwhile, the second approach involves preserving the outliers under the assumption that these extreme values do not derive from any measurement or processing errors, incorrect data entry or poor sampling. In this case, the 'outliers' are considered to be statistically significant and should be incorporated into the development of an 'accurate' ML model. Thus, the disadvantages of this approach are contrary to the advantages of the first; a decrease in statistical power and reduction in model accuracy. Thus, the decision to remove outliers is typically context dependent and generally left to the analyst's discretion. 

For our case, and to keep our efforts within the project scope, we simplify matters by relating these characteristics (age, debt, years of employment, credit score and income) to the context of the data. Firstly, we are confident that banking institutions adhere to strict data entry practices. Thus, the possibility of these extreme values deriving from a clerical error is low. Secondly, the customer would have been carefully vetted throughout the approval process as per the standard policies normally in place. Then without taking other factors into account, the likelihood of a customer being approved or denied solely based on any one factor scoring undesirably can be assumed to be low. This can be generalised to how the decision to approve or deny a credit card application is multi-facted and considers these characteristics as a whole to be reflective of a customer's unique situation. That is, the combination of these characteristics give the most influence towards a decision, without one characteristic necessarily being preferred over another. With these points in mind, we deem these outliers to be statistically significant and opt for their inclusion during the development of the ML models.

## 4. Classification using Machine Learning
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

With the data now preprocessed and ready for training, three ML algorithms will be implemented: Logistic Regression, KNN and Random Forest. As mentioned earlier, properly building their accompanying ML models requires a determination of their hyperparameters; a process within the machine learning stage referred to as 'tuning'. This is due to these hyperparmeters needing manual specification as the algorithms do not learn for these values from the training data. This is the motivation behind the validation set, which offers the algorithms a separate partition of the original data on which to tune these hyperparameters while avoiding any data leakage between the training and testing sets. This leads to the traditional "three-way holdout" method, where the original data is partitioned into a training, validation and testing set. This way, a reasonable approach to tuning would entail iterating through combinations of hyperparameter values, using these values and the training set to build a specific model, then evaluating its performance using the validation set. Once a model is found which optimises a desired performance metric, such as classification accuracy or Receiver Operating Characteristic (ROC) curve, its corresponding hyperparameters are then used to retrain the algorithm on a recombination of the training and validation sets. Finally, the performance of the resulting model is assessed through a single application to the testing set. While this method of tuning avoids any data leakage between the testing and training sets, reusing the validation set when iterating through hyperparameters biases the final model to this data and can result in overly optimistic estimates of its generalised performance. In essence, the validation set ironically leaks information to the training set. Meanwhile, for those scenarios involving smaller datasets (such as ours), partitioning via the three-way holdout method while providing the algorithm with enough training data to reduce pessimistic bias (model capacity) is not always feasible; thus our partitioning the data via the two-way holdout method in Section 3.1. The following subsection delineates our application of a method that addresses these shortcomings brought about from the small dataset.

### 4.1 Hyperparameter Tuning via Nested k-fold Cross-Validation
To circumvent these issues as best as possible, we use the well-known nested k-fold cross-validation method to tune the hyperparameters for these three algorithms. This method entails a robust procedure for experimentally performing hyperparameter tuning on smaller datasets without succumbing to data leakage. Such a method also enables us to compare the generalised performances of the algorithms under test as each data point is used during validation.

We begin by initialising the three classifiers.
```python
# Initialising the classifiers
clf1 = LogisticRegression(solver = 'newton-cg', random_state = 42)
clf2 = KNeighborsClassifier(algorithm = 'ball_tree')
clf3 = RandomForestClassifier(random_state = 42)
```
We then set up a dictionary of parameters for the grid search method to use during the nested k-fold cross-validation. To simplify matters for this project, we tune for only one parameter for each of the algorithms: the penalty for Logistic Regression (none or L2), the number of neighbours for KNN, and the maximum tree depth for Random Forest. Note that this code can be easily expanded to include any other hyperparameters and combinations thereof for these algorithms, and others.
```python
# Setting up the parameter grids
param_grid1 = [{'penalty': ['l2'] + [None]}]
param_grid2 = [{'n_neighbors': list(range(1, 10))}]
param_grid3 = [{'max_depth': list(range(1, 10)) + [None]}]
```
Initialise an array whose elements contain a GridSearchCV object for each algorithm.
```python
# Initialise array of GridSearchCV objects
gridcvs = {}
```
Now, generate the train/test split indices for the inner loop of the nested k-fold cross validation. For the inner loop, we apply a 2-fold cross-validation. Then, use a for loop to iterate through classifiers ```clf1```, ```clf2``` and ```clf3``` and initialise an exhaustive parameter grid search for each, whose grids are respectively specified by ```param_grid1```, ```param_grid2``` and ```param_grid3```.
```python
# Generate train/test split indices for inner loop of cross-validation
inner_cv = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 42)
# Generate array containing a GridSearchCV object for each estimator
for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3), (clf1, clf2, clf3), ('LR', 'KNN', 'RF')):
    gcv = GridSearchCV(estimator = est, param_grid = pgrid, scoring = 'accuracy', n_jobs = 1, cv = inner_cv, verbose = 0, refit = True)
    gridcvs[name] = gcv
```
Generate the train/test split indices for the outer loop of the nested k-fold cross validation. For the outer loop, we apply a 5-fold cross-validation. We then parse the ```gridcvs``` array to a for loop which executes the nested k-fold cross-validation method using ```cross_val_score```.
```python
# Generate train/test split indices for outer loop of cross-validation
outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
# Perform nested k-fold cross validation
for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, X = X_train_scaled, y = y_train, cv = outer_cv, n_jobs = 1)
    print('%s | outer ACC: %f%% +- %f' % (name, nested_score.mean() * 100, nested_score.std() * 100))
print('\n')
```
This generates the following results:
```python
KNN | outer ACC: 87.149877% +- 3.406925
LR | outer ACC: 87.511876% +- 2.711699
RF | outer ACC: 88.597871% +- 2.849419
```
We see that all three models exhibit good performance and give similar accuracy scores, with the Random Forest algorithm scoring the highest. 

### 4.2 Retraining with Optimised Hyperparameters
We now retrain the algorithms using their optimised hyperparameters on the complete training set ```X_train_scaled``` to assess their generalised performances. Note that we first assign ```target_names``` as an array containing the labels ```-``` and ```+``` for the two target classes, in that order, so as to remain consistent with the output of the ```confusion_matrix``` method from the ```sklearn.metrics``` library. Here, ```-``` indicates the class of rejected applications while ```+``` indicates the class of approved applications, with us referring to these classes as 'negative' and 'positive' respectively.
```python
# Assign target names contingent to formatting of output from sklearn.metrics.confusion_matrix
target_names = ['-', '+']
# Retrain classifiers with optimised hyperparameters and print their scores
for i, name in enumerate(list(('LR', 'KNN', 'RF'))):
    print('//-----' + name + '-----//\n')
    opt_algo = gridcvs[name]
    opt_algo.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_true = y_train, y_pred = opt_algo.predict(X_train_scaled))
    test_acc = accuracy_score(y_true = y_test, y_pred = opt_algo.predict(X_test_scaled))
    conf_matrix = confusion_matrix(y_true = y_test, y_pred = opt_algo.predict(X_test_scaled))
    class_report = classification_report(y_true = y_test, y_pred = opt_algo.predict(X_test_scaled), target_names = target_names)

    print('Accuracy (averaged over CV test folds): %f%%\n' % (opt_algo.best_score_ * 100))
    print('Best parameters: %s\n' % opt_algo.best_params_)
    print('Training accuracy: %f%%\n' % (train_acc * 100))
    print('Testing accuracy: %f%%\n' % (test_acc * 100))
    print('Confusion matrix: \n')
    print(conf_matrix)
    print('Classification report: \n')
    print(class_report)
```
```python
//-----LR-----//
Accuracy (averaged over CV test folds): 86.956522%
Best parameters: {'penalty': 'l2'}
Training accuracy: 87.862319%
Testing accuracy: 78.985507%
Confusion matrix: 
[[56 21]
 [ 8 53]]
Classification report: 
              precision    recall  f1-score   support
           -       0.88      0.73      0.79        77
           +       0.72      0.87      0.79        61
    accuracy                           0.79       138
   macro avg       0.80      0.80      0.79       138
weighted avg       0.80      0.79      0.79       138
//-----KNN-----//
Accuracy (averaged over CV test folds): 86.956522%
Best parameters: {'n_neighbors': 8}
Training accuracy: 88.949275%
Testing accuracy: 81.159420%
Confusion matrix: 
[[60 17]
 [ 9 52]]
Classification report: 
              precision    recall  f1-score   support
           -       0.87      0.78      0.82        77
           +       0.75      0.85      0.80        61
    accuracy                           0.81       138
   macro avg       0.81      0.82      0.81       138
weighted avg       0.82      0.81      0.81       138
//-----RF-----//
Accuracy (averaged over CV test folds): 87.500000%
Best parameters: {'max_depth': 3}
Training accuracy: 89.673913%
Testing accuracy: 84.057971%
Confusion matrix: 
[[69  8]
 [14 47]]
Classification report: 
              precision    recall  f1-score   support
           -       0.83      0.90      0.86        77
           +       0.85      0.77      0.81        61
    accuracy                           0.84       138
   macro avg       0.84      0.83      0.84       138
weighted avg       0.84      0.84      0.84       138
```

## 5. Results and Evaluation
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

Firstly, inspect the training and testing accuracies for the three models. We see that for all three classifiers, their training accuracies are within sensible percentage margins and reflect that none of the models appear to be overfitting. This is particularly so for the RF model, as tree-based models are usually more prone to overfitting. However, its training accuracy of ~89.67% indicates there is still a good degree of generalised performance; likewise for the LR and KNN models. Meanwhile, compared to their training accuracies, the testing accuracy of the LR model exhibits a -8.876812% difference, the KNN model a -7.789855% difference, and the RF model a -5.615942% difference; the testing accuracies for all models lie within -10% of their training accuracies. At this point of our discussion, we make the following remark: contingent with how a nested k-fold cross-validation procedure enables a fairer assessment of the best competing model, it would be remiss for us to conclude from just the training and testing accuracies alone that the RF model is better overall. This is because the determination of which model is truly 'better' is a much more complex matter. A true determination should require us to not only consider other useful statistical metrics (such as ROC AUC scores, etc) and methods of analyses (such as the Bayesian Test or McNemar's Test), but to look further beyond them. That is, despite how these metrics help quantify a model's performance, the determination of which is better is ultimately domain-specific and shoulder consider the context and importance of each metric with regards to the specific application. In essence, a top-down approach is normally exercised: for a classification problem, this could involve firstly prioritising the classifications, then working backwards to conceptualise how proxy metrics such as false negative and false positive percentages can bias these priorities accordingly, followed by assessing an array of ML models through a battery of relevant metrics, along with any further statistical analyses.

Given these complexities, and to avoid further extending this body of work, we proceed to cross-examine our three models using only the above metrics and relating them to the context of credit card application approvals. While simplified in nature, this approach already succeeds in achieving the initial outcome of this project, which was to further consolidate on our understanding of those principles fundamental to machine learning. We note that more rigorous analyses utilising more involved statistical metrics will become the subject of future work.

### 5.1 Comparison based on precision score
For the positive class, we see the precision obtained by LR was 72%, KNN was 75%, and RF was 85%. For the negative class, the precision obtained by LR was 88%, KNN was 87% and RF was 83%. Based on this metric alone, the RF model attains the highest precision of 85% for the positive class, while LR attains the highest precision of 88% for the negative class.

### 5.2 Comparison based on recall score
For the positive class, we see the recall obtained by LR was 87%, KNN was 85%, and RF was 77%. For the negative class, the recall obtained by LR was 73%, KNN was 78%, and RF was 90%. Based on this metric alone, the LR model attains the highest recall of 87% for the positive class, while RF attains the highest recall of 90% for the negative class.

### 5.3 Comparison based on F1 score
For the positive class, we see the F1 score obtained by LR was 79%, KNN was 80%, and RF was 81%. For the negative class, the F1 score obtained by LR was 79%, KNN was 82%, and RF was 86%. Based on this metric alone, the RF model attained the highest F1 scores for both the positive and negative classes, of 81% and 86% respectively.

Having summarised these results, we now relate the metrics of precision, recall and F1 score to the classification problem at hand. We first generalise that for any problem involving classification, models possessing higher precision are cautious of making positive predictions (as false positives are deemed undesirable), while models possessing higher recall are cautious of making negative predictions (as false negatives are deemed) undesirable. Translating this to our context, we can ascertain the following for the positive and negative classes. For the positive class (application is approved), the bank should aim to minimise the occurrence of false negatives. This means a model with a higher recall score for the positive class would prove more beneficial for correctly approving an application. In this case, the LR model serves as an appropriate solution, yielding the highest recall score of 87%. For the negative class (application is denied), the bank should aim to maximise the occurrence of false positives. This means a model with a higher precision score for the negative class would prove more beneficial for correctly rejecting an application. In this case, the LR model again serves as an appropriate solution, yielding the highest precision score of 88%. Now, we look to the F1 score metric, which functions as the harmonic mean of the precision and recall scores and accounts for not only the number of predictions that were correct, but also the type of errors that were incurred (such as false positives and false negatives). The RF model attains the highest F1 score out of the models for both positive and negative classes. Given that our dataset is balanced, it would be reasonable to conclude that either the LR or RF models would be appropriate for application out of the three. If the dataset were imbalanced, further scrutiny of these metrics, and others, would be required.

## 6. Conclusion and Future Work
<p align="right"> <a href="#predicting-credit-card-approvals-using-machine-learning">🔼 back to top</a> </div>

The aim of this project was to consolidate on our understanding of those principles fundamental to machine learning and to enable knowledge expansion of this domain. Through the context of predicting credit card approvals, we engineered statistical methods with machine learning algorithms in a manner congruent with their applications within the data science discipline. We implemented three popular classification algorithms (Logistic Regression, k-Nearest Neighbours and Random Forest) whose ensuing models were compared using the precision, recall and F1 score metrics. It was determined that the Random Forest model would be the most appropriate for this scenario, attaining the highest score across these metrics. Future efforts involve extending this project so as to further advance our knowledge of machine learning methods and the data science practice as a whole. Possible extensions would include properly assessing the distribution of the numerical features via more rigorous statistical methods, such as hypothesis tests or non-parametric tests. This would allow for a better determination of whether the extreme values in the dataset can be considered as true outliers. The effect of removing any outliers on the performance of these three ML models can then be examined. Additionally, further exploration of those hyperparameters specific to these algorithms, and combinations thereof, can be studied in conjunction with their effects on their model's predictive power. This would also be supplemented with exploring other classification algorithms, such as support vector machines (SVM) or gradient boosting.







