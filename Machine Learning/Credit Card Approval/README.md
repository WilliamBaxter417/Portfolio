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
The dataset contains 690 rows with some missing values, with the 'original' key being composed of the columns from the 'features' and 'targets' keys. As such, we will use the 'original' key to perform an initial exploration of the data, whose attributes are:
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
- A16: +,- (class)

## 2. Inspecting the data
Beginning our inspection of the data, we check the fields for those variables comprising the `credit_approval` dataframe.
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
To gain further understanding of the dataset, we check the information and summary statistics.
```python
credit_df = credit_approval.data.original
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
The above summary statistics have been automatically limited to those columns possessing a numerical data type. Furthermore, the ```count``` field reveals that columns A2 and A14 do not include all 690 rows of the dataset. Despite the previous ```.info()``` method returning datatypes for A2 and A14 as ```float64``` and ```int64``` respectively, this implies that some of their entries must be missing. This is confirmed by checking with the following,
```python
sum(credit_df.A2.isnull())
```
```python
12
```
```python
sum(credit_df.A14.isnull())
```
```python
13
```
which reveals that columns A2 and A14 respectively contain 12 and 13 missing entries. Then limiting our summary statistics to columns A3, A8, A11 and A15, we have:
```python
credit_df[["A3","A8","A11","A15"]].describe()
```
```python
               A3          A8        A11            A15
count  690.000000  690.000000  690.00000     690.000000
mean     4.758725    2.223406    2.40000    1017.385507
std      4.978163    3.346513    4.86294    5210.102598
min      0.000000    0.000000    0.00000       0.000000
25%      1.000000    0.165000    0.00000       0.000000
50%      2.750000    1.000000    0.00000       5.000000
75%      7.207500    2.625000    3.00000     395.500000
max     28.000000   28.500000   67.00000  100000.000000
```
We also realise that the range values in column A15 are several orders of magnitude greater than the other numerical columns. To ensure the cost functions employed by the later ML models can correctly converge to a minimum, feature scaling techniques must be applied later on.

## 3. Preprocessing the data
Following our earlier inspection of the data, it is clearly necessary we preprocess the data before building our ML models. The preprocessing sequence can be broken down into the following tasks:

- Splitting the dataset into training and testing sets.
- Imputing the missing data.
- Converting non-numerical data to numerical.
- Scaling the feature values to a uniform range.

At this point of the preprocessing stage, we make two remarks: first, it is important to impute any missing information in both the training and testing datasets. Ignoring any missed values can negatively affect the performance of the predictive model, with some ML algorithms requiring a complete dataset for their successful operation (such as Logistic Regression). Second, we note how it is ideal to first split the data into the training and testing datasets prior to imputing. This is due to what the training and testing datasets attempt to replicate in practice: the training dataset comprises historical information that is used to first build the predictive model, while the testing dataset serves as future unknown information which the model ingests to predict an outcome. In essence, the training dataset represents the past, with the testing dataset representing the future. Since the ML model is constructed solely from the training dataset, any preprocessing of the data, such as imputing missing values, should be performed on the training dataset alone. In other words, any imputation method applied to the training dataset should consider only those statistics of the training dataset. Then, if the testing dataset also requires imputing, its imputation method should follow that which was applied to the training dataset. This procedure ensures that no information from the testing data is used to preprocess the training data, which would bias the construction of the ML model and its ensuing results. This concept is referred to as 'data leakage', and will be handled accordingly in this project prior to constructing our ML models later on. Our justification for opting to impute the missing values instead of omitting those samples entirely will also be covered.

### 3.1 Splitting the dataset into training and testing sets
- We first begin with a small discussion regarding the technique of _feature selection_ as it pertains to this dataset.
- In Section 1, the anonymised data contained within the ```credit_approval.data.original``` dataframe reveals very little about the nature of the features. However, [this](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) provides good insight to the features most typically used by banking institutions when considering credit card applications, such as Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus.
- Then with this knowledge in mind, we can determine with good confidence that the columns for the ```credit_approval.data.original``` dataframe map to the following features:
  
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

- With this improved understanding of the features comprising the dataframe, feature selection could be exercised. For instance, columns A12 and A14 could be dropped prior to splitting the dataset as drivers license type and zipcode could be deemed as relatively unimportant factors dictating the approval of a credit card application. This small-scale example of feature selection supports how good feature selection practices can facilitate ML modelling by reducing the number of overall features and the introduction of noise to the dataset; improving the performance, efficiency, and interpretability of the ML model.
- In practice however, a more robust method for determining which features are relevant for inclusion would involve measuring the statistial correlation between said features and the target. Since this is beyond the scope of the project, we proceed by adopting the popular convention of splitting the dataframe into its nominal 'features' and 'targets' variables.
```python
y = credit_df['A16']
Xfeatures = credit_df.drop(['A16'], axis = 1)
```
We then split the features and targets variables into their training and testing sets.
```python
# Import train_test_split
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(Xfeatures, y, test_size = 0.33, random_state = 42)
```

### 3.2 Imputing the missing data
Before discussing methods of imputing missing values in the training and testing datasets, we elaborate on our aforementioned choice of imputing the missing values over omitting those samples as a whole. As the total number of samples (690) comprising the original dataset is already small, then omitting any entirely could detract from the performance of the ensuing ML model. On the other hand, as only approximately 0.61% of the available values are missing, one could argue on reasonable ground for their omission. However, conventional practice in most cases is to provide the predictive model with as much training data as possible so as to maximise support during the learning stage. Additionally, any remaining features from those samples containing missing values would still contribute towards fine-tuning the underlying ML algorithm and improving the robustness of its predictive modelling. In conjunction, the decision to remove outliers from any dataset is typically context dependent and is often left to the designer's discretion. Given how the features of this dataset most likely reflect those characteristics outlined in Section 3.1 (gender, age, debt, marital status, etc), we deem it appropriate to preserve any outliers and for their inclusion during the training and testing stages of the ML models.

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

Since the missing data is either categorical or continuous in nature, we must implement separate imputation methods for them. Adopting conventional practice, we will apply the method of mean imputation to the continuous data (type ```float64```) and mode imputation to the categorical data (type ```O```). To facilitate this, we build the following subfunction, which takes a training and testing dataframe (in that order) and imputes them according to their data types.
```python
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

```
Input ```Xtrain``` and ```Xtest``` to this function (contained in ```CCASubs.py```) and verify that they contain no missing values.
```python
import CCASubs

CCASubs.impute_train_test(Xtrain, Xtest)
Xtrain.isna().sum()
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



### 3.3 Converting non-numeric to numeric
Since machine learning (ML) algorithms require all feature variables to be of the numeric data type, we will need to apply some preprocessing to the data.
