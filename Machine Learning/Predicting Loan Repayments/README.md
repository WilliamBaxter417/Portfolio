# Predicting Loan Repayments using Decision Trees and Feature Selection

## 1. Import packages and dataset
We begin by importing the necessary Python libraries.
```python
# Math and Plotting libraries
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.patches import Patch

# Personal libraries
import PLRsubs

# Miscellaneous initialisations
pd.set_option('display.max_columns', 50)
matplotlib.use('TkAgg')
```
This project utilises the publicised data from [LendingClub](https://archive.ics.uci.edu/dataset/27/credit+approval) imported from the UCI Machine Learning Repository which has been anonymised for confidentiality. We import the dataset as follows:
```python
# Load dataset from CSV file into a dataframe
data = pd.read_csv("lending_club_loan_two.csv")
```
Perform an initial inspection of the data (along with viewing the dataframe within the IDE):
```python
# Initial inspection of data
data.head()
```
```python
   loan_amnt        term  int_rate  installment grade sub_grade               emp_title  emp_length  home_ownership  annual_inc  verification_status   issue_d  loan_status             purpose                    title    dti  earliest_cr_line  open_acc  pub_rec   revol_bal  revol_util  total_acc  initial_list_status  application_type   mort_acc  pub_rec_bankruptcies                                             address
0    10000.0   36 months     11.44       329.48     B        B4               Marketing   10+ years            RENT    117000.0         Not Verified  Jan-2015   Fully Paid            vacation                 Vacation  26.24          Jun-1990      16.0      0.0     36369.0        41.8       25.0                    w        INDIVIDUAL        0.0                   0.0      0174 Michelle Gateway\r\nMendozaberg, OK 22690              
1     8000.0   36 months     11.99       265.68     B        B5          Credit analyst     4 years        MORTGAGE     65000.0         Not Verified  Jan-2015   Fully Paid  debt_consolidation       Debt consolidation  22.05          Jul-2004      17.0      0.0     20131.0        53.3       27.0                    f        INDIVIDUAL        3.0                   0.0   1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113        
2    15600.0   36 months     10.49       506.97     B        B3            Statistician    < 1 year            RENT     43057.0      Source Verified  Jan-2015   Fully Paid         credit_card  Credit card refinancing  12.79          Aug-2007      13.0      0.0     11987.0        92.2       26.0                    f        INDIVIDUAL        0.0                   0.0   87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113                   
3     7200.0   36 months      6.49       220.65     A        A2         Client Advocate     6 years            RENT     54000.0         Not Verified  Nov-2014   Fully Paid         credit_card  Credit card refinancing   2.60          Sep-2006       6.0      0.0      5472.0        21.5       13.0                    f        INDIVIDUAL        0.0                   0.0             823 Reid Ford\r\nDelacruzside, MA 00813                            
4    24375.0   60 months     17.27       609.33     C        C5  Destiny Management Inc.    9 years        MORTGAGE     55000.0             Verified  Apr-2013  Charged Off         credit_card    Credit Card Refinance  33.95          Mar-1999      13.0      0.0     24584.0        69.8       43.0                    f        INDIVIDUAL        1.0                   0.0              679 Luna Roads\r\nGreggshire, VA 11650             
```
Before generating summary statistics, we gather some base information about the dataframe.
```python
# Get information about dataframe
data.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 396030 entries, 0 to 396029
Data columns (total 27 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   loan_amnt             396030 non-null  float64
 1   term                  396030 non-null  object 
 2   int_rate              396030 non-null  float64
 3   installment           396030 non-null  float64
 4   grade                 396030 non-null  object 
 5   sub_grade             396030 non-null  object 
 6   emp_title             373103 non-null  object 
 7   emp_length            377729 non-null  object 
 8   home_ownership        396030 non-null  object 
 9   annual_inc            396030 non-null  float64
 10  verification_status   396030 non-null  object 
 11  issue_d               396030 non-null  object 
 12  loan_status           396030 non-null  object 
 13  purpose               396030 non-null  object 
 14  title                 394274 non-null  object 
 15  dti                   396030 non-null  float64
 16  earliest_cr_line      396030 non-null  object 
 17  open_acc              396030 non-null  float64
 18  pub_rec               396030 non-null  float64
 19  revol_bal             396030 non-null  float64
 20  revol_util            395754 non-null  float64
 21  total_acc             396030 non-null  float64
 22  initial_list_status   396030 non-null  object 
 23  application_type      396030 non-null  object 
 24  mort_acc              358235 non-null  float64
 25  pub_rec_bankruptcies  395495 non-null  float64
 26  address               396030 non-null  object 
dtypes: float64(12), object(15)
memory usage: 81.6+ MB
```
There are 27 total features and 396030 samples. As our focus is on building a model which predicts whether a loan has been fully repaid, it is clear that the ```loan_status``` feature is the only target variable. We also see that some features include missing values (non-null count field does not equal 396030). The table below summarises the descriptions for each of the features present in the dataframe.

| Feature | Description |
|---------|-------------|
| loan_amnt | Listed loan amount applied for by the borrower (any adjustments to loan amount were reflected in this value). |
| term | Number of payments on the loan (values are in months, and are either 36 or 60). |
| int_rate | Interest rate on the loan. |
| installment | Monthly payment owed by the borrower (if loan is approved). |
| grade | Loan grade (assigned by LendingClub) |
| sub_grade | Loan subgrade (assigned by LendingClub) |
| emp_title | Job title listed by the borrower during registration. |
| emp_length | Employment length (values are in years, ranging from 0 to 10, with 0 meaning < 1 year and 10 meaning >= 10 years). |
| home_ownership | Home ownership status provided by the borrower during registration (values are categories RENT, OWN, MORTGAGE, OTHER, NONE ANY). |
| annual_inc | Self-reported annual income provided by the borrower during registration. |
| verification_status | Indicates if income was verified by LendingClub, or if the income source was verified. |
| issue_d | Month in which the loan was funded. |
| loan_status | Current status of the loan. |
| purpose | Category defining reason for loan request as selected by the borrower. |
| title | Loan title provided by the borrower. |
| zip_code | First 3 numbers of the zip code provided by the borrower during registration. |
| addr_state | State provided by the borrower during registration. |
| dti | Debt-to-income ratio as calculated using the borrower’s total monthly debt payments (excluding mortgages and the requested LendingClub loan) divided by the borrower’s self-reported monthly income. |
| earliest_cr_line | Month the borrower's earliest reported credit line was opened. |
| open_acc | Number of open credit lines in the borrower's credit file. |
| pub_rec | Number of derogatory public records. |
| revol_bal | Total credit revolving balance. |
| revol_util | Revolving line utilization rate, defined as the amount of credit the borrower is using relative to all available revolving credit. |
| total_acc | Total number of credit lines currently in the borrower's credit file. |
| initial_list_status | Initial listing status of the loan (values are 'w', 'f'). |
| application_type | Indicates whether loan is an individual application or a joint application with two co-borrowers. |
| mort_acc | Number of mortgage accounts. |
| pub_rec_bankruptcies | Number of publicly recorded bankruptcies. |

Before performing a formal EDA, we first separate the feature variables and target variable.
```python
# Separate feature and target
y = data['loan_status']
X_features = data.drop('loan_status', axis = 1)
```
At this point, we split the data into the train and test sets. We do this before proceeding to any formal EDA as we wish to gather statistics about the data only from the train set. This ensures we minimise data leakage between it and the test set later on. As the number of samples is large (396030), one could justify that stratifying our split according to how the target variable outcomes are distributed may not be necessary. However, we do so regardless not only to preserve this distribution across both sets, but in case this distribution is imbalanced and sees the splitting procedure assigning an inadequate number of samples from the underepresented class to the train set.
```python
# Split into train and test sets (with stratify)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size = 0.33, random_state = 42, stratify = y)
```

## 2. Data Exploration
We first look at the target variable and determine the distribution of the binary outcomes.
```python
# List possible target outcomes and their count values
y_train.value_counts()
```
```python
loan_status
Fully Paid     213299
Charged Off     52041
Name: count, dtype: int64
```
Out of the 265340 total samples, 213299 are Fully Paid while 52041 are Charged Off. Determining the proportion:
```python
# Normalise the above counts
y_train.value_counts(normalize = True)
```
```python
loan_status
Fully Paid     0.803871
Charged Off    0.196129
Name: proportion, dtype: float64
```
The Fully Paid outcome accounts for approximately 80% of the results and Charged Off approximately 20%, and so the data in ```X_train``` is imbalanced. Visualising this distribution a bar chart:
```python
# Get x-labels
loan_status_values = y_train.value_counts().keys().tolist()
# Get y-values
loan_status_counts = y_train.value_counts().tolist()
# Generate bar plot
fig, ax = plt.subplots(1, 1, figsize = (7, 5), sharex = True)
ax.bar(x = loan_status_values, height = loan_status_counts, edgecolor = 'black')
plt.xlabel('Loan status', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()
```
<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Predicting%20Loan%20Repayments/images/LoanStatusDistribution.png" />
</p>

Continuing with our EDA, we reinspect our train set:
```python
# Reinspect our train set
X_train.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
Index: 265340 entries, 252076 to 39658
Data columns (total 26 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   loan_amnt             265340 non-null  float64
 1   term                  265340 non-null  object 
 2   int_rate              265340 non-null  float64
 3   installment           265340 non-null  float64
 4   grade                 265340 non-null  object 
 5   sub_grade             265340 non-null  object 
 6   emp_title             250048 non-null  object 
 7   emp_length            253185 non-null  object 
 8   home_ownership        265340 non-null  object 
 9   annual_inc            265340 non-null  float64
 10  verification_status   265340 non-null  object 
 11  issue_d               265340 non-null  object 
 12  purpose               265340 non-null  object 
 13  title                 264174 non-null  object 
 14  dti                   265340 non-null  float64
 15  earliest_cr_line      265340 non-null  object 
 16  open_acc              265340 non-null  float64
 17  pub_rec               265340 non-null  float64
 18  revol_bal             265340 non-null  float64
 19  revol_util            265161 non-null  float64
 20  total_acc             265340 non-null  float64
 21  initial_list_status   265340 non-null  object 
 22  application_type      265340 non-null  object 
 23  mort_acc              240144 non-null  float64
 24  pub_rec_bankruptcies  264983 non-null  float64
 25  address               265340 non-null  object 
dtypes: float64(12), object(14)
memory usage: 54.7+ MB
```
We have 26 features and 265340 samples. Our train data is a combination of numerical and categorical features. Let us capture the column labels corresponding to those numerical and categorical features. For convenience, we implement this by building a function ```get_categorical_numerical_headers()``` in a separate module called ```PLRsubs```.
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
        obj_col[i] = df.columns[obj_idx[i]]
    for i in np.arange(num_idx.shape[0]):
        # Generate headers with string arithmetic and type conversion
        num_col[i] = df.columns[num_idx[i]]

    return obj_idx, num_idx, obj_col, num_col
```
After importing this module, we call this function to obtain the following:
```python
# Get labels of categorical and numerical features
cat_idx, num_idx, cat_label, num_label = PLRsubs.get_categorical_numerical_headers(X_train)
print('Categorical features:')
print('---------------------')
for i in np.arange(len(cat_label)):
    print('"%s" => index: %d' % (cat_label[i], cat_idx[i]))
print('\n')
print('Numerical features:')
print('-------------------')
for i in np.arange(len(num_label)):
    print('"%s" => index: %d' % (num_label[i], num_idx[i]))
```
```python
Categorical features:
---------------------
"term" => index: 1
"grade" => index: 4
"sub_grade" => index: 5
"emp_title" => index: 6
"emp_length" => index: 7
"home_ownership" => index: 8
"verification_status" => index: 10
"issue_d" => index: 11
"purpose" => index: 12
"title" => index: 13
"earliest_cr_line" => index: 15
"initial_list_status" => index: 21
"application_type" => index: 22
"address" => index: 25

Numerical features:
-------------------
"loan_amnt" => index: 0
"int_rate" => index: 2
"installment" => index: 3
"annual_inc" => index: 9
"dti" => index: 14
"open_acc" => index: 16
"pub_rec" => index: 17
"revol_bal" => index: 18
"revol_util" => index: 19
"total_acc" => index: 20
"mort_acc" => index: 23
"pub_rec_bankruptcies" => index: 24
```
We now use a for loop to determine exactly how many features have missing values and for each, the proportion of missing values relative to the size of the train set.
```python
# Determine number of missing values and their proportion relative to the train set.
for column in X_train.columns:
    if X_train[column].isna().sum() != 0:
        missing = X_train[column].isna().sum()
        portion = (missing / X_train.shape[0]) * 100
        if type(X_train[column]) == 'O':
            print('%s (categorical) => Number of missing values: %d ==> %f%%' % (column, missing, portion))
        else:
            print('%s (numerical) => Number of missing values: %d ==> %f%%' % (column, missing, portion))
```
```python
emp_title (numerical) => Number of missing values: 15292 ==> 5.763172%
emp_length (numerical) => Number of missing values: 12155 ==> 4.580915%
title (numerical) => Number of missing values: 1166 ==> 0.439436%
revol_util (numerical) => Number of missing values: 179 ==> 0.067461%
mort_acc (numerical) => Number of missing values: 25196 ==> 9.495741%
pub_rec_bankruptcies (numerical) => Number of missing values: 357 ==> 0.134544%
```
From the above, ```emp_title```, ```emp_length``` and ```title``` are categorical features, while ```revol_util```, ```mort_acc``` and ```pub_rec_bankruptcies``` are numerical features. The ```mort_acc``` feature has the highest proportion of missing values, while the ```revol_util``` and ```pub_rec_bankruptcies``` features have the lowest proportions.

Now, given there are 26 features comprising the training data, let us determine whether any features exhibit collinearity. We first do this for the numerical features, plotting a heatmap matrix of the Spearman's rank correlation coefficients between them. A coefficient of 1 indicates that a pair of features are highly collinear. Determining collinearity across these features may give early indication of which can be ignored during any future feature selection.
```python
# Plot heatmap of Spearman's rank correlation coefficient for numerical features
plt.figure(figsize = (12, 8))
sns.heatmap(X_train[num_label].corr(method = 'spearman'), annot = True, center = 1)
plt.title('Heatmap of Spearman rank correlation coefficient')
plt.show()
```
<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Predicting%20Loan%20Repayments/images/heatmapSpearman.png" />
</p>

We see that loan_amnt and installment exhibit a high degree of correlation, with a coefficient of 0.97. Let us visualise a bivariate scatter plot of these two features.
```python
# Draw scatter plot between loan_amnt and installment features.
plt.figure(figsize = (12, 8))
ax = sns.regplot(data = X_train, x = 'loan_amnt', y = 'installment', fit_reg = True, marker = 'x', color = '.3', line_kws = dict(color = 'r'))
ax.set(xlabel = 'Loan Amount', ylabel = 'Installment', title = 'Scatter plot of Installment vs. Loan amount')
ax.set_xlim(xmin = 0)
ax.set_ylim(ymin = 0, ymax = 1600)
plt.grid()
plt.show()
```
<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Predicting%20Loan%20Repayments/images/scatterplot_installment_vs_loan_amnt.png" />
</p>
In this dataset, ```loan_amnt``` is the listed loan amount applied for by the borrower, while ```installment``` is the monthly payment owed by the borrower if the loan is approved. Then, it makes sense that these two features are collinear (and thus linearly correlated), so there is nothing out of the ordinary here. Looking back at the heatmap, this would explain why ```loan_amnt``` and ```installment``` exhibit similar correlation coefficients with the remaining numerical features. This means that we can drop either one of these features later on. We proceed by plotting the histograms of ```loan_amnt``` and ```installment``` by ```loan_status``` in order to visualise their distributions. We facilitate this by creating a helper function, ```hist_by_loan_status```, in the ```PLRsubs``` module.

```python
def hist_by_loan_status(X_train, y_train, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = vectorise_fun(np.histogram(np.hstack((X_train[label][y_train == 'Fully Paid'], X_train[label][y_train == 'Charged Off'])), bins = 50)[1])
    X_train[label][y_train == 'Fully Paid'].hist(bins = bins, color = 'blue', edgecolor = 'black', alpha = 0.5)
    X_train[label][y_train == 'Charged Off'].hist(bins = bins, color = 'red', edgecolor = 'black', alpha = 0.5)

   return fig, ax, bins
```
Then calling this function to generate the plots:
```python
# Plot histogram for Installment feature.
fig_installment, ax_installment, bins_installment = PLRsubs.hist_by_loan_status(X_train, y_train, 'installment')
fig_installment = PLRsubs.bins_labels(bins_installment, fig_installment, ax_installment)
plt.xlabel('Installment')
plt.ylabel('Count')
plt.title('Installment by Loan Status')
plt.legend(['Fully Paid', 'Charged Off'], title = 'Loan status')
plt.grid(False)

# Plot histogram for loan_amnt feature.
fig_loan_amnt, ax_loan_amnt, bins_loan_amnt = PLRsubs.hist_by_loan_status(X_train, y_train, 'loan_amnt')
fig_loan_amnt = PLRsubs.bins_labels(bins_loan_amnt, fig_loan_amnt, ax_loan_amnt)
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.title('Loan Amount by Loan Status')
plt.legend(['Fully Paid', 'Charged Off'], title = 'Loan status')
plt.grid(False)
```
<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Predicting%20Loan%20Repayments/images/hist_installment_by_loan_status.png" />
</p>

<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Machine%20Learning/Predicting%20Loan%20Repayments/images/hist_loan_amnt_by_loan_status.png" />
</p>
In the installment histogram, the distributions of the Fully Paid and Charged Off borrowers are similar in shape. Both distributions exhibit a clear bell-curve shape, appearing to be normal and right-skewed. For both target classes, this implies their means are greater than the medians. In the histogram for loan amount, the distributions of the Fully Paid and Charged Off borrowers are also similar in shape. However, compared to the installment histogram, we see more fluctuations in count values across the bins for both target classes. Interestingly, within each bin, the relative difference between the two target classes appears similar. We now draw box-plots for the installments and loan amount by loan status.
























## 3. Preprocessing the data


## 4. Classification using Machine Learning


## 5. Results and Evaluation


## 6. Conclusion and Future Work
