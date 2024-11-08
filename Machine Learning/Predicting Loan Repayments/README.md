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


## 2. Data Exploration



## 3. Preprocessing the data


## 4. Classification using Machine Learning


## 5. Results and Evaluation


## 6. Conclusion and Future Work
