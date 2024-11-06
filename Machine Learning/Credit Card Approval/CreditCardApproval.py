###########################################################################################
##################################### IMPORT PACKAGES #####################################
###########################################################################################

# Math and Plotting libraries
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

# My subfunctions
import CCASubs

##########################################################################################
##################################### IMPORT DATASET #####################################
##########################################################################################

# Import UC Irvine Machine Learning Repository
from ucimlrepo import fetch_ucirepo
# Fetch dataset
credit_approval = fetch_ucirepo(id=27)

# Inspect credit_approval dataframes
print('credit_approval.data.features:\n')
print(credit_approval.data.features)
print('\n')
print('credit_approval.data.targets:\n')
print(credit_approval.data.targets)
print('\n')
print('credit_approval.data.original:\n')
print(credit_approval.data.original)

############################################################################################
##################################### DATA EXPLORATION #####################################
############################################################################################

print('credit_approval.variables:\n')
print(credit_approval.variables)

# Reassign credit_approval.data.original to credit_df
credit_df = credit_approval.data.original
# Check information
print('credit_df.info():\n')
print(credit_df.info())

# Check summary statistics
print('credit_df.describe():\n')
pd.options.display.max_columns = None
print(credit_df.describe())

# Extract targets
y = credit_df['A16']
# Extract features
X_features = credit_df.drop(['A16'], axis = 1)

# Get header indices corresponding to numerical features
num_idx = np.where(X_features.dtypes != 'O')[0]

## GENERATE HISTOGRAM OF NUMERICAL FEATURES BEFORE PREPROCESSING
# Get header names for the categorical and numerical columns
cat_cols, num_cols = CCASubs.get_categorical_numerical_headers(X_features)
# Array of characteristics
feature_character = ['Gender', 'Age', 'Debt', 'Marital status', 'Bank customer type', 'Education level', 'Ethnicity', 'Years of Employment', 'Prior default', 'Employment status', 'Credit score', 'Drivers license type', 'Citizenship status', 'Zipcode', 'Income']

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

# Reshape num_idx for easier indexing when labelling
num_idx_reshape = np.reshape(num_idx, (sp_row, sp_col))
# Reshape array of characteristics
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

## COMPARE OUTPUT OF BINARY CLASSIFIER
# Compare the number of approved and declined applications
fig, ax = plt.subplots(1, 1, figsize = (7, 5), sharex = True)
sns.countplot(data = credit_df, x = 'A16', edgecolor = "black", palette = "viridis", order = credit_df['A16'].value_counts().index)
total = credit_df['A16'].value_counts().sum()
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('Application status', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()

##############################################################################################
##################################### SPLITTING THE DATA #####################################
##############################################################################################

# Split X_features into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size = 0.2, random_state = 42, stratify = y)

#############################################################################################
##################################### IMPUTING THE DATA #####################################
#############################################################################################

# Count number of nan entries in X_train before imputing
print('Number of nan entries in X_train before imputing:\n')
print(X_train.isna().sum())
# Count number of nan entries in X_test before imputing
print('Number of nan entries in X_test before imputing:\n')
print(X_test.isna().sum())

# Impute X_train and X_test using the impute_train_test function
CCASubs.impute_train_test(X_train, X_test)

# Count number of nan entries in X_train after imputing
print('Number of nan entries in X_train after imputing:\n')
X_train.isna().sum()
# Count number of nan entries in X_test after imputing
print('Number of nan entries in X_test after imputing:\n')
X_test.isna().sum()


#############################################################################################
##################################### ENCODING THE DATA #####################################
#############################################################################################

# Print categorical columns for X_train and X_test
print('X_train (categorical columns):\n')
print(X_train[cat_cols])
print('\n')
print('X_test (categorical columns):\n')
print(X_test[cat_cols])

# Instantiate OrdinalEncoder() function
ordinal_encoder = OrdinalEncoder()
# Fit and transform the encoder to the training data
X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])
# Transform the testing data using the encoder previously fitted to the training data
X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

# Verify results of the ordinal encoder
print('X_train (encoded):\n')
print(X_train)
print('\n')
print('X_test (encoded):\n')
print(X_test)

## GENERATE HISTOGRAM OF NUMERICAL FEATURES AFTER SPLITTING AND IMPUTING
# Initialise (5 x 3) axes for subplots of histograms (there are 15 numerical features after encoding)
sp_row = 5
sp_col = 3
# Reshape num_cols for convenient indexing later on
num_col_reshape = np.reshape(X_train.columns, (sp_row, sp_col))
# Set number of bins to be the square root of the total number of samples
num_samples = X_train.shape[0]
num_bins = int(np.ceil(np.sqrt(num_samples)))
# Instantiate subplot from matplotlib.pyplot
fig, axs = plt.subplots(sp_row, sp_col)
# Reshape characteristics array for easier indexing when labelling
feature_character_reshape = np.reshape(feature_character, (sp_row, sp_col))
# Iterate over rows of subplot
for i in np.arange(sp_row):
    # Iterate over columns of subplot
    for j in np.arange(sp_col):
        # Plot histogram for corresponding column index of Xfeature dataframe
        axs[i,j].hist(X_train[num_col_reshape[i,j]], bins = num_bins, edgecolor = 'black', linewidth = 1.2)
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_reshape[i,j], fontsize = 10)
        axs[i,j].set_ylabel('Frequency')



############################################################################################
##################################### SCALING THE DATA #####################################
############################################################################################

# Instantiate MinMaxScaler() function
scaler = MinMaxScaler(feature_range = (0, 1))
# Fit and transform the scaler to the training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform the testing data using the scaler previously fitted to the training data
X_test_scaled = scaler.transform(X_test)

## GENERATE HISTOGRAM OF NUMERICAL FEATURES AFTER SCALING
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


## GENERATE BOXPLOT OF NUMERICAL FEATURES AFTER SCALING
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
        # Plot boxplot for corresponding column index of X_feature dataframe
        axs[i,j].boxplot(X_train_scaled_df[num_idx_reshape[i,j]], vert = 0, sym = 'rs', positions=[0], widths=[0.3])
        axs[i,j].set_xlabel(num_col_reshape[i,j] + ' - ' + feature_character_num_reshape[i,j], fontsize = 10)
        axs[i,j].get_yaxis().set_ticks([])
        plt.tight_layout()

###################################################################################################
##################################### MACHINE LEARNING MODELS #####################################
###################################################################################################

# Initialising the classifiers
clf1 = LogisticRegression(solver = 'newton-cg', random_state = 42)
clf2 = KNeighborsClassifier(algorithm = 'ball_tree')
clf3 = RandomForestClassifier(random_state = 42)

# Setting up the parameter grids
param_grid1 = [{'penalty': ['l2'] + [None]}]
param_grid2 = [{'n_neighbors': list(range(1, 10))}]
param_grid3 = [{'max_depth': list(range(1, 10)) + [None]}]

# Initialise array of GridSearchCV objects
gridcvs = {}

# Generate train/test split indices for inner loop of cross-validation
inner_cv = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 42)
# Generate array containing a GridSearchCV object for each estimator
for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3), (clf1, clf2, clf3), ('LR', 'KNN', 'RF')):
    gcv = GridSearchCV(estimator = est, param_grid = pgrid, scoring = 'accuracy', n_jobs = 1, cv = inner_cv, verbose = 0, refit = True)
    gridcvs[name] = gcv

# Generate train/test split indices for outer loop of cross-validation
outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
# Perform nested k-fold cross validation
for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, X = X_train_scaled, y = y_train, cv = outer_cv, n_jobs = 1)
    print('%s | outer ACC: %f%% +- %f' % (name, nested_score.mean() * 100, nested_score.std() * 100))
print('\n')

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
