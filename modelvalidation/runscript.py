from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from outliers import smirnov_grubbs as grubbs
from statsmodels import robust
import math
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectPercentile, chi2, RFE
from sklearn import preprocessing
from bubbly.bubbly import bubbleplot
import plotly_express as px
import plotly.figure_factory as ff
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as py
import plotly
from pandas.plotting import parallel_coordinates
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import joypy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# for some basic operations

# for visualizations

# for interactive visualizations
# python runscript.pyinit_notebook_mode(connected=True)


# for animated visualizations

# for data preprocess and statistical tests


# for modeling
# 1.1 read data into python and describe the datadf = pd.read_csv('insurance_claims-v2.csv', na_values = '?')
df = pd.read_csv('insurance_claims.csv', na_values='?')
df.head()
# check data size
df.shape
# Change setting to display all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Specify dependent variable
target = 'fraud_reported'

# Specify customer id
cust_id = 'policy_number'

# drop target and cust_id from the datset
x = df.drop([target, cust_id], axis=1)
x.head(5)
x.shape
# check the target variabel frequency
count_target = df['fraud_reported'].value_counts()
print(count_target)
print('missing value is:', len(df.fraud_reported)-df.fraud_reported.count())
y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
y.value_counts()
# let's take a look of the data relationship
# if we have a large dataset, then we should not use this due to too much time to cost
sns.pairplot(x)

fraud = df['fraud_reported'].value_counts()
label_fraud = fraud.index
size_fraud = fraud.values
colors = ['green', 'yellow']
trace = go.Pie(labels=label_fraud, values=size_fraud,
               marker=dict(colors=colors), name='Frauds')
layout = go.Layout(title='Distribution of Frauds')
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
# Retrieve all text variables from data
cat_cols = [c for i, c in enumerate(x.columns) if x.dtypes[i] in [np.object]]

# Retrieve all numerical variables from data
num_cols = [c for i, c in enumerate(
    x.columns) if x.dtypes[i] not in [np.object]]

# let's firstly see the feature self plots (distribution, box-plot, histogram)
# plot the distribution for all the numeric features

plt.figure(figsize=(50, 50))
k = 1
for i in num_cols:
    plt.subplot(math.ceil(len(num_cols)/4), 4, k)
    sns.distplot(x[i])
    k = k+1
plt.title('x[i]', fontsize=10)
plt.show()
# plot the distribution for all the numeric features

plt.figure(figsize=(50, 50))
k = 1
for i in num_cols:
    plt.subplot(math.ceil(len(num_cols)/4), 4, k)
    sns.boxplot(x[i])
    k = k+1
plt.title('x[i]', fontsize=10)
plt.show()
# text types histogram

plt.style.use('fivethirtyeight')
plt.figure(figsize=(50, 50))
k = 1
for i in cat_cols:
    plt.subplot(math.ceil(len(cat_cols)/4), 4, k)
    sns.countplot(x[i], palette='spring')
    k = k+1
plt.title('x[i]', fontsize=10)
plt.show()
cat_cols
cat_keys = np.linspace(1, len(cat_cols), len(cat_cols)).astype(int)
cat_keys
dict_cat = {cat_keys[idx]: cat_cols[idx]
            for idx in range(len(cat_cols))}  # for loop inside {}
dict_cat
# let's check the insured occupations

i = 6
var = dict_cat[i]
occu = pd.crosstab(x[var], df['fraud_reported'])
occu.div(occu.sum(1).astype(float), axis=0).plot(
    kind='bar', stacked=True, figsize=(15, 7))
plt.title('Fraud', fontsize=20)
plt.xticks(rotation=45)
plt.legend()
plt.show()
i = 5
j = 12
var1 = dict_cat[i]
var2 = dict_cat[j]

cat_bar1 = pd.crosstab(df[var1], df[var2])
colors = plt.cm.Blues(np.linspace(0, 1, 5))
cat_bar1.div(cat_bar1.sum(1).astype(float), axis=0).plot(kind='bar',
                                                         stacked=False,
                                                         figsize=(15, 7),
                                                         color=colors)
plt.title(var2, fontsize=20)
plt.legend()
plt.show()
i = 5
j = 12
var1 = dict_cat[i]
var2 = dict_cat[j]

cat_bar2 = pd.crosstab(df[var1], df[var2])
colors = plt.cm.inferno(np.linspace(0, 1, 5))
cat_bar2.div(cat_bar2.sum(1).astype(float), axis=0).plot(kind='bar',
                                                         stacked=True,
                                                         figsize=(15, 7),
                                                         color=colors)

plt.title(var2, fontsize=20)
plt.legend()
plt.show()
num_keys = np.linspace(1, len(num_cols), len(num_cols)).astype(int)
num_keys
dict_num = {num_keys[idx]: num_cols[idx]
            for idx in range(len(num_cols))}  # for loop inside {}
dict_num
# numeric variable distribution by fraud

j = 13
var_num = dict_num[j]

fig, axes = joypy.joyplot(df,
                          column=[var_num],
                          by='fraud_reported',
                          ylim='own',
                          figsize=(20, 10),
                          alpha=0.5,
                          legend=True)

plt.title(var_num, fontsize=20)
plt.show()
df.groupby('fraud_reported')['total_claim_amount'].describe()
# Pairwise correlation
i = 7
j = 13
var1 = dict_num[i]
var2 = dict_num[j]

# plotting a correlation scatter plot
fig1 = px.scatter_matrix(df, dimensions=[var1, var2], color="fraud_reported")
fig1.show()

# plotting a 3D scatter plot
fig2 = px.scatter(df, x=var1, y=var2, color='fraud_reported',
                  marginal_x='rug', marginal_y='histogram')
fig2.show()
# numeric variable distribution by categorical variable's level
i = 7
j = 13
var_cat = dict_cat[i]
var_num = dict_num[j]

fig, axes = joypy.joyplot(df,
                          column=[var_num],
                          by=var_cat,
                          ylim='own',
                          figsize=(20, 10),
                          alpha=0.5,
                          legend=True)

plt.title(var_num, fontsize=20)
plt.show()
i = 7
j = 13
var_cat = dict_cat[i]
var_num = dict_num[j]

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

sns.stripplot(df[var_cat], df[var_num], palette='bone')
plt.title(var_num, fontsize=20)
plt.show()
i = 7
j = 13
var_cat = dict_cat[i]
var_num = dict_num[j]

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

sns.boxenplot(df[var_cat], df[var_num], palette='pink')
plt.title(var_num, fontsize=20)
plt.show()
i = 13
j = 16
var_cat = dict_cat[i]
var_num = dict_num[j]

trace = go.Box(
    x=df[var_cat],
    y=df[var_num],
    opacity=0.7,
    marker=dict(
        color='rgb(215, 195, 5, 0.5)'
    )
)
data_boxplot = [trace]

layout = go.Layout(title=var_num)

fig = go.Figure(data=data_boxplot, layout=layout)
py.iplot(fig)
i = 1
j = 3
k = 9
l = 13
m = 14
n = 15
var_cat1 = dict_cat[i]
var_cat2 = dict_cat[j]
var_cat3 = dict_cat[k]
var_num1 = dict_num[l]
var_num2 = dict_num[m]
var_num3 = dict_num[n]

var1 = var_num1
var2 = var_num2
var3 = var_cat1

trace = go.Scatter3d(x=df[var1], y=df[var2], z=df[var3],
                     mode='markers',  marker=dict(size=10, color=df[var1]))

data_3d = [trace]

layout = go.Layout(
    title=' ',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene=dict(
        xaxis=dict(title=var_num1),
        yaxis=dict(title=var_num2),
        zaxis=dict(title=var_cat1)
    )

)
fig = go.Figure(data=data_3d, layout=layout)
py.iplot(fig)
df = df.sort_values(by=['auto_year', 'months_as_customer'])
# dynamic graphs along time line
i = 7
j = 13
var1 = dict_num[i]
var2 = dict_num[j]

figure = bubbleplot(dataset=df, x_column=var1, y_column=var2,
                    bubble_column='fraud_reported', time_column='auto_year', size_column='months_as_customer',
                    color_column='fraud_reported',
                    x_title=var1, y_title=var2,
                    x_logscale=False, scale_bubble=3, height=650)

py.iplot(figure, config={'scrollzoom': True})
# list all the types of the features/variables in the dataset
# there are 3 types of features in total: int64, object and float64, in which the object is text
df.dtypes
# check the data information|
df.info(verbose=True)
# there are some missing values in the dataset
# list the frequency of each level for the categorical variables
x_categori = pd.DataFrame(df, columns=cat_cols)
for col in x_categori.columns:
    col_count = x_categori[col].value_counts()
    print('The Frequency for', col)
    print(col_count)
    print('Total:', x_categori[col].count())
    print('Missing Values:', len(x_categori[col])-x_categori[col].count())
    print('\n')
# list the cross table between any two of the categorical variables
i = 5
j = 7
var1 = cat_cols[i]
var2 = cat_cols[j]
pd.crosstab(x[var1], x[var2])
# describe all the numeric variabl
df.describe()
# compute the median absolute deviation
x_numeric = pd.DataFrame(df, columns=num_cols)
x_numeric.apply(robust.mad).round(decimals=3)
# calculate the correlation on independent variables
x.corr()
sns.heatmap(x.corr(), annot=True)
i = 5
j = 15
cat_var = cat_cols[i]
num_var = num_cols[j]

print('distribution of', num_var, 'at', cat_var)
print('\n')
print(df.groupby(cat_var)[num_var].describe())
# Change setting to display all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Display column missingness
missing = 1 - x.count()/len(x.index)
missing.sort_values()
print('\n')
# check the collision_type distribution
count_collision_type = x['collision_type'].value_counts()
print(count_collision_type)
print('\n')
# check the property_damage distribution
count_property_damage = x['property_damage'].value_counts()
print(count_property_damage)
print('\n')
# check the police_report_available distribution
count_police_report_available = x['police_report_available'].value_counts()
print(count_police_report_available)

# we will replace the '?' by unknow since we do not know.
x['collision_type'].fillna('Unknown', inplace=True)

# It may be the case that there are no responses for property damage then we might take it as No property damage.
x['property_damage'].fillna('NO', inplace=True)

# again, if there are no responses fpr police report available then we might take it as No report available
x['police_report_available'].fillna('NO', inplace=True)

x.isnull().any().any()
# Display column missingness
x.isnull().sum()
# the result indicates there is no missing values
# now we need to consider dealing with the categorical variables
# we also drop auto_make since it is correlated to the auto_model, and auto_model is the subsegment of auto_make
# in addition, we drop the feature incident_date
# drop the location and the date also. they are not meaningful due to each location or date has almost only one incident.
# Thus no any date/location will cause more incident
x_keep = x.drop(['policy_bind_date', 'incident_date', 'incident_location',
                'auto_model', 'total_claim_amount', 'age'], axis=1)
# Retrieve all text variables from data
cat_cols1 = [c for i, c in enumerate(
    x_keep.columns) if x_keep.dtypes[i] in [np.object]]

# Retrieve all numerical variables from data
num_cols1 = [c for i, c in enumerate(
    x_keep.columns) if x_keep.dtypes[i] not in [np.object]]

# let's firstly see the feature self plots (distribution, box-plot, histogram)
cat_keys1 = np.linspace(1, len(cat_cols1), len(cat_cols1)).astype(int)
dict_cat1 = {cat_keys1[idx]: cat_cols1[idx]
             for idx in range(len(cat_cols1))}  # for loop inside {}
dict_cat1
num_keys1 = np.linspace(1, len(num_cols1), len(num_cols1)).astype(int)
dict_num1 = {num_keys1[idx]: num_cols1[idx]
             for idx in range(len(num_cols1))}  # for loop inside {}
dict_num1
i = 1

for i in range(len(num_cols1)):
    num_var = num_cols1[i]
    print('outlier detected for', num_var, 'the location is')
    grubbs.min_test_indices(x_keep[num_var], alpha=.05)
    grubbs.max_test_indices(x_keep[num_var], alpha=.05)
    print('outlier detected for ', num_var, ' the values is')
    grubbs.min_test_outliers(x_keep[num_var], alpha=.05)
    grubbs.max_test_outliers(x_keep[num_var], alpha=.05)
    print('\n')
    i = i+1

# the result indicates no outliers


def find_anomalies(var):
    # define a list to accumlate anomalies
    anomalies = []

    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(var)
    random_data_mean = np.mean(var)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    print('the lower limit is: ', lower_limit,
          'the upper limit is: ', upper_limit)
    # Generate outliers
    for outlier in var:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
#     return anomalies
    print(anomalies)


k = 1
for k in range(len(num_cols1)):
    var = num_cols1[k]
    print('outlier detected for', var)
    find_anomalies(x_keep[var])
    print('\n')
    k = k+1
x_keep.shape
# Encode categorical features into dataframe
x_dummy = pd.get_dummies(x_keep, columns=cat_cols1)
x_dummy.head(5)
x_dummy.info(verbose=True)
x_dummy.shape
x_check = x_dummy.loc[:, (x_dummy != 0).any(axis=0)]
x_check.shape
# standardize the features
sc = StandardScaler()
x_scaled = sc.fit_transform(x_dummy)
x_scaled_df = pd.DataFrame(x_scaled, columns=x_dummy.columns)
x_scaled_df.head(10)
x_scaled_df.describe()

vif_data = pd.DataFrame()
vif_data["feature"] = x_scaled_df.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_scaled_df.values, i)
                   for i in range(len(x_scaled_df.columns))]
vif_data.sort_values("VIF", ascending=False)
x_categori = pd.DataFrame(x_keep, columns=cat_cols1)
for col in x_categori.columns:
    col_count = x_categori[col].value_counts()
    print('The Frequency for', col)
    print(col_count)
    print('Total:', x_categori[col].count())
    print('Missing Values:', len(x_categori[col])-x_categori[col].count())
    print('\n')
pd.crosstab(df['fraud_reported'], df['insured_hobbies'])
x_final = x_scaled_df.drop(['insured_hobbies_camping', 'incident_type_Vehicle Theft',
                           'incident_severity_Trivial Damage', 'authorities_contacted_None', 'insured_occupation_adm-clerical', 'insured_education_level_Masters',
                            'insured_relationship_husband', 'collision_type_Unknown', 'incident_city_Northbrook', 'incident_state_PA', 'auto_make_Jeep'], axis=1)
# x_final=x_scaled_df
# test vif for features matrix

vif_data = pd.DataFrame()
vif_data["feature"] = x_final.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_final.values, i)
                   for i in range(len(x_final.columns))]
vif_data.sort_values("VIF", ascending=False)
x_final.shape
x_final.head(10)
# split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
X_train, X_test, y_train, y_test = train_test_split(
    x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.22222222222222224, random_state=321)


# pdy_train = pd.DataFrame(X_train)
# pdy_train.to_csv('X_train_JN.csv', index=False)
# pdpred_mlp_train1 = pd.DataFrame(y_train)
# pdpred_mlp_train1.to_csv('y_train_JN.csv', index=False)
# define a function to evaluate models

def evaluate_model(val_pred, val_probs, train_pred, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    # baseline, ROC=0.5 with equal chance to classify for any randome selected target account
    baseline = {}
    baseline['recall'] = recall_score(y_val, [1 for _ in range(len(y_val))])
    baseline['precision'] = precision_score(
        y_val, [1 for _ in range(len(y_val))])
    baseline['roc'] = 0.5

    # Calculate ROC on validation dataset
    val_results = {}
    val_results['recall'] = recall_score(y_val, val_pred)
    val_results['precision'] = precision_score(y_val, val_pred)
    val_results['roc'] = roc_auc_score(y_val, val_probs)

    # Calculate ROC on training dataset
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_pred)
    train_results['precision'] = precision_score(y_train, train_pred)
    train_results['roc'] = roc_auc_score(y_train, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Validation: {round(val_results[metric], 2)} Training: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_val, [1 for _ in range(len(y_val))])
    model_fpr, model_tpr, _ = roc_curve(y_val, val_probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')


# Fit Random Forest Classification to the Training set with default setting
rf_model = RandomForestClassifier(n_estimators=100, criterion="gini",
                                  random_state=50,
                                  max_features='sqrt',
                                  n_jobs=-1, verbose=1)
rf_model.fit(X_train, y_train)
# check the average of the nodes and depth.
n_nodes = []
max_depths = []
for ind_tree in rf_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')
# Test the model
pred_rf_val = rf_model.predict(X_val)
pred_rf_prob_val = rf_model.predict_proba(X_val)[:, 1]

pred_rf_train = rf_model.predict(X_train)
pred_rf_prob_train = rf_model.predict_proba(X_train)[:, 1]

# Get the model performance
print('classification report on training data')
print(classification_report(y_train, pred_rf_train))
print('\n')
print('classification report on validation data')
print(classification_report(y_val, pred_rf_val))
evaluate_model(y_val, pred_rf_prob_val, y_train, pred_rf_prob_train)

print(y_train.count())
# show the confusion matrix for training data
cnf_matrix = confusion_matrix(y_train, pred_rf_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# show the confusion matrix for validation data
print(y_val.count())
cnf_matrix = confusion_matrix(y_val, pred_rf_val, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_rf_prob_val).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
print('new_val')
print(new_val)
new_val.columns = ['fraud_reported', 'Probability']
print('new_val2')
print(new_val)
prob_to_1 = new_val.loc[new_val.fraud_reported ==
                        1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.fraud_reported ==
                        0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_rf_prob_val)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_rf_val)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_rf_prob_val)
abs_error_class = abs(y_val - pred_rf_val)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_rf_val)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_rf_val))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_rf_val))

print('\n')
print('Accuracy on classes:', accuracy_class)
print('\n')
print('KS test:', ks)
print('\n')
print('AUC Score on probability:', auc_prob)
print('AUC Score on classes:', auc_class)
print('\n')
print('GINI Score on probability:', gini_prob)
print('GINI Score on classes:', gini_class)
print('\n')
print('MAE on probability:', mae_prob)
print('MAE on classes:', mae_class)
