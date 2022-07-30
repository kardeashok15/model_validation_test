from numpy.random import rand
from numpy.random import seed
from scipy.stats import randint as sp_randint
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import os
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

BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.
user_name = "user1"
file_path = os.path.join(BASE_DIR, 'static\csv_files\\')

csv_file_name = "csvfile_"+user_name
savefile_x_final = file_path + csv_file_name + "_x_model.csv"
df = pd.read_csv(savefile_x_final)
targetVar = 'status'
# split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
y_model = df[targetVar]
x_model = df.drop(targetVar, axis=1)
# split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
X_train, X_test, y_train, y_test = train_test_split(
    x_model, y_model, test_size=0.1, random_state=321)  # Predictor and target variables
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.22222222222222224, random_state=321)

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
plt.show()
evaluate_model(y_val, pred_rf_val, y_train, pred_rf_train)
plt.show()


# show the confusion matrix for training data
cnf_matrix = confusion_matrix(y_train, pred_rf_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# show the confusion matrix for validation data
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
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
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


# Hyperparameter grid
param_grid = {'criterion': ['entropy', 'gini'],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'max_depth': list(np.linspace(10, 200, 50).astype(int)),
              'min_samples_leaf': list(np.linspace(2, 20, 10).astype(int)),
              'min_samples_split': [2, 5, 7, 10, 12, 15],
              'n_estimators': np.linspace(10, 200, 50).astype(int),
              'max_leaf_nodes': list(np.linspace(5, 100, 20).astype(int)),
              'bootstrap': [True, False]
              }

# Estimator for use in random search
model = RandomForestClassifier()

# Create the random search model
rf_search = RandomizedSearchCV(model, param_grid, n_jobs=3,
                               scoring='roc_auc', cv=5,
                               n_iter=100, verbose=1, random_state=50)

# Fit
rf_search.fit(X_train, y_train)

rf_search.best_params_

rf_random = rf_search.best_estimator_
rf_random

# Test the model
pred_rf_val0 = rf_random.predict(X_val)
pred_rf_prob_val0 = rf_random.predict_proba(X_val)[:, 1]

pred_rf_train0 = rf_random.predict(X_train)
pred_rf_prob_train0 = rf_random.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_rf_train0))
print(classification_report(y_val, pred_rf_val0))

evaluate_model(y_val, pred_rf_prob_val0, y_train, pred_rf_prob_train0)
plt.show()
evaluate_model(y_val, pred_rf_val0, y_train, pred_rf_train0)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_rf_train0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_rf_val0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_rf_prob_val0).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_rf_prob_val0)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_rf_val0)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_rf_prob_val0)
abs_error_class = abs(y_val - pred_rf_val0)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_rf_val0)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_rf_val0))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_rf_val0))

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

# Hyperparameter grid
param_grid = {'criterion': [rf_search.best_params_['criterion']],
              'max_features': [rf_search.best_params_['max_features']],
              'bootstrap': [rf_search.best_params_['bootstrap']],

              'max_depth': [rf_search.best_params_['max_depth']-5,
                            rf_search.best_params_['max_depth'],
                            rf_search.best_params_['max_depth']+5, ],

              'min_samples_leaf': [rf_search.best_params_['min_samples_leaf'] - 2,
                                   rf_search.best_params_['min_samples_leaf'],
                                   rf_search.best_params_['min_samples_leaf'] + 2],

              'min_samples_split': [rf_search.best_params_['min_samples_split'] - 2,
                                    rf_search.best_params_[
                                        'min_samples_split'],
                                    rf_search.best_params_['min_samples_split'] + 2],

              'n_estimators': [rf_search.best_params_['n_estimators'] - 5,
                               rf_search.best_params_['n_estimators'],
                               rf_search.best_params_['n_estimators'] + 5],

              'max_leaf_nodes': [rf_search.best_params_['max_leaf_nodes'] - 3,
                                 rf_search.best_params_['max_leaf_nodes'],
                                 rf_search.best_params_['max_leaf_nodes'] + 3]
              }


# Estimator for use in random search
model = RandomForestClassifier()

# Create the random search model
rf_grid = GridSearchCV(model, param_grid, n_jobs=3,
                       scoring='roc_auc', cv=5, verbose=1)

# Fit
rf_grid.fit(X_train, y_train)

rf_grid.best_params_

rf_grid = rf_grid.best_estimator_
rf_grid

# Test the model
pred_rf_val1 = rf_grid.predict(X_val)
pred_rf_prob_val1 = rf_grid.predict_proba(X_val)[:, 1]

pred_rf_train1 = rf_grid.predict(X_train)
pred_rf_prob_train1 = rf_grid.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_rf_train1))
print(classification_report(y_val, pred_rf_val1))

evaluate_model(y_val, pred_rf_prob_val1, y_train, pred_rf_prob_train1)
plt.show()
evaluate_model(y_val, pred_rf_val1, y_train, pred_rf_train1)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_rf_train1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_rf_val1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_rf_prob_val1).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_rf_prob_val1)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_rf_val1)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_rf_prob_val1)
abs_error_class = abs(y_val - pred_rf_val1)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_rf_val1)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_rf_val1))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_rf_val1))

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


mlp = MLPClassifier(hidden_layer_sizes=100, alpha=0.0001, random_state=1,
                    learning_rate_init=0.001, max_iter=200)

mlp.fit(X_train, y_train)

# Test the model
pred_mlp_val = mlp.predict(X_val)
pred_mlp_prob_val = mlp.predict_proba(X_val)[:, 1]

pred_mlp_train = mlp.predict(X_train)
pred_mlp_prob_train = mlp.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_mlp_train))
print(classification_report(y_val, pred_mlp_val))

evaluate_model(y_val, pred_mlp_prob_val, y_train, pred_mlp_prob_train)
plt.show()
evaluate_model(y_val, pred_mlp_val, y_train, pred_mlp_train)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_mlp_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_mlp_val, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_mlp_prob_val).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_mlp_prob_val)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_mlp_val)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_mlp_prob_val)
abs_error_class = abs(y_val - pred_mlp_val)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_mlp_val)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_mlp_val))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_mlp_val))

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

# Code for using Random Search to tune hyperparameters

# generate random floating point values
# seed random number generator
seed(1)


parameters = {
    'hidden_layer_sizes': [sp_randint.rvs(30, 100, 1), sp_randint.rvs(30, 100, 1), ],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': rand(10),
    'momentum': list(np.arange(0.1, 1, 0.1)),
    'learning_rate': ['constant', 'adaptive'],
}

# Estimator for use in random search
estimator = MLPClassifier(max_iter=100)

# Create the random search model
mlp_search = RandomizedSearchCV(estimator, parameters, n_jobs=3,
                                scoring='roc_auc', cv=5,
                                n_iter=100, verbose=1, random_state=50)

# Fit
mlp_search.fit(X_train, y_train)

mlp_search.best_params_

mlp_random = mlp_search.best_estimator_
mlp_random

# Test the model
pred_mlp_val0 = mlp_random.predict(X_val)
pred_mlp_prob_val0 = mlp_random.predict_proba(X_val)[:, 1]

pred_mlp_train0 = mlp_random.predict(X_train)
pred_mlp_prob_train0 = mlp_random.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_mlp_train0))
print(classification_report(y_val, pred_mlp_val0))

evaluate_model(y_val, pred_mlp_prob_val0, y_train, pred_mlp_prob_train0)
plt.show()
evaluate_model(y_val, pred_mlp_val0, y_train, pred_mlp_train0)
plt.show()


cnf_matrix = confusion_matrix(y_train, pred_mlp_train0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_mlp_val0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_mlp_prob_val0).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_mlp_prob_val0)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_mlp_val0)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_mlp_prob_val0)
abs_error_class = abs(y_val - pred_mlp_val0)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_mlp_val0)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_mlp_val0))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_mlp_val0))

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

# Hyperparameter grid
param_grid = {'activation': [mlp_search.best_params_['activation']],
              'solver': [mlp_search.best_params_['solver']],
              'learning_rate': [mlp_search.best_params_['learning_rate']],
              'hidden_layer_sizes': [mlp_search.best_params_['hidden_layer_sizes']],

              'alpha':            [mlp_search.best_params_['alpha']*0.8,
                                   mlp_search.best_params_['alpha'],
                                   mlp_search.best_params_['alpha']*1.2],

              'momentum':         [mlp_search.best_params_['momentum']*0.8,
                                   mlp_search.best_params_['momentum'],
                                   mlp_search.best_params_['momentum']*1.2],
              }

# Estimator for use in random search
estimator = MLPClassifier(max_iter=500)

# Create the random search model
mlp_grid = GridSearchCV(estimator, param_grid, n_jobs=3,
                        scoring='roc_auc', cv=5, verbose=1)

# Fit
mlp_grid.fit(X_train, y_train)

mlp_grid.best_params_

mlp_final = mlp_grid.best_estimator_
mlp_final

# Test the model
pred_mlp_val1 = mlp_final.predict(X_val)
pred_mlp_prob_val1 = mlp_final.predict_proba(X_val)[:, 1]

pred_mlp_train1 = mlp_final.predict(X_train)
pred_mlp_prob_train1 = mlp_final.predict_proba(X_train)[:, 1]


# Get the model performance
print(classification_report(y_train, pred_mlp_train1))
print(classification_report(y_val, pred_mlp_val1))

evaluate_model(y_val, pred_mlp_prob_val1, y_train, pred_mlp_prob_train1)
plt.show()
evaluate_model(y_val, pred_mlp_val1, y_train, pred_mlp_train1)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_mlp_train1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_mlp_val1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_mlp_prob_val1).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_mlp_prob_val1)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_mlp_val1)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_mlp_prob_val1)
abs_error_class = abs(y_val - pred_mlp_val1)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_mlp_val1)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_mlp_val1))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_mlp_val1))

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

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Test the model
pred_knn_val = knn.predict(X_val)
pred_knn_prob_val = knn.predict_proba(X_val)[:, 1]

pred_knn_train = knn.predict(X_train)
pred_knn_prob_train = knn.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_knn_train))
print(classification_report(y_val, pred_knn_val))

evaluate_model(y_val, pred_knn_prob_val, y_train, pred_knn_prob_train)
plt.show()
evaluate_model(y_val, pred_knn_val, y_train, pred_knn_train)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_knn_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_knn_val, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_knn_prob_val).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_knn_prob_val)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_knn_val)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_knn_prob_val)
abs_error_class = abs(y_val - pred_knn_val)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_knn_val)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_knn_val))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_knn_val))

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

parameters = {
    'n_neighbors': np.linspace(2, 10).astype(int),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# Estimator for use in random search
estimator = KNeighborsClassifier()

# Create the random search model
knn_search = RandomizedSearchCV(estimator, parameters, n_jobs=3,
                                scoring='roc_auc', cv=5,
                                n_iter=100, verbose=1, random_state=50)

# Fit
knn_search.fit(X_train, y_train)

knn_search.best_params_

knn_random = knn_search.best_estimator_
knn_random

Test the model
pred_knn_val0 = knn_random.predict(X_val)
pred_knn_prob_val0 = knn_random.predict_proba(X_val)[:, 1]

pred_knn_train0 = knn_random.predict(X_train)
pred_knn_prob_train0 = knn_random.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_knn_train0))
print(classification_report(y_val, pred_knn_val0))

evaluate_model(y_val, pred_knn_prob_val0, y_train, pred_knn_prob_train0)
plt.show()
evaluate_model(y_val, pred_knn_val0, y_train, pred_knn_train0)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_knn_train0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_knn_val0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_knn_prob_val0).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_knn_prob_val0)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_knn_val0)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_knn_prob_val0)
abs_error_class = abs(y_val - pred_knn_val0)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_knn_val0)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_knn_val0))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_knn_val0))

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

param_grid = {'n_neighbors': [knn_search.best_params_['n_neighbors'] - 2,
                              knn_search.best_params_['n_neighbors'],
                              knn_search.best_params_['n_neighbors'] + 2],
              'weights':     [knn_search.best_params_['weights']],
              'algorithm':   [knn_search.best_params_['algorithm']],
              'p':           [knn_search.best_params_['p']]
              }

# Estimator for use in random search
estimator = KNeighborsClassifier()

# Create the random search model
knn_grid = GridSearchCV(estimator, param_grid, n_jobs=3,
                        scoring='roc_auc', cv=5, verbose=1)

# Fit
knn_grid.fit(X_train, y_train)

knn_grid.best_params_

knn_final = knn_grid.best_estimator_
knn_final

# Test the model
pred_knn_val1 = knn_final.predict(X_val)
pred_knn_prob_val1 = knn_final.predict_proba(X_val)[:, 1]

pred_knn_train1 = knn_final.predict(X_train)
pred_knn_prob_train1 = knn_final.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_knn_train1))
print(classification_report(y_val, pred_knn_val1))

evaluate_model(y_val, pred_knn_prob_val1, y_train, pred_knn_prob_train1)
plt.show()
evaluate_model(y_val, pred_knn_val1, y_train, pred_knn_train1)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_knn_train1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_knn_val1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_knn_prob_val1).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_knn_prob_val1)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_knn_val1)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_knn_prob_val1)
abs_error_class = abs(y_val - pred_knn_val1)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_knn_val1)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_knn_val1))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_knn_val1))

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


svm_clf = svm.SVC(probability=True)
svm_clf.fit(X_train, y_train)

pred_svm_val = svm_clf.predict(X_val)
pred_svm_prob_val = svm_clf.predict_proba(X_val)[:, 1]

pred_svm_train = svm_clf.predict(X_train)
pred_svm_prob_train = svm_clf.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_svm_train))
print(classification_report(y_val, pred_svm_val))

evaluate_model(y_val, pred_svm_prob_val, y_train, pred_svm_prob_train)
plt.show()
evaluate_model(y_val, pred_svm_val, y_train, pred_svm_train)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_svm_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_svm_val, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_svm_prob_val).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_svm_prob_val)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_svm_val)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_svm_prob_val)
abs_error_class = abs(y_val - pred_svm_val)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_svm_val)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_svm_val))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_svm_val))

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

parameters = {
    'C':  list(np.arange(0.01, 1, 0.1)),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'probability': [True],
    'class_weight': [dict, 'balanced', None]
}

# Estimator for use in random search
estimator = svm.SVC()

# Create the random search model
svm_search = RandomizedSearchCV(estimator, parameters, n_jobs=3,
                                scoring='roc_auc', cv=5,
                                n_iter=100, verbose=1, random_state=50)

# Fit
svm_search.fit(X_train, y_train)

svm_search.best_params_

svm_random = svm_search.best_estimator_
svm_random

# Test the model
pred_svm_val0 = svm_random.predict(X_val)
pred_svm_prob_val0 = svm_random.predict_proba(X_val)[:, 1]

pred_svm_train0 = svm_random.predict(X_train)
pred_svm_prob_train0 = svm_random.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_svm_train0))
print(classification_report(y_val, pred_svm_val0))

evaluate_model(y_val, pred_svm_prob_val0, y_train, pred_svm_prob_train0)
plt.show()
evaluate_model(y_val, pred_svm_val0, y_train, pred_svm_train0)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_svm_train0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_svm_val0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_svm_prob_val0).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_svm_prob_val0)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_svm_val0)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_svm_prob_val0)
abs_error_class = abs(y_val - pred_svm_val0)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_svm_val0)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_svm_val0))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_svm_val0))

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

param_grid = {'C':  [svm_search.best_params_['C']*0.8,
                     svm_search.best_params_['C'],
                     svm_search.best_params_['C']*1.2],
              'kernel': [svm_search.best_params_['kernel']],
              'gamma': [svm_search.best_params_['gamma']],
              'probability': [svm_search.best_params_['probability']],
              'class_weight': [svm_search.best_params_['class_weight']]
              }

# Estimator for use in random search
estimator = svm.SVC()

# Create the random search model
svm_grid = GridSearchCV(estimator, param_grid, n_jobs=3,
                        scoring='roc_auc', cv=5, verbose=1)

# Fit
svm_grid.fit(X_train, y_train)

svm_grid.best_params_

svm_final = svm_grid.best_estimator_
svm_final

pred_svm_val1 = svm_final.predict(X_val)
pred_svm_prob_val1 = svm_final.predict_proba(X_val)[:, 1]

pred_svm_train1 = svm_final.predict(X_train)
pred_svm_prob_train1 = svm_final.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_svm_train1))
print(classification_report(y_val, pred_svm_val1))

evaluate_model(y_val, pred_svm_prob_val1, y_train, pred_svm_prob_train1)
plt.show()
evaluate_model(y_val, pred_svm_val1, y_train, pred_svm_train1)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_svm_train1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_svm_val1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_svm_prob_val1).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_svm_prob_val1)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_svm_val1)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_svm_prob_val1)
abs_error_class = abs(y_val - pred_svm_val1)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_svm_val1)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_svm_val1))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_svm_val1))

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

bc_model = BaggingClassifier()
bc_model.fit(X_train, y_train)

pred_bc_val = bc_model.predict(X_val)
pred_bc_prob_val = bc_model.predict_proba(X_val)[:, 1]

pred_bc_train = bc_model.predict(X_train)
pred_bc_prob_train = bc_model.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_bc_train))
print(classification_report(y_val, pred_bc_val))

evaluate_model(y_val, pred_bc_prob_val, y_train, pred_bc_prob_train)
plt.show()
evaluate_model(y_val, pred_bc_val, y_train, pred_bc_train)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_bc_train, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix - Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_bc_val, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix - Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_bc_prob_val).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_bc_prob_val)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_bc_val)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_bc_prob_val)
abs_error_class = abs(y_val - pred_bc_val)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_bc_val)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_bc_val))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_bc_val))

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

# Code for using Random Search to tune hyperparameters

parameters = {'base_estimator': [None, KNeighborsClassifier()],
              'n_estimators': list(np.linspace(10, 200, 50).astype(int)),
              'max_samples': list(np.linspace(1, 20, 20).astype(int)),
              'max_features': list(np.arange(0.5, 1, 0.1)),
              'bootstrap': [True, False],
              'bootstrap_features': [True, False],
              'oob_score': [True, False]
              }

# Estimator for use in random search
estimator = BaggingClassifier(random_state=50)

# Create the random search model
bc_search = RandomizedSearchCV(estimator, parameters, n_jobs=3,
                               scoring='roc_auc', cv=5,
                               n_iter=100, verbose=1, random_state=50)

# Fit
bc_search.fit(X_train, y_train)

bc_search.best_params_


bc_random = bc_search.best_estimator_
bc_random

# Test the model
pred_bc_val0 = bc_random.predict(X_val)
pred_bc_prob_val0 = bc_random.predict_proba(X_val)[:, 1]

pred_bc_train0 = bc_random.predict(X_train)
pred_bc_prob_train0 = bc_random.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_bc_train0))
print(classification_report(y_val, pred_bc_val0))

evaluate_model(y_val, pred_bc_prob_val0, y_train, pred_bc_prob_train0)
plt.show()
evaluate_model(y_val, pred_bc_val0, y_train, pred_bc_train0)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_bc_train0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_bc_val0, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_bc_prob_val0).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_bc_prob_val0)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_bc_val0)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_bc_prob_val0)
abs_error_class = abs(y_val - pred_bc_val0)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_bc_val0)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_bc_val0))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_bc_val0))

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

# Hyperparameter grid
param_grid = {'base_estimator': [bc_search.best_params_['base_estimator']],

              'n_estimators': [bc_search.best_params_['n_estimators'] - 10,
                               bc_search.best_params_['n_estimators'],
                               bc_search.best_params_['n_estimators'] + 10],

              'max_samples': [bc_search.best_params_['max_samples'] - 1,
                              bc_search.best_params_['max_samples'],
                              bc_search.best_params_['max_samples'] + 1],

              'max_features':  [bc_search.best_params_['max_features']*0.8,
                                bc_search.best_params_['max_features'],
                                bc_search.best_params_['max_features']*1.2],

              'bootstrap': [bc_search.best_params_['bootstrap']],
              'bootstrap_features': [bc_search.best_params_['bootstrap_features']],
              'oob_score': [bc_search.best_params_['oob_score']]
              }

# Estimator for use in random search
estimator = BaggingClassifier(random_state=50)

# Create the random search model
bc_grid = GridSearchCV(estimator, param_grid, n_jobs=3,
                       scoring='roc_auc', cv=5, verbose=1)

# Fit
bc_grid.fit(X_train, y_train)

bc_grid.best_params_

bc_final = bc_grid.best_estimator_
bc_final

pred_bc_val1 = bc_final.predict(X_val)
pred_bc_prob_val1 = bc_final.predict_proba(X_val)[:, 1]

pred_bc_train1 = bc_final.predict(X_train)
pred_bc_prob_train1 = bc_final.predict_proba(X_train)[:, 1]

# Get the model performance
print(classification_report(y_train, pred_bc_train1))
print(classification_report(y_val, pred_bc_val1))

evaluate_model(y_val, pred_bc_prob_val1, y_train, pred_bc_prob_train1)
plt.show()
evaluate_model(y_val, pred_bc_val1, y_train, pred_bc_train1)
plt.show()

cnf_matrix = confusion_matrix(y_train, pred_bc_train1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Training Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

cnf_matrix = confusion_matrix(y_val, pred_bc_val1, labels=[0, 1])
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix: Validation Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# perform the test on model performance

# perform ks test
y_val1 = pd.DataFrame(y_val).reset_index()
pred_prob_val1 = pd.DataFrame(pred_bc_prob_val1).reset_index()
new_val = pd.concat([y_val1, pred_prob_val1], axis=1).reset_index()
new_val = new_val.drop(['level_0', 'index'], axis=1)
new_val.columns = ['status', 'Probability']
prob_to_1 = new_val.loc[new_val.status == 1, ['Probability']].sort_index()
prob_to_0 = new_val.loc[new_val.status == 0, ['Probability']].sort_index()
prob_to_1 = np.array(prob_to_1).reshape(len(prob_to_1))
prob_to_0 = np.array(prob_to_0).reshape(len(prob_to_0))

ks = ks_2samp(prob_to_0, prob_to_1)


# perform auc and ginin test
fpr1, tpr1, thresholds = roc_curve(y_val,  pred_bc_prob_val1)
auc_prob = round(auc(fpr1, tpr1), 3)
gini_prob = 2*auc_prob-1

fpr2, tpr2, thresholds = roc_curve(y_val,  pred_bc_val1)
auc_class = round(auc(fpr2, tpr2), 3)
gini_class = 2*auc_class-1

# perform MAE test
abs_error_prob = abs(y_val - pred_bc_prob_val1)
abs_error_class = abs(y_val - pred_bc_val1)

mae_prob = round(np.mean(abs_error_prob), 4)
mae_class = round(np.mean(abs_error_class), 4)

# perform Accuracy test
accuracy_class = accuracy_score(y_val, pred_bc_val1)

print('\n')
print('Confusion Matrix - Validation:')
print(confusion_matrix(y_val, pred_bc_val1))

print('\n')
print('Classification Report - Validation:')
print(classification_report(y_val, pred_bc_val1))

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
