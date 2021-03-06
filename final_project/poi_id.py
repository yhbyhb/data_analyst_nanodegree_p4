#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from helper import *

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = [
    'salary',
    'deferral_payments',
    'total_payments',
    'loan_advances',
    'bonus',
    'restricted_stock_deferred',
    'deferred_income',
    'total_stock_value',
    'expenses',
    'exercised_stock_options',
    'other',
    'long_term_incentive',
    'restricted_stock',
    'director_fees',
    ]
email_features = [
    'to_messages',
    'from_poi_to_this_person',
    'from_messages',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    ]
poi_label = ['poi']

features_list = poi_label + financial_features + email_features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Count valid values
valid_counts = count_valid_values(data_dict)

### Task 2: Remove outliers
outliers = [
    'TOTAL',
    'THE TRAVEL AGENCY IN THE PARK',
    'LOCKHART EUGENE E',
    ]
remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)
k=7
k_best_features, k_best_scores = get_k_best_features(data_dict, features_list, k)
# print k_best_features,  k_best_scores
my_features = poi_label + k_best_features
add_custum_features(data_dict, my_features)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

nb_pipeline = Pipeline(steps=[('clf', GaussianNB())])
dt_pipeline = Pipeline(steps=[('clf', DecisionTreeClassifier(random_state=42))])
lr_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(tol=0.001, random_state=42))
])
lr_b_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', tol=0.001, random_state=42))
])
pipeline_param_grid = {
    nb_pipeline : {},
    # dt_pipeline :
    # {
    #     'clf__criterion': ['gini', 'entropy'],
    #     'clf__min_samples_split': range(2, 5),
    # },
    lr_pipeline :
    {
        'clf__C' : 10.0 ** np.arange(-12, 15, 3),
        'clf__penalty' : ('l1', 'l2')
    },
    lr_b_pipeline :
    {
        'clf__C' : 10.0 ** np.arange(-12, 15, 3),
        'clf__penalty' : ('l1', 'l2')
    },
}

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("Finding best parameters...")
test_size = 0.2
score_func = 'recall'
for pipeline, params in pipeline_param_grid.iteritems():
    gridcv_clf = find_best_parameters(pipeline, params, score_func, my_dataset,
                                      my_features, test_size=test_size)

    print(gridcv_clf)
    # for params, mean_score, scores in gridcv_clf.grid_scores_:
    #     print("{:0.3f} for {}".format(mean_score, params))
    # print("Best {} score: {:0.3f}".format(score_func, gridcv_clf.best_score_))
    print("Best parameters set:")
    best_parameters = gridcv_clf.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t{}: {}".format(param_name, best_parameters[param_name]))

    print('')

nb_clf = GaussianNB()
dt_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=2,
                                random_state=42)
lr_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1e-12, penalty='l2',
                                tol=0.001, random_state = 42))
])
lr_b_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1e-12, penalty='l2', class_weight='balanced',
                                 tol=0.001, random_state = 42))
])

print("\nMy validation")
def printValidation(clf, dataset, features, test_size):
    precision, recall = validation(clf, dataset, features, test_size = test_size)
    print(clf)
    print("precision : {}, recall : {}".format(precision, recall))
    print('')

printValidation(nb_clf, my_dataset, my_features, test_size)
# printValidation(dt_clf, my_dataset, my_features, test_size)
printValidation(lr_clf, my_dataset, my_features, test_size)
printValidation(lr_b_clf, my_dataset, my_features, test_size)

print("\nProvided validation")
test_classifier(nb_clf, my_dataset, my_features)
# test_classifier(dt_clf, my_dataset, my_features)
test_classifier(lr_clf, my_dataset, my_features)
test_classifier(lr_b_clf, my_dataset, my_features)

### Dump your classifier, dataset, and my_features so 
### anyone can run/check your results.
clf = lr_clf
dump_classifier_and_data(clf, my_dataset, my_features)