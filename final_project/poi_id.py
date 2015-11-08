#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from proj_helper import *

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
    # 'email_address',# this is string 
    'from_poi_to_this_person',
    'from_messages',
    'from_this_person_to_poi',
    # 'poi',
    'shared_receipt_with_poi',
    ]
poi_label = ['poi']

features_list = poi_label + financial_features + email_features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
outliers = [
    'TOTAL',
    'THE TRAVEL AGENCY IN THE PARK',
    'LOCKHART EUGENE E',
    ]
remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)
add_custum_features(data_dict, features_list)

k = 10
k_best_features, k_best_scores = get_k_best_features(data_dict, features_list, k)
print k_best_features,  k_best_scores


### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = poi_label + k_best_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
svm_clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel = 'rbf', C = 1000, gamma = 0.0001))
    ])

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = Pipeline(steps=[
            ('scaler', MinMaxScaler()),
            ('clf', RandomForestClassifier())
    ])

from sklearn.linear_model import LogisticRegression
lr_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                          random_state = 42))
])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(nb_clf, my_dataset, features_list)
test_classifier(dt_clf, my_dataset, features_list)
test_classifier(lr_clf, my_dataset, features_list)
# test_classifier(svm_clf, my_dataset, features_list)


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(nb_clf, my_dataset, features_list)