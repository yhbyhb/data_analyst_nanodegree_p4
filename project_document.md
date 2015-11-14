Identify Fraud from Enron Email
========================================================
by HanByul Yang, November 13, 2015

## Overview ##
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

In this project, by leveraging `scikit-learn`,  I built a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal.

## Questions ##

#### Question 1 ####
**Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The goal of this project is to build a predictive model based on machine learning algorithm that can identify POI. The dataset contains 146 records with 21 features, 14 financial features, 6 email features and 1 poi label feature. There are 18 persons of interest (POI) and 128 non-POIs.

There are three outliers in the dataset. I removed them.

 - `TOTAL` : This is summary of rest of records.
 - `THE TRAVEL AGENCY IN THE PARK` : This record is not a person and also have no information.
 - `LOCKHART EUGENE E` : This gives no information. All feature are 'NaN'.

Following table shows valid values of each feature. `loan_advances` has significally low number of valid value.

Feature | # of valid values
---|---:
bonus | 82
deferral_payments | 39
deferred_income | 49
director_fees | 17
email_address | 111
exercised_stock_options | 102
expenses | 95
from_messages | 86
from_poi_to_this_person | 86
from_this_person_to_poi | 86
loan_advances | 4
long_term_incentive | 66
other | 93
poi | 146
restricted_stock | 110
restricted_stock_deferred | 18
salary | 95
shared_receipt_with_poi | 86
to_messages | 86
total_payments | 125
total_stock_value | 126

### Question 2 ###
**What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]**

Before choosing features, I listed up feature scores by using `SelectKBest` of `scikit-learn`.

Feature | Score
---|---:
exercised_stock_options | 24.815
total_stock_value | 24.183
bonus | 20.792
salary | 18.290
deferred_income | 11.458
long_term_incentive | 9.922
restricted_stock | 9.213
total_payments | 8.773
shared_receipt_with_poi | 8.589
loan_advances | 7.184
expenses | 6.094
from_poi_to_this_person | 5.243
other | 4.187
from_this_person_to_poi | 2.383
director_fees | 2.126
to_messages | 1.646
deferral_payments | 0.225
from_messages | 0.170
restricted_stock_deferred | 0.065

The table of feature score has a strong drop at 5th feature `deferred_income` and a small drop at 10th feature `loan_advances`. So I tuned number of features with final identifier and evaluation metrics.

number of features | precision | recall
---|---:|---:
3  | 0.519 | 0.234
4  | 0.437 | 0.242
5  | 0.502 | 0.345
6  | 0.514 | 0.410
7  | 0.502 | 0.423
8  | 0.491 | 0.429
9  | 0.434 | 0.420
10 | 0.401 | 0.370
11 | 0.370 | 0.381

By the result, I chose top 7 features: `exercised_stock_options`, `total_stock_value`, `bonus`, `salary`, `deferred_income`, `long_term_incentive`and `restricted_stock`.

I created two new features, `total_income` and `ratio_poi_email`. `total_income` is aggregation of all financial income. It is sum of `salary`, `bonus`, `exercised_stock_options` and `total_stock_value`. `ratio_poi_email` is the ratio of sum of `from_poi_to_this_person` and `from_this_person_to_poi` to total number of emails sent or received of each person. Since there is no email features among automatically selected features, `ratio_poi_email` would be a representative of email features. Thus, total 9(7+2) features are used for final analysis.

By adding new 2 features. It slightly incresed recall but slightly decreased precision.

number of features | precision | recall
---|---:|---:
7  | 0.502 | 0.423
7 + 2 new features | 0.482 | 0.448

There are several units in features. Financial features are described in USD. unit of email features is number of emails. Due to the difference of units, I used `StandardScaler` for fianl analysis before the training the classifiers.

I also tried `DecisionTreeClassifier` and its feature importances are below.

features | feature_importance
---|---:
shared_receipt_with_poi | 0.218
total_income | 0.207
restricted_stock | 0.175
ratio_poi_email | 0.140
total_stock_value | 0.136
exercised_stock_options | 0.044
long_term_incentive | 0.044
bonus | 0.035
salary | 0
deferred_income | 0
total_payments | 0
loan_advances | 0

`shared_receipt_with_poi` is the most important feature while `bonus`, `salary`, `deferred_income`, `total_payments` and `loan_advances` are 0 feature importance. By same method described above, Top 10 features and 2 new features are used for decision tree classifier.

### Question 3 ###
**What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**

I tried 3 algorhithms such as naive bayes, decision tree and [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) which is not covered in class. Logistic regression is generally used to model dichotomous outcome variables. For example, suppose that we are interested in the factors that influence whether a political candidate wins an election. The outcome (response) variable is binary (0/1);  win or lose. It is similar to our goal that classifying poi and non-poi.

The algorithim I end up using is logistic regression. Since it performs best among the algorithms that I tried. Surprisingly, the result of naive bayes was unexpectedly good. Its precision and recall are higher than our goal `0.3`.

### Question 4 ###
**What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]**

Tuning the parameters of an algorithm is a process to find optimal parameters for best performance. Without this process, I might get a lower performance than I expected. I tuned the parameters of decision tree and logistic regression by using `GridSearchCV` with following parameters with three scoring function such as precision, recall and f1. (In case of Naive Bayes, There is no parameter to optimize.)

 - `DecisionTreeClassifier`
    - criterion : The function to measure the quality of a split.
    - min_samples_split : The minimum number of samples required to split an internal node.
 - `LogisticRegression`
    - C : Inverse of regularization strength.
    - penalty : Used to specify the norm used in the penalization.

 After finding best scores for each score function, scoring function `recall` performed best in evalation metrics discussed on question 6.

 Followings are tuned parameters for each algorithm:

- `DecisionTreeClassifier` : criterion = 'gini', min_samples_split = 2
- `LogisticRegression` : C = 1e-12, penaly = 'l2'


### Question 5 ###
**What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**
Validation is the process to ensure that classifiers works robustly with given parameters. The classic mistake is over-fitting. If over-fitted, the machine learning model works well with training dataset and performs poorly on test dataset. To avoid overfitting, I held out 20 % of dataset for test set and put 80 % into training set. Since dataset is small and labels are skewed towards non-POI (18 POI and 125 non-POI after removing outliers),  I used stratified method `StratifiedShuffleSplit` to achive robust results.

 I took averages on 1000 trials of precision and recall. I also evaluated my classifier by validation method `test_classifier()` in `tester.py` .

### Question 6 ###
**Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

Precision and recall are used for evaluation. Precision is ratio of number of true positive to all positive prediction. High precision means low chance of false alarm. Recall shows the ratio of number of true positive to actual POIs. To identify POI as many as possible, increasing recall is needed.

I found similar results in both vaildation methods described in question 5. The results are as follows:

My validation (test_size=0.2, num_iter=1,000)

Classifier | Precision | Recall
---|---:|---:
Naive Bayes | 0.500 | 0.395
Decision Tree | 0.278 | 0.268
**Logistic Regression** | **0.482** | **0.448**
LR (class_weight='balanced') | 0.289 | 0.650

Provided validation (`test_classifier` in `tester.py`)

Classifier | Precision | Recall
---|---:|---:
Naive Bayes | 0.481 | 0.393
Decision Tree | 0.294 | 0.291
**Logistic Regression** | **0.463** | **0.444**
LR (class_weight='balanced') | 0.279 | 0.641

As I mentioned above, Navie Bayes performed unexpectedly good and logistic regression shows best scores on both evaluation metrics. Logistic regression with balanced class weight performed the highest recall score.

## Conclusion ##
I chose logistic regression classifiers as final identifier. Since it performed the most balanced and highest results in both validations. For optimzing recall, logistic regression classifier with balanced class weight would be the best identifier.

## References ##
- [scikit-learn](http://scikit-learn.org/stable/documentation.html)
- [course resources](https://www.udacity.com/course/intro-to-machine-learning--ud120)

## Contents ##
- `final_project/`
    - `my_dataset.pkl`, `my_classifier.pkl`, `my_feature_list.pkl` : pickle files of my work
    - `final_project_dataset.pkl` : The dataset for the project
    - `helper.py` : helper function for identifying POI, used in `poi_id.py`
    - `poi_id.py` : script for project result
    - `tester.py` : provided file for validation and generation of result pickle files
- `tools/`
    - `feature_format.py` : provided file for formatting the dataset.
- `project_document.html` : documentation of my work.
- `project_document.md` : markdown file of project_document.html