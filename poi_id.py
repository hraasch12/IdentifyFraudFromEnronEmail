#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from numpy import mean
from sklearn.cross_validation import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']
features_list = poi_label + email_features + financial_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
	
# Number of data points
print("Number of data points: %i" %len(data_dict))

# POI vs non-POI
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Number of poi: %i" % poi)
print("Number of non-poi: %i" % (len(data_dict) - poi))
       
# Number of features used
all_features = data_dict[data_dict.keys()[0]].keys()
print("Features per person: %i" %(len(all_features)))
print("Features used: %i" %(len(features_list)))

# Missing Values?
missing_values = {}
for f in all_features:
    missing_values[f] = 0
for person in data_dict:
    for f in data_dict[person]:
        if data_dict[person][f] == "NaN":
            missing_values[f] += 1
print("Missing values per feature: ")
for f in missing_values:
    print("%s: %i" %(f, missing_values[f]))

    
### Task 2: Remove outliers

def displayPlot(data_set, feature_x, feature_y):
    """
    This function displays a 2d plot of 2 features when given 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
	
# Visualize data to identify outliers

#print(displayPlot(data_dict, 'total_payments', 'total_stock_value'))
#print(displayPlot(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(displayPlot(data_dict, 'salary', 'bonus'))
#print(displayPlot(data_dict, 'total_payments', 'other'))
identity = []

for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print("Outliers:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# Find persons whose financial features are all "NaN"
fi_nan_dict = {}
for person in data_dict:
    fi_nan_dict[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            fi_nan_dict[person] += 1
sorted(fi_nan_dict.items(), key=lambda x: x[1])

# Find persons whose email features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

print(displayPlot(data_dict, 'salary', 'bonus'))

### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict

# create new features
def calcFraction(poi_messages, all_messages):
    """ 
	Returns a fraction of messages to/from a person 
	that are to/from a POI
    """
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)

    return fraction

def sortSecond(elem):
    """ 
	Sort by second element
    """
    return elem[1]

for emp in my_dataset:
    from_poi_to_this_person = my_dataset[emp]['from_poi_to_this_person']
    to_messages = my_dataset[emp]['to_messages']
    frac_from_poi = calcFraction(from_poi_to_this_person, to_messages)
    # print frac_from_poi
    my_dataset[emp]['frac_from_poi'] = frac_from_poi

    from_this_person_to_poi = my_dataset[emp]['from_this_person_to_poi']
    from_messages = my_dataset[emp]['from_messages']
    frac_to_poi = calcFraction(from_this_person_to_poi, from_messages)
    my_dataset[emp]['frac_to_poi'] = frac_to_poi

features_list_n = features_list
features_list_n =  features_list_n + ['frac_from_poi', 'frac_to_poi']
print 'New features list: ', features_list_n

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Removes all features whose variance is below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Remove all but the k highest scoring features
from sklearn.feature_selection import f_classif
k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best Features:")
scores = zip(features_list_n[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['frac_from_poi', 'frac_to_poi'], \
                     sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def eval_clf(grid_search, features, labels, params, reps=100):
    acc = []
    prec = []
    recall = []
    for i in range(reps):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        prec = prec + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "Accuracy: {}".format(mean(acc))
    print "Precision: {}".format(mean(prec))
    print "Recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))

from sklearn import naive_bayes        
nb_clf = naive_bayes.GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param)

print("Evaluate naive bayes model")
eval_clf(nb_grid_search, features, labels, nb_param)
#Accuracy: 0.854761904762
#Precision: 0.432977633478
#Recall:    0.373191558442

print("Evaluate Naive Bayes Model with new features:")
eval_clf(nb_grid_search, new_features, new_labels, nb_param)
#Accuracy: 0.842619047619
#Precision: 0.395617965368
#Recall:    0.37384992785

#from sklearn import tree
#dt_clf = tree.DecisionTreeClassifier()
#dt_param = {'criterion':('gini', 'entropy'),
#'splitter':('best','random')}
#dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)
#print("Evaluate Decision Tree Model:")
#eval_clf(dt_grid_search, features, labels, dt_param)
#Accuracy: 0.802380952381
#Precision: 0.246413863914
#Recall:    0.270544372294


#print("Evaluate Decision Tree Model with new features:")
#eval_clf(dt_grid_search, new_features, new_labels, dt_param)
#Accuracy: 0.812857142857
#Precision: 0.290127317127
#Recall:    0.284605339105


#from sklearn import linear_model
#from sklearn.pipeline import Pipeline
#lo_clf = Pipeline(steps=[
#        ('scaler', preprocessing.StandardScaler()),
#        ('classifier', linear_model.LogisticRegression())])
         
#lo_param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], \
#            'classifier__C': [0.1, 0.01, 0.001, 0.0001]}
#lo_grid_search = GridSearchCV(lo_clf, lo_param)
#print("Evaluate Logistic Regression Model:")
#eval_clf(lo_.02_search, features, labels, lo_param)
#classifier__tol = 1, 
#classifier__C = 0.1, 
#Accuracy: 0.859761904762
#Precision: 0.400333333333
#Recall:    0.190000721501

#print("Evaluate Logistic Regression Model with new features:")
#eval_clf(lo_grid_search, new_features, new_labels, lo_param)
#Accuracy: 0.859285714286
#Precision: 0.466736263736
#Recall:    0.244305916306

#from sklearn import svm
#s_clf = svm.SVC()
#s_param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
#           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}    
#s_grid_search = GridSearchCV(s_clf, s_param)
#print("Evaluate SVM Model")
#eval_clf(s_grid_search, features, labels, s_param)
#kernel = 'linear'
#C = 1 
#gamma = 1

#Accuracy: 0.866428571429
#Precision: 0.141666666667
#Recall:    0.0384523809524

#print("Evaluate SVM Model using new features")
#eval_clf(s_grid_search, new_features, new_labels, s_param)

#Accuracy: 0.86619047619
#Precision: 0.0938333333333
#Recall:    0.0442738095238

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
final_features_list = optimized_features_list 

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_features_list)