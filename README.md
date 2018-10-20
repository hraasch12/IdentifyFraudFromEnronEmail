# IdentifyFraudFromEnronEmail
#### By Heidi Raasch

**********************

## Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will play detective, and build a person of interest (POI) identifier based on the financial and email data made public as a result of Enron scandal. 

## Questions

#### 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?**

The goal for this project is to use machine learning tools to identify Persons of Interest (POI) from the Enron financial and email dataset. 

### Dataset Background:

* Total number of data points: 146
* Total number of poi: 18
* Total number of non-poi: 128

There are 21 features available for each person.

Some features have missing values. 

| Feature | # of Missing Values |
| :---- | :---: |
| Salary  | 51|
| To Messages | 60 |
| Deferral Payments | 107 |
| Total Ppayments | 21 |
| Loan Advances | 142 |
| Bonus | 64 |
| Email Address | 35 |
| Restricted Stock Deferred | 128 |
| Total Stock Value | 20 |
| Shared Receipt With POI | 60 |
| Long Term Incentive | 80 |
| Exercised Stock Options | 44 |
| From Messages | 60 |
| Other | 53 |
| From POI To This Person | 60 |
| From This Person To POI | 60 |
| POI | 0 |
| Deferred Income | 97 |
| Expenses | 51 |
| Restricted Stock | 36 |
| Director Fees | 129 |

## Outliers

We can see in the plot of Bonus vs Salary that there is an oulier. This outlier is representing Total and it should be removed to avoid having it impact our prediction models. 

I also noticed that there was an agency included in the dataset. Since "THE TRAVEL AGENCY IN THE PARK" is not a person I removed it from the dataset. 

There was one last record that needed to be removed and that was for "LOCKHART EUGENE E". This record was removed because all of its values were NaN.

![Original Plot](https://github.com/hraasch12/IdentifyFraudFromEnronEmail/blob/master/Bonus_Salary_Plot.PNG)

Once the outliers were removed we have a much better plot.

![Plot Without Outlier](https://github.com/hraasch12/IdentifyFraudFromEnronEmail/blob/master/Bonus_Salary_Post_Plot.PNG)

#### 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. 

Two new features were created and added to the original features_list. The new features were frac_from_poi and frac_to_poi. frac_from_poi is the ratio of the messages that were from the POI to this person verses all the messages sent to this person. The frac_to_poi is the ratio of emails from this person to the POI versus all messages sent from this person. The theory behind these two features was that a POI is more likely to send/recieve more emails to/from other POIs than non-POIs. 

When evaluating the featres I used SelectKBest to select the most powerful features. The scores for each of the features were:

| Feature | Score |
| :---- | :--- |
| exercised_stock_options | 24.815079733218194 |
| total_stock_value | 24.18289867856688 |
| bonus | 20.792252047181535 |
| salary | 18.289684043404513 |
| frac_to_poi | 16.40971254803579|
| deferred_income | 11.458476579280369 |
| long_term_incentive | 9.922186013189823 |
| restricted_stock | 9.2128106219771 |
| total_payments | 8.772777730091676 |
| shared_receipt_with_poi | 8.589420731682381 |
| loan_advances | 7.184055658288725 |
| expenses | 6.094173310638945 |
| from_poi_to_this_person | 5.243449713374958 |
| other | 4.187477506995375 | 
| frac_from_poi | 3.128091748156719 |
| from_this_person_to_poi | 2.382612108227674 |
| director_fees | 2.1263278020077054 | 
| to_messages | 1.6463411294420076 |
| deferral_payments | 0.2246112747360099 |
| from_messages | 0.16970094762175533 |
| restricted_stock_deferred | 0.06549965290994214 |

I chose to only use the top 7 features based on their score above. Then I used min-max scalers to rescale each feature to a common range since the range of raw values veried so widely. 

Next I examined the effect of my new features by looking at the performance of several algorithms using hte original features and the new features.

| Algorithm | Accuracy Original | Precision Original | Recall Original | Accuracy New | Precision New | Recall New | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Bayes | 0.854761904762 | 0.432977633478 | 0.373191558442 | 0.842619047619 | 0.395617965368 | 0.37384992785 |
| Decision Tree | 0.802380952381 | 0.246413863914 | 0.270544372294 | 0.812857142857 | 0.290127317127 | 0.284605339105 |
| Logistic Regression | 0.859761904762 | 0.400333333333 | 0.190000721501 | 0.859285714286 | 0.466736263736 | 0.244305916306 |
| SVM | 0.866428571429 | 0.141666666667 | 0.0384523809524 | 0.86619047619 | 0.0938333333333 | 0.0442738095238 |

We see that the performance is a little different between each of the algorithms. The only algorithm that had better performance with the new features was Decision Tree. Naive Bayes, Logistic Regression, and SVM performed better with the original features. 

#### 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  

I used 4 different algorithms and in the end used Naive Bayes as it had the best performance. The other algorithms I evaluated were Decision Tree, Logistic Regression, and SVM. 

The Logistic Regression model and SVM model took significantly longer to run than the other models. 

#### 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). 

Typically the parameters used in algorithms need to be optimized to get the algorithm to perform at its best. This process is called Tuning. This is one of the final steps and if not done properly it can lead to data being improperly fitted. 

  For tuning the parameters I used GridSearchCV from sklearn. GridSearch methodically builds and evaluates a model for each combination of parameters specified in a grid. 

The results of the tuning were:
- Niave Bayes doesn't take any parameters so there was no tuning needed.
- SVM: kernel = 'rbf', C = 1000, and gamma = 0.01
- Decision Tree: 
    Without new features: splitter = 'random', criterion = 'gini'
    With new features: splitter = 'best', criterion = 'entropy'
- Logistic Regression: C = 0.1, tol = 1

#### 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? 

Validation is a way to assess how a trained model will generalize with a test dataset. The classic mistake is not splitting data into testing and training datasets. Neglecting to do this leads to overfitting. When a model is overfit it performs well on the training data but does not do well with a new dataset. To prevent this from happening with my analysis I created the eval_clf function which uses cross validation to split the data into test and train sets as well as calculate the accuracy, precision, and recall of each iteration and used the mean of each metric. 

#### 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

I evaluated the accuracy, precision, and recall of the Niave Bayes algorithm. The average accuracy was 0.851, precision was 0.445, and recall was 0.3785. 

Accuracy is used to demonstrate how close a measured value is to the actual value. The accuracy value of 0.851 means that the proportion of true results is 0.851 among all cases. 

Precision measures the algorithm's ability to classify the true positives from all examined positives. A precision of 0.445 means that 43 out of the 100 people classified as POIs were actually POIs. 

Recall measures the algorithm's ability to classify the true positives over all actual positives. Therefore, a recall of 0.3785 means that out of the 100 true POIs in the dataset roughly 38 are correctly classified as POIs. 
