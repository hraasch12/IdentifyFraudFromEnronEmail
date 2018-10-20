# IdentifyFraudFromEnronEmail
#### By Heidi Raasch

**********************

## Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will play detective, and build a person of interest (POI) identifier based on the financial and email data made public as a result of Enron scandal. 

## Questions

**1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?**

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

** 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. 

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
| Naive Bayes | 0.852619047619 | 0.431829365079 | 0.375834054834 | 0.835714285714 | 0.373023809524 | 0.376445165945 |
| Decision Tree | 0.812857142857 | 0.29581998557 | 0.29488997114 | 0.812619047619 | 0.299841269841 | 0.310830808081 |
| Logistic Regression | 0.863333333333 | 0.507341269841 | 0.261941558442 | 0.860714285714 | 0.486965367965 | 0.258267316017 |
| SVM | 0.865714285714 | 0.154666666667 | 0.0600357142857 | 0.863571428571 | 0.0819285714286 | 0.0375952380952 |

We see that the performance is a little different between each of the algorithms. The only algorithm that had better performance with the new features was Decision Tree. Naive Bayes, Logistic Regression, and SVM performed better with the original features. 
