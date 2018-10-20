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

We can see in the plot of Bonus vs Salary that there is an oulier. This outlier is an error in the Total column and should be removed as it's likely due to a spreadsheet quirk. 

(IdentifyFraudFromEnronEmail/Bonus_Salary_Plot.PNG)

Once the outlier was removed we have a much better plot.

(IdentifyFraudFromEnronEmail/Bonus_Salary_Post_Plot.PNG)

