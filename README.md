# Customer Churn Predictions

## Introduction

This is a dataset of leading ecommerce company and we have analysis who are churn(leaving the company service) and have to make predicting churn model.

## About [Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
Data Variable Discerption
* E Comm CustomerID Unique customer ID
* E Comm Churn Churn Flag
* E Comm Tenure Tenure of customer in organization
* E Comm PreferredLoginDevice Preferred login device of customer
* E Comm CityTier City tier
* E Comm WarehouseToHome Distance in between warehouse to home of customer
* E Comm PreferredPaymentMode Preferred payment method of customer
* E Comm Gender Gender of customer
* E Comm HourSpendOnApp Number of hours spend on mobile application or website
* E Comm NumberOfDeviceRegistered Total number of deceives is registered on particular customer
* E Comm PreferedOrderCat Preferred order category of customer in last month
* E Comm SatisfactionScore Satisfactory score of customer on service
* E Comm MaritalStatus Marital status of customer
* E Comm NumberOfAddress Total number of added added on particular customer
* E Comm Complain Any complaint has been raised in last month
* E Comm OrderAmountHikeFromlastYear Percentage increases in order from last year
* E Comm CouponUsed Total number of coupon has been used in last month
* E Comm OrderCount Total number of orders has been places in last month
* E Comm DaySinceLastOrder Day Since last order by customer
* E Comm CashbackAmount Average cashback in last month

## Goal

Build a predictive model that can accurately identify customers who are at risk of leaving the company (churn) based on the provided variables. This can help the company take proactive steps to retain these customers and reduce the rate of churn.

Perform a thorough exploratory analysis of the provided customer data to gain insights into the behavior and characteristics of the customers. This includes analyzing patterns and trends in variables. This analysis can help the company understand its customers better and inform future decision-making.

## Machine Learning Models and Evaluation

This project implements and evaluates several machine learning models for classification tasks. Below are the details of the models used, the evaluation metrics, and the hyperparameter tuning process.

### Models Used
I have employed the following models for our classification tasks:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **XGBoost Classifier**

### Evaluation Metrics
To evaluate the performance of each model,I used the following metrics:

- **Accuracy Score**: Measures the ratio of correctly predicted instances to the total instances.
- **Precision Score**: The number of true positive predictions divided by the number of true positive and false positive predictions.
- **Recall Score**: The number of true positive predictions divided by the number of true positive and false negative predictions.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
- **Confusion Matrix**: A table used to describe the performance of a classification model by displaying the true positive, false positive, true negative, and false negative predictions.

### Hyperparameter Tuning
Hyperparameter tuning was conducted to optimize the performance of each model. This involved adjusting various parameters specific to each model to achieve the best possible accuracy and generalization.

#### Decision Tree
- **Hyperparameters Tuned**:
  - `max_depth`: The maximum depth of the tree
  - `min_samples_split`: The minimum number of samples required to split an internal node

#### Random Forest
- **Hyperparameters Tuned**:
  - `n_estimators`: The number of trees in the forest
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

#### SVM
- **Hyperparameters Tuned**:
  - `C`: Regularization parameter
  - `kernel`: Specifies the kernel type to be used in the algorithm
  - `gamma`: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'

#### XGBoost
- **Hyperparameters Tuned**:
  - `n_estimators`: Number of boosting rounds
  - `max_depth`: Maximum depth of a tree
  - `learning_rate`: Boosting learning rate

### [Results](notebook/Machine_learning.ipynb)

Model Performance: Tree-based models (Random Forest, XGBoost) demonstrated superior performance in predicting customer churn compared to Logistic Regression, Decision Tree and SVM, as evidenced by higher accuracy, precision, recall, and F1-scores.

Feature Importance: Tenure and complaint history emerged as the most critical factors influencing churn, while other variables like preferred order category, number of addresses, and marital status also played significant roles.

These findings highlight the importance of customer retention strategies focused on long-tenured customers and addressing customer complaints promptly.