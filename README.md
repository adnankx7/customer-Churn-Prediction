# Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn based on various features such as demographics, account information, and service usage. The goal is to identify customers who are likely to leave (churn) so that businesses can take proactive measures to retain them. We use different machine learning models to classify customers as either likely to churn or not. The dataset used contains customer details and whether they exited the service or remained active.

## Dataset

The dataset consists of the following features:

- **Age**: The customer's age.
- **Balance**: The account balance of the customer.
- **CreditScore**: The credit score of the customer.
- **Geography**: The geographic region of the customer (e.g., France, Germany, Spain).
- **Gender**: The gender of the customer.
- **Exited**: Whether the customer has churned (1) or not (0).
- **HasCrCard**: Whether the customer has a credit card (1) or not (0).
- **IsActiveMember**: Whether the customer is an active member (1) or not (0).
- **NumOfProducts**: The number of products the customer has with the bank.
- **Tenure**: The number of years the customer has been with the bank.

## Project Goals

The main goal of this project is to:

1. **Preprocess the Data**: Handle missing values, duplicate data, and irrelevant columns. Transform categorical variables and scale numerical features.
2. **Perform Exploratory Data Analysis (EDA)**: Analyze the dataset to understand the distribution of features, detect relationships between features, and visualize the data.
3. **Build and Evaluate Machine Learning Models**: Train several classification models and evaluate them based on various performance metrics like accuracy, precision, recall, F1 score, and ROC AUC.
4. **Model Comparison**: Compare multiple models (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Decision Tree) to identify the best performer for customer churn prediction.
5. **Conclusion**: Draw conclusions based on the performance metrics and recommend the most effective model.

## Key Models Used

The following machine learning models were used for the prediction task:

1. **Logistic Regression**: A statistical model that is used for binary classification.
2. **Decision Tree**: A non-linear model that splits the data based on feature values to classify customers.
3. **Random Forest**: An ensemble of decision trees, which reduces overfitting and improves accuracy.
4. **Gradient Boosting**: A boosting algorithm that builds models sequentially, where each model corrects the errors of the previous one.
5. **AdaBoost**: A boosting algorithm that combines multiple weak learners to create a strong model.

## Data Preprocessing

Data preprocessing involves cleaning and transforming the raw dataset into a form suitable for machine learning algorithms. The preprocessing steps in this project include:

- **Handling Missing Values**: Any missing data points were identified and appropriately handled (either by filling missing values or dropping rows).
- **Dropping Irrelevant Columns**: Columns like `RowNumber`, `CustomerId`, `Surname`, and `Gender` were removed as they are not useful for prediction.
- **Feature Scaling**: Numerical features were standardized to ensure models are not biased by the magnitude of feature values.
- **One-Hot Encoding**: Categorical variables were encoded into numerical values using one-hot encoding for better compatibility with machine learning models.

## Exploratory Data Analysis (EDA)

EDA is an essential part of the project to understand the distribution of the data and detect patterns. Key visualizations included:

- **Histograms**: To show the distribution of numerical features like `Age` and `Balance`.
- **Count Plots**: To display the frequency distribution of categorical variables such as `Geography` and `Exited`.
- **Heatmaps**: To identify correlations between numerical features and help with feature selection.
- **Pair Plots**: To show relationships between multiple variables, especially with respect to the target variable `Exited`.
- **Box Plots**: To check for outliers and visualize the spread of values for features like `Balance` against the `Exited` label.

## Model Training and Evaluation

Each model was trained on the preprocessed data, and its performance was evaluated using the following metrics:

- **Accuracy**: The ratio of correct predictions to total predictions.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, used to balance the two metrics.
- **ROC AUC Score**: A performance measurement for classification problems that tells us how well the model distinguishes between classes.

The models were trained on the training set and evaluated on the test set to ensure their generalizability.

## Best Model Performance

After training and evaluating the models, the **Random Forest** and **Logistic Regression** models were found to be the best performers. Both models achieved identical results, with high scores across all metrics:

- **Accuracy**: 99.90%
- **F1 Score**: 99.90%
- **Precision**: 99.75%
- **Recall**: 99.75%
- **ROC AUC Score**: 99.84%

These results demonstrate that both models are highly effective for predicting customer churn and can be deployed in a real-world business setting.

## Conclusion

Both **Random Forest** and **Logistic Regression** demonstrated excellent performance, with identical results across all metrics. These models provide a reliable way to predict customer churn, and businesses can use them to target at-risk customers and improve retention strategies.

This project demonstrates the importance of data preprocessing, feature engineering, and model evaluation to develop a robust machine learning model for customer churn prediction.
