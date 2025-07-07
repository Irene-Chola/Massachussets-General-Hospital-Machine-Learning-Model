# Predicting Heart Disease Risk 
## Introduction 
The objective of this project is to develop a machine learning model to predict the risk of heart disease based on various health-related features. 

Predicting heart disease risk is crucial for early intervention and preventive measures, contributing to better healthcare outcomes and improved patient well-being. 

The dataset used for this analysis contains a variety of attributes such as age, sex, cholesterol levels, blood pressure, and other relevant health indicators. 

Here is the dataset used []()

### Data Preprocessing 
- Handling nulls and missing data 
By running a null check scan, we can find rows of data that possibly have null values. We can 
handle these rows of data by removing/dropping them. We replaced our nulls with mode for 
categorical values and mean for continuous values.

### Exploratory Data Analysis (EDA) 
Exploratory data analysis involves visualization of the relationship of various variables with the target variable. In the dataset the target variable is the tenYearCHD which is a binary value indicating the risk of coronary heart disease among the participants in a ten-year span.  

In the exploratory data analysis, the following variable relationships were visualized: 
- Sex
- Age
- Smoking status
- Blood pressure meds usage 


### Model Selection 
To split and test the data various machine learning algorithms are used. In this project the following algorithms were used: 
1. Decision tree
2. Logistic regression
3. K-Nearest Neighbors(KNN)
4. Support Vector Machine (SVM)
5. Random forest  

The models are chosen based on their interpretability, performance, and suitability for the task of predicting heart disease risk. 

### Model Training and Validation 

The dataset is split into training and testing sets to train and validate the selected models. 

The five models namely Decision tree, Logistic regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM) and Random Forest models are trained on the training set. The models are validated on the testing set and their performance is evaluated based on the accuracy score, recall, F1 score and ROC_AUC graph.  

## Conclusion 
The performance of each model is as follows:
### 1. Decision Tree Model
- Accuracy - 75%
- Classification report Precision - 0.86 
- Recall - 0.85
- F1 Score - 0.85

### 2. Logistic Regression Model
- Accuracy - 86%
- Classification report Precision - 0.86
- Recall - 0.97
- F1 Score - 0.91

### 3. The K Nearest Neighbor Model (KNN)
- Accuracy - 84%
- Classification report Precision - 0.86 
- Recall - 1.00
- F1 Score - 0.92
  
### 4. The Support Vector Machine Model (KVM)
- Accuracy - 85%
- Classification report Precision - 0.86 
- Recall - 1.00
- F1 Score - 0.92

### 5. The Random Forest Model
- Accuracy - 85%
- Classification report Precision - 0.86 
- Recall - 0.99
- F1 Score - 0.92

## Conclusion
Overall, the models demonstrate positive performance in predicting heart disease risk based on the selected features. The best model chosen for this machine learning task is **Logistic regression model** as it gives the highest accuracy score.
  
## Recommendations 
To further improve the predictive ability of the models the following can be considered: 

● Refining the models with additional feature engineering techniques or exploring alternative machine learning algorithms can be used to improve predictive performance. 

● Experts can be used to record more data points from the participants to increase the accuracy of the models.
