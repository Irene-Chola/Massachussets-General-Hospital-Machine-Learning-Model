# Predicting Heart Disease Risk 
## Introduction 
The objective of this project is to develop a machine learning model to predict the risk of heart disease based on various health-related features. 

Predicting heart disease risk is crucial for early intervention and preventive measures, contributing to better healthcare outcomes and improved patient well-being. 

The dataset used for this analysis contains a variety of attributes such as age, sex, cholesterol levels, blood pressure, and other relevant health indicators. 

Here is the dataset used [MGH Prediction Dataset](https://github.com/Irene-Chola/Massachussets-General-Hospital-Machine-Learning-Model/blob/main/MGH_Prediction_DataSet.csv)

## Table of Contents
[Data Processing](data-processing)

[Exploratory Data Analysis](exploratory-data-analysis)

[Model Training and Validation ](model-training-and-validation)

[Model Performance](model-performance)

[Conclusion](Conclusion)

[Recommendations ](recommendations)


### Data Preprocessing 
Handling nulls and missing data - By running a null check scan, we can find rows of data that possibly have null values. We can handle these rows of data by removing/dropping them. We replaced our nulls with mode for categorical values and mean for continuous values.

```python
# Replace the nulls
# Categorical Data - Replace with Mode
dataset['education'].fillna(dataset['education'].mode()[0], inplace = True)
dataset['BPMeds'].fillna(dataset['BPMeds'].mode()[0], inplace = True)

# Continous Data - Replace with Mean
dataset['cigsPerDay'].fillna(dataset['cigsPerDay'].mean(), inplace = True)
dataset['totChol'].fillna(dataset['totChol'].mean(), inplace = True)
dataset['BMI'].fillna(dataset['BMI'].mean(), inplace = True)
dataset['heartRate'].fillna(dataset['heartRate'].mean(), inplace = True)
dataset['glucose'].fillna(dataset['glucose'].mean(), inplace = True

```

## Exploratory Data Analysis
Exploratory data analysis involves visualization of the relationship of various variables with the target variable. In the dataset the target variable is the tenYearCHD which is a binary value indicating the risk of coronary heart disease among the participants in a ten-year span.  

In the exploratory data analysis, the following variable relationships were visualized: 
### 1. Age
### 2. Gender
  - Distribution of TenYearCHD for each Gender
```
import matplotlib.pyplot as plt

# Counting occurrences of TenYearCHD for each sex
chd_by_sex = dataset.groupby(['sex', 'TenYearCHD']).size().unstack()

# Plotting the grouped bar chart
plt.figure(figsize=(8, 6))
ax = chd_by_sex.plot(kind='bar', stacked=False, color=['navy', 'red'])
plt.title('Distribution of TenYearCHD by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], rotation=0)  # Replace 0 and 1 with actual sex labels
plt.legend(['Not at Risk', 'At Risk'], loc='upper right')

# Displaying counts above each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')


```

[TenYearCHD by Gender Chart](https://github.com/Irene-Chola/Massachussets-General-Hospital-Machine-Learning-Model/blob/main/Screenshot_7-7-2025_17129_.jpeg)

## 3. Smoking status
## 4. Blood pressure meds usage
   -Grouping data by BPMeds and TenYearCHD 
   
```import pandas as pd
import matplotlib.pyplot as plt

# Grouping data by BPMeds and TenYearCHD
bpmeds_chd_counts = dataset.groupby(['BPMeds', 'TenYearCHD']).size().unstack()

# Calculating percentages
total_no_bpmeds = bpmeds_chd_counts.loc[0].sum()
total_bpmeds = bpmeds_chd_counts.loc[1].sum()

percentage_no_bpmeds_at_risk = (bpmeds_chd_counts.loc[0, 1] / total_no_bpmeds) * 100
percentage_bpmeds_at_risk = (bpmeds_chd_counts.loc[1, 1] / total_bpmeds) * 100

# Plotting pie chart for No BPMeds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(bpmeds_chd_counts.loc[0], labels=['Not at Risk', f'At Risk ({round(percentage_no_bpmeds_at_risk, 2)}%)'], autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title('Distribution of TenYearCHD Versus No BPMeds')

# Plotting pie chart for BPMeds
plt.subplot(1, 2, 2)
plt.pie(bpmeds_chd_counts.loc[1], labels=['Not at Risk', f'At Risk ({round(percentage_bpmeds_at_risk, 2)}%)'], autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title('Distribution of TenYearCHD Versus BPMeds')

plt.tight_layout()
plt.show()
```
[BPMeds and TenYearCHDChart](https://github.com/Irene-Chola/Massachussets-General-Hospital-Machine-Learning-Model/blob/main/Screenshot_7-7-2025_17054_.jpeg)

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

The five models are trained on the training set. The models are validated on the testing set and their performance is evaluated based on the accuracy score, recall, F1 score and ROC_AUC graph.  

## Observations 
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
