#  Heart Disease Risk Prediction Model
Machine Learning approach to predict cardiovascular disease risk using clinical data

# Table of Contents
[Project Overview](project-overview)

[Dataset Information](dataset-information)

[Data Preprocessing](data-preprocessing)

[Exploratory Data Analysis](exploratory-data-analysis)

[Data Visualization](data-visualization)

[Models Tested](models-tested)

[Models Training and Validation](model-training-and-validation)

[Models Performance](models-performance)

[Model Selection](model-selection)

[Findings](findings)

[Project Limitations](project-limitations)

## Project Overview
This machine learning project predicts the 10-year risk of coronary heart disease (CHD) using patient health data. The model analyzes various clinical indicators to provide early risk assessment, enabling proactive healthcare interventions.
Predicting heart disease risk is crucial for early intervention and preventive measures, contributing to better healthcare outcomes and improved patient well-being. 

## Dataset Information

•	Total Records: 4,240 patients

•	Features: 15 clinical and demographic variables

•	Target: 10-year CHD risk (binary classification)

•	Source: Massachusetts General Hospital clinical data

### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| Age | Patient age in years | Continuous |
| Sex | Gender (0=Female, 1=Male) | Binary |
| Smoking | Cigarettes per day | Continuous |
| Blood Pressure | Systolic/Diastolic readings | Continuous |
| Cholesterol | Total cholesterol levels | Continuous |
| BMI | Body Mass Index | Continuous |
| Heart Rate | Resting heart rate | Continuous |
| Glucose | Blood glucose levels | Continuous |

Here is the dataset used - [MGH Prediction Dataset](https://github.com/Irene-Chola/Massachussets-General-Hospital-Machine-Learning-Model/blob/main/MGH_Prediction_DataSet.csv)

Here is the full python script for this project - [Python Script](https://github.com/Irene-Chola/Massachussets-General-Hospital-Machine-Learning-Model/blob/main/PREDICTING%20HEART%20DISEASE%20RISK%20PYTHON%20PROJECT.ipynb)


## Data Preprocessing 
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

**Key Insights Discovered**

**1. Gender and Heart Disease Risk**

Males: Higher risk compared to females
Risk Ratio: 2.3x higher for males

**2. Age Distribution**

High Risk Group: Ages 50-65 years
Peak Risk: 58-62 years age range

**3. Blood Pressure Medication Usage**

On BP Meds: 23.4% higher CHD risk
No BP Meds: 8.7% CHD risk

**4. Smoking Impact**

Smokers: 18.3% CHD risk
Non-smokers: 11.2% CHD risk

 ---
## Data Visualization

The following variable relationships were visualized: 

(Two sample charts and their respective pythong codes provided)
#### 1. Age
#### 2. Gender
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

#### 3. Smoking status
#### 4. Blood pressure meds usage
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



## Models Tested
Machine Learning Models

1. Decision Tree - Interpretable, rule-based predictions
2. Logistic Regression - Linear probability modeling
3. K-Nearest Neighbors (KNN) - Instance-based learning
4. Support Vector Machine (SVM) - Margin-based classification
5. Random Forest - Ensemble tree-based method


The models are chosen based on their interpretability, performance, and suitability for the task of predicting heart disease risk. 

## Models Training and Validation 

The dataset is split into training and testing sets to train and validate the selected models. 

The five models are trained on the training set. The models are validated on the testing set and their performance is evaluated based on the accuracy score, recall, F1 score and ROC_AUC graph.  

## Models Performance 

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 86% | 0.86 | 1.00 | 0.92 |
| Random Forest | 85% | 0.86 | 0.99 | 0.92 |
| SVM | 85% | 0.85 | 1.00 | 0.92 |
| KNN | 84% | 0.86 | 0.97 | 0.91 |
| Decision Tree | 75% | 0.86 | 0.85 | 0.85 |

## Model Selection

**Best Model:**

**Logistic Regression**

**-Highest accuracy** (86%) with balanced precision and recall

**-Clinical Relevance:** Provides probability scores for risk assessment

**-Interpretability:** Clear feature importance for medical decision-making

----

## Findings

**Feature Importance:**

-Age - Most significant predictor (0.234)

-Systolic BP - Second most important (0.187)

-Smoking - Third highest impact (0.156)

-Cholesterol - Fourth predictor (0.143)

-BMI - Fifth most relevant (0.128)

**Clinical Insights:**

**1. High-Risk Patient Profile**

-Age: 50+ years

-Gender: Male

**2. Health Indicators**

-Systolic BP > 140 mmHg
  
-Total Cholesterol > 240 mg/dL
  
-BMI > 30
  
-Active smoker

**3. Risk Factors Impact**

-Modifiable Factors: Smoking, BP, cholesterol, BMI

-Non-modifiable: Age, gender, family history


---

## Project Limitations

1. Dataset Size: Limited to 4,240 patients
   
3. Geographic Scope: Massachusetts population only
   
5. Time Frame: 10-year prediction window
   
7. Feature Limitations: Missing genetic and lifestyle factors

   


