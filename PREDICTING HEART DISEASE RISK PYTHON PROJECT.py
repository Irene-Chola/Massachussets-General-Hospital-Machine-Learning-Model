#!/usr/bin/env python
# coding: utf-8

#  ## Predicting Heart Disease Risk
# 

# In[1]:


# important needed libraries
import pandas as pd
import numpy as np


# In[2]:


# load dataset
dataset = pd.read_csv('MGH_PredictionDataSet.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# # CLEANING

# In[5]:


dataset.info()


# In[6]:


# View sum of nulls in each column
dataset.isnull().sum()


# In[7]:


# Replace the nulls

# Categorical Data - Replace with Mode
dataset['education'].fillna(dataset['education'].mode()[0], inplace = True)
dataset['BPMeds'].fillna(dataset['BPMeds'].mode()[0], inplace = True)

# Continous Data - Replace with Mean
dataset['cigsPerDay'].fillna(dataset['cigsPerDay'].mean(), inplace = True)
dataset['totChol'].fillna(dataset['totChol'].mean(), inplace = True)
dataset['BMI'].fillna(dataset['BMI'].mean(), inplace = True)
dataset['heartRate'].fillna(dataset['heartRate'].mean(), inplace = True)
dataset['glucose'].fillna(dataset['glucose'].mean(), inplace = True)


# In[8]:


# Remove duplicates
dataset = dataset.drop_duplicates()


# In[9]:


# View sum of nulls in each column
dataset.isnull().sum()


# In[10]:


# View the new shape after replacing nulls
dataset.shape


# # EXPLORATORY DATA ANALYSIS (EDA)

# This section helps us understand the structure of the data, identify patterns, anomalies, develop hypotheses, check assumptions, summarize statistics and display it graphically.

# In[11]:


# Summary Statistics
dataset.describe()


# In[12]:


# Correlation between variables
dataset.corr(numeric_only = True)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


heatmap_data = dataset.corr(numeric_only = True)


# In[15]:


# Represent correlation matrix as a heatmap
fig, ax = plt.subplots(figsize =(15,15)) # this code adjusts the size of the heatmap
sns.heatmap(heatmap_data, annot = True,cmap = 'coolwarm')


# In[16]:


# Histogram Using Matplotlib
dataset['age'].hist(bins=20, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age in Years')
plt.ylabel('Frequency')
plt.grid(True)


# In[17]:


import matplotlib.pyplot as plt
# Counting the occurrences of each unique value in TenYearCHD
chd_counts = dataset['TenYearCHD'].value_counts()

# Plotting the distribution of TenYearCHD using a bar chart
plt.figure(figsize=(6, 4))
bars = chd_counts.plot(kind='bar', color=['blue', 'orange'])
# Adding count labels on top of each bar
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width()/2 - 0.05, bar.get_height() + 0.05, f'{int(bar.get_height())}', ha='center', color='black', fontsize=10)

plt.title('Distribution of TenYearCHD')
plt.xlabel('TenYearCHD')
plt.ylabel('Count')
plt.xticks(rotation=0)


# In[18]:


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


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# Defining age brackets in 10-year intervals
age_bins = [i for i in range(20, 101, 10)]

# Creating age brackets and group by TenYearCHD
dataset['age_group'] = pd.cut(dataset['age'], bins=age_bins)
age_chd_counts = dataset.groupby(['age_group', 'TenYearCHD']).size().unstack()

# Plotting the grouped bar chart
plt.figure(figsize=(12, 8))
ax = age_chd_counts.plot(kind='bar', stacked=False, color=['navy', 'turquoise'])
plt.title('Distribution of Age by TenYearCHD')
plt.xlabel('Age Group (years)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(['Not at Risk', 'At Risk'], loc='upper right')

# Displaying counts above each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width(), p.get_height()), ha='center', va='bottom')


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt

# Grouping data by currentSmoker and TenYearCHD
smoker_chd_counts = dataset.groupby(['currentSmoker', 'TenYearCHD']).size().unstack()

# Plotting the grouped bar chart
plt.figure(figsize=(8, 6))
ax = smoker_chd_counts.plot(kind='bar', stacked=False,color=['purple', 'navy'])
plt.title('Distribution of Current Smokers by TenYearCHD')
plt.xlabel('Current Smoker')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Smoker', 'Smoker'], rotation=0)
plt.legend(['Not at Risk', 'At Risk'], loc='upper right')

# Displaying counts above each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')


# In[21]:


import pandas as pd
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


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt

# Selecting numerical features for pairwise scatter plots
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns

# Creating pairwise scatter plots
sns.pairplot(dataset[numerical_features])
plt.suptitle("Pairwise Scatter Plots")
plt.show()


# # FEATURE SELECTION

# In[23]:


dataset.columns


# In[24]:


# Select features from the columns and target
features = ['sex', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']


# In[25]:


features


# In[26]:


X = dataset[features]


# In[27]:


X.head()


# In[28]:


# identify the target denoted by y
target = ['TenYearCHD']


# In[29]:


target


# In[30]:


y = dataset[target]


# In[31]:


y.head()


# # SPLITTING THE DATASET

# In[32]:


from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets (80% train, 20% spolit)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[33]:


X_train.shape


# In[34]:


y_train.shape


# In[35]:


X_test.shape


# In[36]:


y_train.shape


# # MODEL SELECTION AND TRAINING

# ### 1. Decision Tree Classifier

# In[37]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Model
tree_model = DecisionTreeClassifier()


# In[38]:


# Fitting the model with the training data
tree_model.fit(X_train, y_train)


# In[39]:


accuracy_score = tree_model.score(X_test, y_test)
accuracy_score


# In[40]:


from sklearn.metrics import classification_report

# Predictions for Decision tree Regression
y_pred_lr = tree_model.predict(X_test)

# Classification Report for Logistic Regression
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))



# ### 2. Logistic Regression

# In[41]:


# This code ignores warnings
import warnings
warnings.filterwarnings('ignore')


# In[42]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()


# In[43]:


# Fitting the model with the training data
logistic_model.fit(X_train,y_train)


# In[44]:


accuracy_score = logistic_model.score(X_test, y_test)
accuracy_score


# In[45]:


from sklearn.metrics import classification_report

# Predictions for Logistic Regression
y_pred_lr = logistic_model.predict(X_test)

# Classification Report for Logistic Regression
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# ### 3. K-Nearest Neighbours (KNN)

# In[46]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors =5)


# In[47]:


# Fitting the model with the training data
knn_model.fit(X_train,y_train)


# In[48]:


accuracy_score = knn_model.score(X_test, y_test)
accuracy_score


# In[49]:


from sklearn.metrics import classification_report

# Predictions for knn model
y_pred_lr = knn_model.predict(X_test)

# Classification Report for Logistic Regression
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# ### 4. Support Vector Machine (SVM)

# In[50]:


from sklearn.svm import SVC
svm_model = SVC(kernel='linear', probability=True)


# In[51]:


# Fitting the model with training data
svm_model.fit(X_train,y_train)


# In[52]:


accuracy_score = svm_model.score(X_test, y_test)
accuracy_score


# In[53]:


from sklearn.metrics import classification_report

# Predictions for svm model
y_pred_lr = svm_model.predict(X_test)

# Classification Report for Logistic Regression
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# ### 5. Random Forest

# In[54]:


from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[55]:


# Fitting the model with the training data
forest_model.fit(X_train,y_train)


# In[56]:


accuracy_score = forest_model.score(X_test, y_test)
accuracy_score


# In[57]:


from sklearn.metrics import classification_report

# Predictions for random forest
y_pred_lr = forest_model.predict(X_test)

# Classification Report for Logistic Regression
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# # PREDICTION

# The best model chosen for this machine learning task is Logistic regression as it gives the highest accuracy score

# In[58]:


features


# In[59]:


# Define a sample list of feature values
random_features = [1,35,3,0,10,1,0,1,0,150,120,80,27,72,75]


# In[60]:


# Convert the features into a dataframe
feature_df = pd.DataFrame([random_features])


# In[61]:


# Prediction using the logistic regression model
prediction = logistic_model.predict(feature_df)


# In[62]:


prediction


# # Combined ROC_AUC graph

# This graph assesses the model's ability to distinguish between binary classes. 
# **Points above the diagonal line represents good classification. The more the model is skewed towards the upper left corner, the better the performance.**

# In[63]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#probabilities and labels for each model
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]
y_prob_dt = tree_model.predict_proba(X_test)[:, 1]
y_prob_rf = forest_model.predict_proba(X_test)[:, 1]
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]


#  Logistic Regression model ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

#Decision Tree model
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

# Random Forest model
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

#SVM model
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)

#KNN model
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = roc_auc_score(y_test, y_prob_knn)

# combined ROC curve
plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label='Decision Tree (area = %0.2f)' % roc_auc_dt)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (area = %0.2f)' % roc_auc_rf)
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_knn, tpr_knn, color='purple', lw=2, label='KNN (area = %0.2f)' % roc_auc_knn)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (All Models)')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




