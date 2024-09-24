#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:/Users/bendh/Desktop/data science/JN/filtered_customer_booking.csv", encoding="ISO-8859-1")
df = df.drop(columns=['Unnamed: 0'])
df.head()


# ### One Hot Encode

# In[5]:


#one hot encode categorical values
from sklearn.preprocessing import OneHotEncoder

df2 = df

#create instanc"e of one hot encoder
encoder = OneHotEncoder(handle_unknown='ignore')

#one hot encode Sales Channel
encoder_df = pd.DataFrame(encoder.fit_transform(df[["sales_channel"]]).toarray())
encoder_df = encoder_df.rename(columns={0:'Internet', 1:'Phone'})
df2 = df2.join(encoder_df)

#one hot encode trip type
encoder_df = pd.DataFrame(encoder.fit_transform(df[["trip_type"]]).toarray())
encoder_df = encoder_df.rename(columns={0:'RoundTrip', 1:'OneWayTrip',2:'CircleTrip'})
df2 = df2.join(encoder_df)



# In[6]:


#drop categorical columns now
df2.drop(['sales_channel', 'trip_type', 'booking_origin', 'route'], axis=1, inplace = True)


# In[7]:


#store the label for supervised learning
label = df["booking_complete"]


# In[8]:


df2 = df2.drop("booking_complete", axis=1)


# In[9]:


df2


# ## Normalizing values

# In[10]:


from sklearn.preprocessing import StandardScaler

#create a standard scaler object
scaler = StandardScaler()

#fit and transform the data
scaled_df = scaler.fit_transform(df2)


# In[11]:


#create a dataframe of scaled data
scaled_df = pd.DataFrame(scaled_df, columns = df2.columns)


# In[12]:


#add the labels back to the dataframe
scaled_df['label'] = label


# In[13]:


scaled_df


# ## Correlation matrix

# In[14]:


corr = scaled_df.corr()

plt.figure(figsize=(10,7))

#plot the heatmap
sns.heatmap(corr)


# ### Splitting Train and Test Data

# In[15]:


from sklearn.model_selection import train_test_split

X = scaled_df.iloc[:,:-1]
y = scaled_df['label']

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.20, random_state = 42) 


# In[16]:


get_ipython().system('pip install yellowbrick')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.inspection import permutation_importance

from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


# In[17]:


#create functions to fit and predict the values of wether customer would complete the booking or not
#create functions with metrics to evaluate the model prediction

#check how well the model is performing on known data
def model_fit_predict(model, X, y, X_predict):
    model.fit(X,y)
    return model.predict(X_predict)

def acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    
def pre_score(y_true, y_pred):
    return precision_score(y_true, y_pred)

def f_score(y_true, y_pred):
    return f1_score(y_true, y_pred)


# # Random Forest Classifier

# In[18]:


#create an isntance of the classifier and fit the training data
clf_rf = RandomForestClassifier(max_depth = 50, min_samples_split= 5, random_state= 0)


# ### Checking Training Accuracy

# In[19]:


y_pred_train = model_fit_predict(clf_rf, X_train, y_train, X_train)
set(y_pred_train)

#f1 score for training data : It balances both false positives and false negatives
f1 = round(f1_score(y_train, y_pred_train),2) 

#accuracy score for training data : the ratio of correctly predicted instances to total instances.
acc = round(accuracy_score(y_train, y_pred_train),2) 

#precision score for training data : how many of the predicted positive cases were actually correct.
pre = round(precision_score(y_train, y_pred_train),2) 

print(f"Accuracy, precision and f1-score for training data are {acc}, {pre} and {f1} respectively")


# In[20]:


# confusion matrix shows the number of correct and incorrect predictions for each class.
cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, y_train)

cm.score(X_train, y_train)


# ### Checking Testing accuracy

# In[21]:


#create an instance of the classifier and fit the training data
clf_rf = RandomForestClassifier(max_depth =50 , min_samples_split=5,random_state=0)

y_pred_test = model_fit_predict(clf_rf, X_train, y_train, X_test)

#f1 score for training data
f1 = round(f1_score(y_test, y_pred_test),2) 

#accuracy score for training data
acc = round(accuracy_score(y_test, y_pred_test),2) 

#precision score for training data
pre = round(precision_score(y_test, y_pred_test),2) 

print(f"Accuracy, precision and f1-score for training data are {acc}, {pre} and {f1} respectively")


# In[22]:


cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, y_train)

cm.score(X_test, y_test)


# In[23]:


plt.figure(figsize=(10,5))
sorted_idx = clf_rf.feature_importances_.argsort()
plt.barh(scaled_df.iloc[:,:-1].columns[sorted_idx], clf_rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# In[24]:


# One major problem behind getting low F1 score is imbalanced dataset. We have higher entries that are classified 0 than 1.
#We could reduce the number of entries that are classified 0 to be equal around the number of entries that are classified as 1.


# ### Balancing the dataset

# In[25]:


scaled_df.label.value_counts()


# In[26]:


#create a dataframe having all labels 0 with 10000 samples
scaled_df_0 = scaled_df[scaled_df.label ==0].sample(n=8000)


# In[27]:


#concatenate the two dataframee, one havng all labels 0 and other having all labels as 1
scaled_df_new = pd.concat([scaled_df[scaled_df.label==1], scaled_df_0], ignore_index=True)


# In[28]:


#shuffle the dataframe rows
scaled_df_new = scaled_df_new.sample(frac = 1).reset_index(drop=True)


# In[29]:


scaled_df_new


# In[30]:


X = scaled_df_new.iloc[:,:-1]
y = scaled_df_new['label']

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.20, random_state=42)


# In[31]:


#create an instance of the classifier and fit the training data
clf_rf = RandomForestClassifier(n_estimators=50,max_depth =50 , min_samples_split=5,random_state=0)


# In[32]:


y_pred_test = model_fit_predict(clf_rf, X_train, y_train, X_test)

#f1 score for training data
f1 = round(f1_score(y_test, y_pred_test),2) 

#accuracy score for training data
acc = round(accuracy_score(y_test, y_pred_test),2) 

#precision score for training data
pre = round(precision_score(y_test, y_pred_test),2) 

#Measures how well the model identifies all the true positives (completed bookings).
recall = round(recall_score(y_test, y_pred_test),2)

#Measures how well the model identifies the true negatives (non-completed bookings).
specificity = round(recall_score(y_test, y_pred_test, pos_label=0),2)

print(f"Accuracy, precision, recall and f1-score for training data are {acc}, {pre}, {recall}, {specificity} and {f1} respectively") 


# In[33]:


cm = ConfusionMatrix(clf_rf, classes=[0,1])
cm.fit(X_train, y_train)

cm.score(X_test, y_test)


# In[34]:


plt.figure(figsize=(10,8))
sorted_idx = clf_rf.feature_importances_.argsort()
plt.barh(scaled_df.iloc[:,:-1].columns[sorted_idx], clf_rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# In[35]:


from sklearn.metrics import roc_curve, auc


# In[36]:


y_pred_proba = clf_rf.predict_proba(X_test)[:, 1]


# In[37]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[38]:


roc_auc = auc(fpr, tpr)


# In[39]:


plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




