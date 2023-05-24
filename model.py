import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## For preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

####################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


## Machine Learning Algorithm
#from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
#from sklearn.ensemble import VotingClassifier

## For Evaluation Model
from sklearn.metrics import accuracy_score

## Export model
import pickle

path = 'C:/Users/user/bodyPerformance.csv'
df = pd.read_csv(path)
df_raw = df.copy

sex = df[['gender']]
target = df[['class']]

label_encoder = LabelEncoder()
label_encoder.fit(df['class'])
label_encoded_class = label_encoder.transform(df['class'])

label_encoder = LabelEncoder()
label_encoder.fit(df['gender'])
label_encoded_gender = label_encoder.transform(df['gender'])

df.drop(columns=['gender', 'class'], axis=1, inplace=True)

#df['gender'] = label_encoded_gender

xtrain, xtest, ytrain, ytest = train_test_split(df,label_encoded_class, test_size=0.2, random_state=42)

XgbModel = XGBClassifier()
XgbModel.fit(xtrain, ytrain)

y_pred = XgbModel.predict(xtest)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

pickle.dump(XgbModel,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))