
# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loead dataset
dataset = pd.read_csv(r'S:\DOCS\Oct\Oct_31_CUSTOMERS REVIEW DATASET_capstone_project\Restaurant_Reviews.TSV',delimiter='\t',quoting=3)
new_dataset = pd.concat([dataset,dataset], ignore_index=True)


# cleaning the text 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[] # cleaned text will stored
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# bag of words

from sklearn.feature_extraction.text import TfidfVectorizer
bow = TfidfVectorizer()
x= bow.fit_transform(corpus).toarray()
x=np.concatenate([x,x],axis=0)
y= new_dataset.iloc[:,-1].values

# split into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)



# using multiple model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score


classifier ={
    'LogisticReg':LogisticRegression(),
    'decisiontree':DecisionTreeClassifier(),
    'svc' : SVC(),
    'knn' : KNeighborsClassifier(),
    'rf' : RandomForestClassifier(),
    'adaboost' : AdaBoostClassifier(),
    'xgboost' : XGBClassifier(),
    'Lightgbm':lgb.LGBMClassifier()
    }

result = pd.DataFrame(columns=['model','accuracy','bias','variance'])


for model_name,model_instance in classifier.items():
    model = model_instance.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    ac = accuracy_score(y_test, y_pred)
    bias = model.score(x_train,y_train)
    variance = model.score(x_test,y_test) 
    result = result.append({'model':model_name, 'accuracy':ac, 'bias':bias, 'variance':variance},ignore_index=True)
    
  
# applying k-fold cv

k_fold_result = pd.DataFrame(columns=['model','kfold_mean_score' ])
for model_name,model_instance in classifier.items():
    model = model_instance
    k_accuracy = cross_val_score(estimator=model, X=x_train,y=y_train,cv=5)
    kfold =k_accuracy.mean() 
    k_fold_result = k_fold_result.append({'model':model_name,'kfold_mean_score':kfold},ignore_index=True)
    