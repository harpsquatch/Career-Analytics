# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:26:16 2022

@author: Harpreet Singh
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('eda_data.csv')

#Things to do
#Choose relevant columns 
#get dummy data - each categorical data need their dummy variable 
#Train_test split 
#Multiple linear regression 
#Lasso regression 
#randomforest
#WE can also use gradientboost, support vector regression and stuff

#After all this we will tune our model using gridsearchcv 


#Lets choose relevant columns 
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

#get dummy data for categorical data 

df_dum = pd.get_dummies(df_model)


from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1) #Takiing everything except the avg salary which we are going to predict 

y = df_dum.avg_salary.values #.values creates an array instead of a series. The array is recommneded to be used in models 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

#multiple linear regression 
import statsmodels.api as sm
X = sm.add_constant(X) #i dont know what is happening here 
model = sm.OLS(y,X)


from sklearn.linear_model import LinearRegression
lm = LinearRegression() #lm is the linear regression model, this will be our base line 
lm.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score 
np.mean(cross_val_score(lm,X_train,y_train,scoring = 'neg_mean_absolute_error', cv = 3))


from sklearn.linear_model import Lasso
#Lasso regression 
lassoreg = Lasso()
np.mean(cross_val_score(lassoreg,X_train,y_train,scoring = 'neg_mean_absolute_error', cv = 3))
#This model is performing worse than before 

#Lets change the aplha of lass regression and see if we can increase the mae 
alpha = []
error = []

for i in range(1,100): 
    alpha.append(i/100)
    lassoreg = Lasso(alpha = (i/109))
    error.append(np.mean(cross_val_score(lassoreg,X_train,y_train,scoring = 'neg_mean_absolute_error', cv = 3)))
    

plt.plot(alpha,error) #We can see that at alpha = 0.2, the mae is less

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err['error'] == max(df_err.error)]

#There alpha = 0.19 is the best value 

#RandomForestRegressor 
from sklearn.ensemble import RandomForestRegressor 
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train,scoring = 'neg_mean_absolute_error', cv = 3))
#RandomTree is working like a charm 


#Lets tune the models with gridsearchCV: it runs all the models and spits out the one which has the best results 

from sklearn.model_selection import GridSearchCV
#number of estimators 
#criterion 
#Different max features 
parameters ={'n_estimators':range(10,100,10),'criterion':('mse', 'mae'), 'max_features':('auto','sqrt','log2')}
#There are ton of different things that you can tune, need to learn more about it

gs = GridSearchCV(rf,parameters,scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(X_train,y_train)
#gs.best_score_  #Out gs best score is little better than the previous one random forest. But not alot ++++++++
#gs.best_estimator_  #gives you a best random forest with best parameters

#Now its time to predict and compare the results 

#PRedictions 
pred_lm = lm.predict(X_test)
#pred_lassoreg = lassoreg.predict(X_test)
pred_rf= gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,pred_lm)
#mean_absolute_error(y_test,pred_lassoreg)
mean_absolute_error(y_test,pred_rf)

#our randomforest is killing it. 
#Lets productionalize this machine learning model 


#Lets first pickle the model 

#Pickling converts the object into a byte stream which can be stored, transferred, and converted back to the original model at a later time. Pickles are one of the ways python lets you save just about any object out of the box.
import pickle 
pickl = {'model': gs.best_estimator_}
pickle.dump(pickl,open('model_file' + ".p",'wb'))

file_name = 'model_file.p'
with open(file_name,'rb') as pickled: 
    data = pickle.load(pickled)
    model = data['model']

model_prediction = model.predict(X_test.iloc[1,:])#THis is to see if "model" works. It works




