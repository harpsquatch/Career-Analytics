# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:27:32 2022

@author: Harpreet Singh
"""

import pandas as pd 
df = pd.read_csv("C:\\Users\\Harpreet Singh\\Documents\\Machine Learning\\glassdoor_jobs.csv")

#So right off the bat we can make out that the following things are required to be done
# 1) Salary parksing - Remove texts like Ks and (Glassdoor est)
# 2) Company name should include text only 
# 3) State field 
# 4) Parsing of the job description (python,etc)

# SALARY PARSING 
#Lets first remove those -1 values, those things are not related 
df = df[df['Salary Estimate'] != '-1']
#In order to ensure that we dont miss important information, lets mention things like per hour and employer provided salary 
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0 )
                                          #check kar agar per hour hai x mein (lower mein convert karke pehle)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)


#Now that we have saved some important metric, lets remove all the texts that is available
salary = df['Salary Estimate'].apply(lambda x:x.split('(')[0]) 
#Im taking the salary component, splitting it by ( and making a list. with [0] im taking the first element of the list 
removealltexts = salary.apply(lambda x: x.lower().replace('k','').replace('$','').replace('per hour', '').replace('employer provided salary:', ''))

#Now we have min and max salary, lets make two columns, min and max

df['min_salary'] = removealltexts.apply(lambda x: x.split('-')[0]) #Takes the first value of the list. 

df['max_salary'] = removealltexts.apply(lambda x: x.split('-')[1]) #Takes the second value of the list  
 

#Lets convert their data types to int 
df['min_salary'] = pd.to_numeric(df['min_salary'])
df['max_salary'] = pd.to_numeric(df['max_salary'])
df['average_salary'] = (df['min_salary'] + df['max_salary'])/2

#We are done with the salary parsing 




#Company Name Parsing, I want it clean, without any rating 

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3], axis = 1)#Everything until last three characters, the statment alone will throw an error because axis is rows by deafult

#State feild
df['Job_State'] = df['Location'].apply(lambda x: x.split(',')[1])

#Age of comoany 
df['age'] = df['Founded'].apply(lambda x: x if x<0 else 2020-x)







#Now lets see nside if we have the the popular tools in job description

#Python 
df['Python y/n'] = df['Job Description'].apply(lambda x: 1 if "python" in x.lower() else 0)

#spark 
df['spark y/n'] = df['Job Description'].apply(lambda x: 1 if "spark" in x.lower() else 0)

#excel
df['excel y/n'] = df['Job Description'].apply(lambda x: 1 if "excel" in x.lower() else 0)

#r studio 
df['r studio'] = df['Job Description'].apply(lambda x: 1 if "r studio" in x.lower() else 0)


df_out = df.drop(['Unnamed: 0'],axis =1)
df_out.to_csv('salary_data_cleaned.csv',index = False)