import pandas as pd
import numpy as np
import random

#from sklearn.feature_extraction.text import TCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

urls_data=pd.read_csv('urldata.csv')

type(urls_data)



urls_data.head()

def makeTokens(f):
        tkns_BySlash=str(f.encode('utf-8')).split('/')
        total_Tokens=[]
        for i in tkns_BySlash:
            tokens=str(i).split('-')
            tkns_ByDot=[]
            for j in range(0,len(tokens)):
                temp_Tokens=str(tokens[j]).split('.')
                tkns_ByDot=tkns_ByDot+temp_Tokens
            total_Tokens=total_Tokens+tkns_ByDot+tkns_BySlash
        total_Tokens=list(set(total_Tokens))
        if 'com' in total_Tokens:
            total_Tokens.remove('com')
        return total_Tokens

y=urls_data['label']

url_list=urls_data['url']

vectorizer=TfidfVectorizer(tokenizer=makeTokens)

X=vectorizer.fit_transform(url_list)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Lr=LogisticRegression()

Lr.fit(X_train,Y_train)

print("Accuracy",Lr.score(X_test,Y_test))

X_predict=["wikepedia.com",'google.com','https://github.com/Jcharis/Machine-Learning-In-Julia-JCharisTech','sanath.com/test.exe']

X_predict=vectorizer.transform(X_predict)
Y_predict=Lr.predict(X_predict)
print (Y_predict)
