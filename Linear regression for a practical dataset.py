import pandas as pd
import numpy as np

df=pd.read_csv('cardio_dataset.csv')
#read the csv into a pandas dataframe
#print(df.head[5])

df=df.values
#converting the dataframe object into a numpy array
#print(df)

data=df[:,0:7]
target=df[:,7]

target=np.reshape(target, (-1,1))

from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data,target,test_size=0.1)

from sklearn.linear_model import LinearRegression

clsfr=LinearRegression()

clsfr.fit(train_data,train_target)

result=clsfr.predict(test_data)

from sklearn.metrics import r2_score

r2_value=r2_score(test_target,result)

print('r2_score:',r2_value)

print('Actual value:',test_target[0:10])
print('Predicted value',result[0:10])
