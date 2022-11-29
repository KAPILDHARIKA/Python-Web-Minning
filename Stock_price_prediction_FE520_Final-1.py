#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


# In[2]:


df = quandl.get("WIKI/GOOGL")
df.to_csv('sap_stock.csv')
df.to_excel('saved_file.xlsx')

#print(data)
print(df)


# In[3]:


df.info()


# In[4]:


df['Split Ratio'].value_counts()


# In[5]:


df.columns


# In[6]:


Google_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]


# In[7]:


Google_df


# In[8]:


x = Google_df[['Open', 'High', 'Low', 'Volume']]
y = Google_df['Close']


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[14]:


print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test.shape)


# In[15]:


model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)


# In[17]:


import sklearn.metrics as sm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("Explain variance score =", round(sm.explained_variance_score(y_test, prediction), 2))
print("R2 score =", round(sm.r2_score(y_test, prediction)))



#For LSTM

import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns

dataset_train = web.DataReader('FB',data_source='yahoo', start='2016-01-01',end='2021-05-01' )
dataset_train.to_csv('sap_stock.csv')
training_data = pd.read_csv('sap_stock.csv')

training_data = training_data.iloc[:, 1:2]
training_data.shape
training_data.head()

# feature scaling
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range = (0, 1))
training_data = mm.fit_transform(training_data)

# Getting the inputs and outputs
x_train = training_data[0:1291]
y_train = training_data[50:1341]
print(x_train.shape)
print(y_train.shape)

# reshaping
x_train = np.reshape(x_train, (1291, 1, 1))
print(x_train.shape)

# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# initializing the model
model = Sequential()
# adding the input layer and the LSTM layer
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
# adding the output layer
model.add(Dense(units = 1))
# compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 32, epochs = 200)


test_data = pd.read_csv('sap_stock.csv')
#targetting the High value"
real_stock_price = test_data.iloc[:,1:2]  
real_stock_price.head()

# getting the predicted stock price
import numpy as np
inputs = real_stock_price
inputs = mm.transform(inputs)
inputs = np.reshape(inputs, (1341,1,1))
predicted_stock_price = model.predict(inputs)
predicted_stock_price = mm.inverse_transform(predicted_stock_price)
print(predicted_stock_price)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Stock Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()

# Evaluating the model
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)
print(r2_score(real_stock_price, predicted_stock_price))






# Prophet Model



import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns

from fbprophet import Prophet

from datetime import datetime

from sklearn.metrics import mean_absolute_error

from pandas.tseries.offsets import *



FB = web.DataReader('FB',data_source='yahoo', start='2016-01-01',end='2020-12-31' )
FB.to_csv('FB.csv')
FB = pd.read_csv('FB.csv')



print('FB')
print(FB.head())

#print(FB.describe())


FB = FB[['Date','Adj Close']]

FB = FB.rename(columns = {'Date':'ds', 'Adj Close':'y'})


m = Prophet(daily_seasonality = True)


FB_copy = FB.copy('deep')


split_date = '2020-01-01'

df_training = FB.loc[FB['ds'] <= split_date]
df_test = FB.loc[FB['ds'] > split_date]



m.fit(df_training)






####################################################
# Insample forecast

'''
future = list()

date = pd.date_range('2020-01-01', periods=366).tolist()


future = date
  '''

  
future = pd.DataFrame(df_test['ds'])
future.columns = ['ds']



future['ds'] = pd.to_datetime(future['ds'])



forecast = m.predict(future)

df_test = df_test.set_index('ds')



ax1 = m.plot(forecast)
plt.title('In Sample Prediction - for FB data')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.tight_layout()
plt.savefig('InsampleFB.png',dpi=300)
plt.show()



##########################################################

# Manually evaluate forecast model



y_true = df_test[['y']].values
y_pred = forecast['yhat'].values

mae  = mean_absolute_error(y_true,y_pred)

plt.plot(y_true, label = 'Actual')
plt.plot(y_pred, label  = 'Predicted')
plt.legend()
plt.title('Actual vs Pred  - for FB data year 2020')
plt.xlabel('Day of year')
plt.ylabel('Adj Close')
plt.tight_layout()
plt.savefig('ActaulVPredFB.png',dpi=300)
plt.show()




