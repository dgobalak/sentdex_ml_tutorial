import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Feature - Input Variable(s)
# Label - THe thing we're predicting

df = quandl.get('WIKI/GOOGL')   # Quandl used to get the dataset

# For linear regression, we want to only use the necessary features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High'] / df['Adj. Close'] - 1.0) * 100.0)
df['PCT_change'] = ((df['Adj. Close'] / df['Adj. Open'] - 1.0) * 100.0)

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out) # 30; We'll be prediciting prices 30 days in advance

# The label column will be the Adj Close column from forecast_out days ago
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))

# Will help with training/testing but adds to processing time
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

# 20% of data will be used as testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Making classifier
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR()
clf.fit(X_train, y_train)

# Testing classifier
accuracy = clf.score(X_test, y_test)
# print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
# print(df.head())
for i in forecast_set:
    next_data = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Sets forecast column in row to the prediction. df.loc[] is the index for each row
    df.loc[next_data] = [np.nan for _ in range(len(df.columns)-1)] + [i]   

# print(df.head())
# print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show() 