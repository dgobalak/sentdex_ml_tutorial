import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import time


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
print(forecast_out) # 30; We'll be prediciting prices 30 days in advance

# The label column will be the Adj Close column from forecast_out days ago
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Will help with training/testing but adds to processing time
X = preprocessing.scale(X)

# 20% of data will be used as testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Making classifier
clf = LinearRegression(n_jobs=-1)
start = time.time()
# clf = svm.SVR()
clf.fit(X_train, y_train)
end = time.time()


# Testing classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)

print(f'Elapsed Time: {end - start} sec')