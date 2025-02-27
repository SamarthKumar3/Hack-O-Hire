# -*- coding: utf-8 -*-
"""barclays.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1312yqx3e-v7hmJIKBL2pmxtEKsXcvuf8
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure
import seaborn as sns


pd.options.mode.chained_assignment = None
# %matplotlib inline

def read_data(data = 'data_name')->pd.DataFrame:
  df = pd.read_csv(data, delimiter = ';', low_memory = False)
  df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
  return df

df.to_csv('output.csv')

df = read_data(data = 'trans.csv')
df.head()

df.columns

df.info()

df.columns

def filter_features(ls = [], df = pd.DataFrame)->pd.DataFrame:
  df = df[['date', 'account_id', 'type', 'amount']]
  if 'data' in ls:
    df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
  return df

df_new = filter_features(ls = df.columns, df = df )
df_new

sns.displot(df_new['amount'], bins = 50, kde = True)

to_replace = {'PRIJEM': 'CREDIT', 'VYDAJ': 'WITHDRAWAL', 'VYBER': 'NOT SURE', 'DEBIT': 'WITHDRAWAL'}
df_new['type'] = df_new['type'].replace(to_replace)

df_new

sns.countplot(x = 'type', data = df_new)

df_new.head()

df_new = df_new[df_new['type'] == 'WITHDRAWAL']
df_new.sort_values(by = 'account_id')

df_new.set_index('date',inplace = True)
df_new[0:5]

df_new['sum_5days'] = df_new.groupby('account_id')['amount'].transform(lambda s: s.rolling(timedelta(days=5)).sum())
df_new['count_5days'] = df_new.groupby('account_id')['amount'].transform(lambda s: s.rolling(timedelta(days=5)).count())

df_new

sns.distplot(df_new['sum_5days'], bins=50)

# sns.countplot(x='count_5days', data=df_new)

# sns.catplot(x="count_5days", y="sum_5days",
#             kind="box", data=df_new, order=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             aspect=3.0)

sns.countplot(x='count_5days', data=df_new)

!pip install pyod

from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


anomaly_proportion = 0.001

clf_name = 'Anomaly Detection - Isolation Forest'
clf = IForest(contamination = anomaly_proportion)

X = df_new[['count_5days', 'sum_5days']]

clf.fit(X)

# get the prediction labels and outlier scores of the training data
df_new['y_pred'] = clf.labels_ # binary labels (0: inliers, 1: outliers)
df_new['y_scores'] = clf.decision_scores_ # raw outlier scores. The bigger the number the greater the anomaly.

df_new[df_new['y_pred'] == 1]

df_withdrawals = df_new
xx , yy = np.meshgrid(np.linspace(0, 11, 200), np.linspace(0, 180000, 200))

# decision function calculates the raw anomaly score for every point
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])*-1
Z = Z.reshape(xx.shape)


threshold = (df_withdrawals.loc[df_withdrawals['y_pred'] == 1, 'y_scores'].min()*-1)/2 + (df_withdrawals.loc[df_withdrawals['y_pred'] == 0, 'y_scores'].max()*-1)/2


subplot = plt.subplot(1, 1, 1)

# fill blue colormap from minimum anomaly score to threshold value
subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)

# draw red contour line where anomaly score is equal to threshold
a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

# fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')


msk = df_withdrawals['y_pred'] == 0
x = df_withdrawals.loc[msk, ['count_5days', 'sum_5days']].values

# scatter plot of inliers with white dots
b = subplot.scatter(x[:, 0], x[:, 1], c='white',s=20, edgecolor='k')


msk = df_withdrawals['y_pred'] == 1
x = df_withdrawals.loc[msk, ['count_5days', 'sum_5days']].values

# scatter plot of outliers with black dots
c = subplot.scatter(x[:, 0], x[:, 1], c='black',s=20, edgecolor='r')
subplot.axis('tight')



subplot.legend(
    [a.collections[0], b, c],
    ['learned decision function', 'inliers', 'outliers'],
    prop=matplotlib.font_manager.FontProperties(size=10),
    loc='upper right')

subplot.set_title(clf_name)
subplot.set_xlim((0, 11))
subplot.set_ylim((0, 180000))

subplot.set_xlabel("5-day count of withdrawal transactions.")
subplot.set_ylabel("5-day sum of withdrawal transactions")

import pickle

# Assuming 'clf' is your trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

