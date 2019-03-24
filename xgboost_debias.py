import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv('salary_data.csv', sep=',', index_col=False)

# one hot encoding all categorical data
data = pd.get_dummies(df, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis = 1),
                                                    np.log1p(data.salary.values),
                                                    test_size =.30,
                                                    random_state=12345)
dtrain = xgb.DMatrix(data = X_train, label=y_train)
dtest = xgb.DMatrix(data = X_test, label=y_test)
param = {'eta':0.1,
         'objective':'reg:linear'}
xgb_model = xgb.train(param, dtrain, num_boost_round=100)

X_test['y_hat_with_bias'] = np.exp(xgb_model.predict(dtest)) - 1

print(X_test)

feature_names = dtest.feature_names
interactions = xgb_model.predict(dtest, pred_interactions=True)
pd.DataFrame(interactions[0],
             index = feature_names + ['intercept'],
             columns= feature_names + ['intercept'])

bias_var = np.array('gender_male')
print(bias_var)
bias_idx = np.argwhere(np.isin(np.array(feature_names), bias_var))[0]
interactions[:, bias_idx, :] = 0
interactions[:, :, bias_idx] = 0
y_hat_no_bias = np.exp(interactions.sum(axis = 1).sum(axis = 1)) - 1

print(y_hat_no_bias)
