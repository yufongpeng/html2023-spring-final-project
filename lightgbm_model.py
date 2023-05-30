import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict, GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, make_scorer
from skopt import BayesSearchCV
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

#import data and select only the numeric columns
df = pd.read_csv('kaggle/html2023-spring-final-project/train.csv')
df['Artist'] = df['Artist'].astype('category')
df['Composer'] = df['Composer'].astype('category')
y = df['Danceability']
test_df = pd.read_csv("kaggle/html2023-spring-final-project/test.csv", delimiter=",", header=0)
#x_train = df.drop(['Danceability','Album_type','Album','Licensed','official_video','id','Track','Uri','Url_spotify','Url_youtube','Description','Title','Channel'], axis =1)
n_train = df.shape[0]
n_test = test_df.shape[0]
encoded_df = pd.get_dummies(df.append(test_df), columns=['Artist', 'Composer'])
encoded_df = encoded_df.drop(['Danceability','Album_type','Album','id','Licensed','official_video','Track','Uri','Url_spotify','Url_youtube','Description','Title','Channel'], axis =1)
x_train = encoded_df.iloc[:n_train, :]
x_test = encoded_df.iloc[n_train:, :]

def mask(x):
    np.putmask(x, x>9, 9)
    np.putmask(x, x<0, 0)
    return np.round(x)


#param_space for bayesSearch
param_space = {
    'regressor__learning_rate': (0.01,0.1,'uniform'),
    'regressor__num_leaves': (2, 63, 'uniform'),
    'regressor__max_depth': (3, 4, 5, 6),
    'regressor__n_estimators': (10,100,1000),
    #'min_split_gain': (50,1000, 'uniform'),
    #'min_child_weight':(0.001, 0.01, 0.1, 1), 
    'regressor__subsample': (0.8, 0.9, 1.0),
    'regressor__colsample_bytree': (0.8, 0.9, 1.0),
    'regressor__min_child_samples': (1,10,100,200,300),
    'regressor__reg_alpha': (0,1000, 'uniform'), #l1 regularizer
    'regressor__reg_lambda': (0,1000, 'uniform') #l2 regularizer
}

model = TransformedTargetRegressor(regressor=lgb.LGBMRegressor(), inverse_func=mask, check_inverse=False)
#consruct bayesSearch object
opt = BayesSearchCV(estimator = model, search_spaces = param_space, n_iter = 20, cv = 5, scoring='neg_mean_absolute_error', random_state=1)
opt.fit(x_train.to_numpy(), y)
best_model = opt.best_estimator_
best_param = opt.best_params_
mae = mean_absolute_error(cross_val_predict(best_model, x_train.to_numpy(), y, cv = 5), y)
print(mae) #-------> 1.5724519510774606
print(best_param) #------> OrderedDict([('regressor__colsample_bytree', 0.8), ('regressor__learning_rate', 0.1), ('regressor__max_depth', 6), ('regressor__min_child_samples', 1), ('regressor__n_estimators', 1000), ('regressor__num_leaves', 63), ('regressor__reg_alpha', 0), ('regressor__reg_lambda', 1000), ('regressor__subsample', 0.8)])

### Output Manipulation
# Rounding
def Reg_for_Cla(y):
    y = y.round()
    for i in range(y.shape[0]):
        for j in range (y.shape[1]):
            if y[i][j] < 0:
                y[i][j] = 0
            elif y[i][j] > 9:
                y[i][j] = 9
    return y

# Decision Stump
def Stump(y_reg, y_cla):
    n = y_reg.shape[0]
    y_reg = y_reg.reshape((n,1))
    y_cla = y_cla.reshape((n,1))
    y = np.concatenate((y_reg, y_cla), axis=1)
    y = y[y[:, 0].argsort()]
    thr = np.empty(9)
    for t in range(9):
        y2 = np.copy(y)
        for i in range (n):
            if y[i][1] <= t:
                y2[i][1] = -1
            else:
                y2[i][1] = 1
        gtrain = np.zeros((n))
        for i in range(n):
            if y2[i][1] == -1:
                gtrain[0] += 1
        for i in range(1, n):
            if y2[i-1][1] == -1:
                gtrain[i] = gtrain[i-1] - 1
            else:                  
                gtrain[i] = gtrain[i-1] + 1
        ming = 0
        for i in range(n):
            if gtrain[i] < gtrain[ming]:
                ming = i
        if ming == 0:  
            g = -1
        else:
            g = (y[ming][0] + y[ming-1][0]) / 2
        thr[t] = g
        #print(str(t)+': '+str(gtrain[ming]/n))
    return thr

def Stump_Apply(y, stump):
    y_pred = np.copy(y)
    for i in range(y_pred.shape[0]):
        if y_pred[i] < stump[0]:
            y_pred[i] = 0
        elif stump[8] < y_pred[i]:
            y_pred[i] = 9
        else:
            for t in range (1, 9):
                if stump[t-1] < y_pred[i] <= stump[t]:
                    y_pred[i] = t
    return y_pred

# Stump()+Stump_Apply()
def Stump_Set(y_test, y_train_pred, y_train_true):
    stump = Stump(y_train_pred, y_train_true)
    y_test_new = Stump_Apply(y_test, stump)
    return y_test_new

id = test_df[['id']].copy()
id = id.to_numpy()
submit = np.zeros((n_test, 2))
submit[:, 0] = id[:, 0]

y_train_pred = best_model.predict(x_train.to_numpy())
y_pred = best_model.predict(x_test.to_numpy())
y_pred_stump = Stump_Set(y_pred, y_train_pred, y.to_numpy())
submit[:, 1] = y_pred_stump
df = pd.DataFrame(submit, columns = ['id','Danceability'])
df = df.astype({"id": int})
df.to_csv('submission.csv', index=False)