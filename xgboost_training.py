import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict, GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, make_scorer
from skopt import BayesSearchCV
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

#import data and select only the numeric columns
df = pd.read_csv("C:\\Users\\user\\Downloads\html2023-spring-final-project\\train.csv", sep = ',')
y = df['Danceability']
encoded_df = pd.get_dummies(df, columns=['Artist', 'Composer'])
x_train = encoded_df.drop(['Danceability','Album_type','Album','Licensed','official_video','id','Track','Uri','Url_spotify','Url_youtube','Description','Title','Channel'], axis =1)



#param_space for bayesSearch
param_space = {
    'learning_rate': (0.01,0.06,'uniform'),
    'max_depth': (7,15),
    'n_estimators': (800,2000),
    'min_child_weight':(10,30), # the smaller, the more easy to overfit
    'gamma': (0,0.0001), #loss reduction
    'lambda': (0.01,0.4) #l2 regularizer
}
model = xgb.XGBRegressor(eval_metrix = 'mae')
#consruct bayesSearch object
opt = BayesSearchCV(estimator = model, search_spaces = param_space, n_iter = 20, cv = 5, scoring='neg_mean_absolute_error', random_state=1)
opt.fit(x_train, y)
best_model = opt.best_estimator_
best_param = opt.best_params_
y_pred = cross_val_predict(best_model, x_train, y, cv = 5)
y_pred_int = y_pred.round()
mae = mean_absolute_error(y_pred_int, y)
print(mae) -------> 1.577810133954572
print(best_param) ------> OrderedDict([('gamma', 0.0001), ('lambda', 0.09720470800367997), ('learning_rate', 0.02033464927013854), ('max_depth', 10), ('min_child_weight', 20), ('n_estimators', 1195)])

# Save the trained model to a file
joblib.dump(best_model, 'best_model.joblib')
