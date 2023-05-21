import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict, GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
import joblib

#import data and select only the numeric columns
df = pd.read_csv("C:\\Users\\user\\Downloads\html2023-spring-final-project\\train.csv", sep = ',')
y = df['Danceability']
x = df.drop(['Danceability','Artist','Album_type','Album','Licensed','official_video','id','Track','Uri','Url_spotify','Url_youtube','Description','Title','Channel','Composer'], axis =1)

#preprocess the data 
iqrs = x.apply(lambda x: np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))
x_scale = x/ iqrs
imputer = KNNImputer(n_neighbors=5)
x_train = imputer.fit_transform(x_scale)
x_train = x_train * iqrs.to_numpy()
for i in range(x_train.shape[0]):
    x_train[i][1] = round(x_train[i][1])


# Define the parameter distributions
param_space = {
    'max_iter': (100, 1000),  # Number of boosting stages
    'learning_rate': (0.02, 0.2, 'uniform'),  # Learning rate
    'max_depth': (1, 20),  # Maximum depth of each tree
    'min_samples_leaf': (1,200  ),  # Minimum number of samples required to split an internal node
}

# Create a BayesSearchCV object
optimizer = BayesSearchCV(HistGradientBoostingRegressor(loss = 'absolute_error'), param_space, n_iter=30, cv=5, scoring='neg_mean_absolute_error', random_state=26)

# Fit the BayesSearchCV object
optimizer.fit(x_train, y)

# Access the results
best_model = optimizer.best_estimator_
best_param = optimizer.best_params_

'''#testing histgradientboost with fixed parameter
model = HistGradientBoostingRegressor()
model.learning_rate = 0.1
model. max_depth = 10
model.min_samples_leaf = 90
model.n_estimators = 10
y_predict = cross_val_predict(model, x_train, y, cv=5)
y_predict_int = y_predict.round()
mae = mean_absolute_error(y, y_predict_int)
print(mae)'''

#estimate the error
y_predict = cross_val_predict(best_model, x_train, y, cv=5)
y_predict_int = y_predict.round()
mae = mean_absolute_error(y, y_predict_int)
print(mae)
print(best_param)