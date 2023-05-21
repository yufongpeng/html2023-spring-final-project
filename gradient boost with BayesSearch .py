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
    'l2_regularizer':(0.01,0.1), #regularizer
}

# Create a BayesSearchCV object
optimizer = BayesSearchCV(HistGradientBoostingRegressor(loss = 'absolute_error'), param_space, n_iter=30, cv=5, scoring='neg_mean_absolute_error', random_state=26)

# Fit the BayesSearchCV object
optimizer.fit(x_train, y)

# Access the results
best_model = optimizer.best_estimator_
best_param = optimizer.best_params_

#estimate the error
y_predict = cross_val_predict(best_model, x_train, y, cv=5)
y_predict_int = y_predict.round()
mae = mean_absolute_error(y, y_predict_int)
print(mae)       #1.6837507280139778
print(best_param)     #OrderedDict([('l2_regularization', 0.08431100392200155), ('learning_rate', 0.0948966816697544), ('max_depth', 14), ('max_iter', 429), ('min_samples_leaf', 113)])

#without knn
#1.6718112987769365
#OrderedDict([('l2_regularization', 0.050797127375640606), ('learning_rate', 0.05117030450929176), ('max_depth', 20), ('max_iter', 747), ('min_samples_leaf', 140)])
