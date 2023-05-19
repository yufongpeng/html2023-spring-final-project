import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict
from scipy.stats import uniform
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error

#import data and select only the numeric columns
df = pd.read_csv("kaggle/html2023-spring-final-project/train.csv", sep = ',')
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
param_dist = {
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'subsample': uniform(0.6, 0.4),
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(),
    param_distributions=param_dist,
    scoring='neg_mean_absolute_error',
    n_iter=10,
    cv=5,
    random_state=42
)

# Fit the RandomizedSearchCV object
random_search.fit(x_train, y)

# Access the results
best_params = random_search.best_params_
best_score = random_search.best_score_
best_model = GradientBoostingRegressor(**best_params)

#estimate the error
y_predict = cross_val_predict(best_model, x_train, y, cv=5)
y_predict_int = y_predict.round()
mae = mean_absolute_error(y, y_predict_int)
print(mae)
------> 1.68
