import pandas as pd
import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

from common.serialization import pickle_load

name = 'feat_imp1.csv'

def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, metric='auc'):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=metric, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X, y, eval_metric=metric)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.to_csv(name)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')


X_train = pickle_load('data/X_train.pkl')
y_train = pickle_load('data/y_train.pkl')
_ID_ = pickle_load('data/_ID_.pkl')

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=400,
 max_depth=5,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1,
 seed=27)

xgb1.fit(X_train, y_train, eval_metric='auc')
modelfit(xgb1, X_train, y_train)
name = 'feat_imp2.csv'

param_test1 = {
 'max_depth': [4, 5, 7, 9],
 'min_child_weight': [1, 3, 5]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=xgb1.get_xgb_params()['n_estimators'], max_depth=5,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                        objective= 'binary:logistic', scale_pos_weight=1, seed=27), 
                        param_grid = param_test1, scoring='roc_auc', n_jobs=1, iid=False, cv=10)
gsearch1.fit(X_train, y_train)
print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

param_test2 = {
    'gamma': [0.0, 0.1, 0.2, 0.3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=xgb1.get_xgb_params()['n_estimators'], 
                        max_depth=gsearch1.best_params_['max_depth'], min_child_weight=gsearch1.best_params_['min_child_weight'], 
                        gamma=0, subsample=0.8, colsample_bytree=0.8,
                        objective= 'binary:logistic', scale_pos_weight=1, seed=27), 
                        param_grid = param_test2, scoring='roc_auc', n_jobs=1, iid=False, cv=10)
gsearch2.fit(X_train, y_train)
print(gsearch2.grid_scores_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)


param_test3 = {
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=xgb1.get_xgb_params()['n_estimators'],
                        max_depth=gsearch1.best_params_['max_depth'], min_child_weight=gsearch1.best_params_['min_child_weight'],
                        gamma=gsearch2.best_params_['gamma'], subsample=0.8, colsample_bytree=0.8,
                        objective= 'binary:logistic', scale_pos_weight=1, seed=27),
                        param_grid = param_test3, scoring='roc_auc', n_jobs=1, iid=False, cv=10)
gsearch3.fit(X_train, y_train)
print(gsearch3.grid_scores_)
print(gsearch3.best_params_)
print(gsearch3.best_score_)


xgb2 = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=gsearch1.best_params_['max_depth'],
                    min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch2.best_params_['gamma'], subsample=0.8, 
                    colsample_bytree=gsearch3.best_params_['colsample_bytree'], objective= 'binary:logistic', scale_pos_weight=1, 
                    seed=27)
modelfit(xgb2, X_train, y_train)

print(xgb2.get_xgb_params())

xgb2.fit(X_train, y_train)
X_test = pickle_load('data/X_test.pkl')
answer_prob = xgb2.predict_proba(X_test)

an = pd.DataFrame({'_ID_': _ID_, '_VAL_': answer_prob[:, 1]})
an.to_csv('output_xgboost.csv', index=False)

