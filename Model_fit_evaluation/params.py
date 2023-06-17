import numpy as np

#Logistic regression parameters
params_LR = {'C':np.logspace(-1, 5, 10), 'class_weight':[None,'balanced'], 'penalty':['l1','l2']}

#Decision tree parameters
params_DT = {
    'max_depth': [10, 20, 50, 100, 200],
    'min_samples_leaf': [10, 20, 50, 100, 200],
    'min_samples_split' : [10, 20, 50, 100, 200],
    'criterion': ["gini", "entropy"]
} 

#Random forest parameters
params_RF = {    
    'n_estimators': [10,12,15],
    'max_features':['sqrt',0.3],
    'max_depth': [10,50],
    'min_samples_leaf': [50,200],
    'min_samples_split' : [50,100],
    'criterion': ["gini"]
    
}


#XGB parameters
params_XGB={
  'learning_rate':[0.01,0.1,0.3,0.5,0.7],
  'max_depth':[2,3,4,10],
  'n_estimators':[10,15,20,50,100,200],
  'subsample':[0.3, 0.5, 0.9],
  'colsample_bytree':[0.3,0.5,0.7],
  'max_features':[8,10,14,16]
}

#Adaboost parameters
params_ada={
  'learning_rate':[0.0001, 0.01, 0.1, 1.0, 1.1, 1.2,0.3,0.5,0.7],
  'n_estimators':[2,5,8,10,15,20,50]
}

#Gradient boosting parameters
params_gb={
  'learning_rate':[0.0001, 0.01, 0.1, 1.0, 1.1, 1.2,0.3,0.5,0.7],
  'n_estimators':[2,5,8,10,15,20,50,100]
}


#Light Gradient boosting parameters
params_lgbm={
  'boosting_type':['gbdt','dart','rf'],
  'learning_rate':[0.0001, 0.01, 0.1, 1.0, 1.1, 1.2,0.3,0.5,0.7],
  'n_estimators':[2,5,8,10,15,20,50,100,200],
  'subsample':[0.3, 0.5, 0.9],
  'max_depth':[-1,2,3,4,5,10],
  'colsample_bytree':[0.3,0.5,0.7,1.]
}

#Catboost parameters
params_cat={
  'boosting_type':["Ordered","Plain"],
  'iterations':[100,200] ,
  'learning_rate':[0.0001, 0.01, 0.1, 1.0, 1.1, 1.2,0.3,0.5,0.7],
  'loss_function':['RMSE','Logloss','MAE','CrossEntropy','MAPE'],
  'subsample':[0.3, 0.5, 0.9],
  'depth':[-1,2,3,4,5,10]
}

#SVM parameters
params_svc = {'C':[i for i in range(1,10,1)],'kernel':['linear','rbf','poly']}

#KNN parameters
params_knn = {'n_neighbors':[i for i in range(1,25,1)],'algorithm':['kd_tree','auto'],'n_jobs':[-1]}

#Naive bayes params
params_gnb={}