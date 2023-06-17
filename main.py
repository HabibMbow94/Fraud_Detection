import numpy as np;
import plotly.express as px #for visualization;
import matplotlib.pyplot as plt #for visualization ;
import matplotlib.pyplot as plt;
import seaborn as sns;

from EDA import load_data;
from Model_fit_evaluation import model_eval, params;
# from EDA.replace_na import replace_median;
# from EDA.correlation import corr;

# path data
path_data = 'data/transaction_dataset.csv'
datafraud = load_data.transaction_data(path_data, 0.8, 0.1, 0.1)

# Get the six first lignes
# print(datafraud.dataset.head())


# Get overview of the data
# datafraud.dataoveriew()

#columns having non-null values
# print(datafraud.dataset.isnull().sum())


# #Find the percentage of null values in each column
# print(round((datafraud.dataset.isnull().sum()/len(datafraud.dataset.index))*100,2))

# #columns having object type values
# object_columns=(dataset.select_dtypes(include=['object'])).columns
# object_columns  

# dataset[' ERC20_most_rec_token_type'].value_counts()
# dataset[' ERC20 most sent token type'].value_counts()
# dataset[' ERC20 most sent token type'].value_counts()

# #Drop the columns ' ERC20 most sent token type' and ' ERC20_most_rec_token_type' as they have many null ,0,None values
# dataset.drop([' ERC20_most_rec_token_type',' ERC20 most sent token type'],axis=1,inplace=True)

# #Drop the columns as they have no values
# dataset.drop([' ERC20 avg time between sent tnx',' ERC20 avg time between rec tnx',' ERC20 avg time between rec 2 tnx',' ERC20 avg time between contract tnx',' ERC20 min val sent contract',' ERC20 max val sent contract',' ERC20 avg val sent contract'],axis=1,inplace=True)

# corr, upper = corr(dataset)
# threshold=0.7
# # Select columns with correlations above threshold
# to_drop = [column for column in upper.columns if (any(upper[column] > threshold) or any(upper[column] < -(threshold)))]

# print('There are %d columns to remove.' % (len(to_drop)))


#  #drop = ['total transactions (including tnx to create contract','max val sent to contract',' ERC20 avg val rec',' ERC20 max val rec', ' ERC20 avg val sent',
# # ' ERC20 min val sent', ' ERC20 max val sent',' ERC20 uniq sent token name',' ERC20 uniq sent token name',' ERC20 uniq rec token name','max val sent to contract','avg value sent to contract']
# dataset.drop(to_drop, axis=1, inplace=True)


# #Heatmap of the numerical values
# # Correlation matrix
# print(datafraud.corr)
# datafraud.plot_corr()

######## ----------------------------------------- Split datasets ------------------------------------------------- #####
# splitting into  train, test ,validation datasets
X_train, X_val, X_test, y_train, y_val, y_test=datafraud.train_val_test_split()
# Inspecting the train and validation datasets
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape,X_val.shape,y_val.shape)
####### ------------------------------------------ ############### ------------------------------------------------- #####


####### ------------------------------------------ Feature Selection ----------------------------------------------- #####
feat_importances, col_x = features_importance = datafraud.importance_feature(X_train, y_train)
# #list of 18 important features
# col_x=features_importance.nlargest(18).index
# print("column", col_x)
X_train=X_train[col_x]
X_val=X_val[col_x]
X_test=X_test[col_x]
# print(feat_importances)
# print(X_train.info())
####### ------------------------------------------ ############### -------------------------------------------------- #####


####### ------------------------------------------  Feature Scaling ----------------------------------------------- #####
## Step 4: Feature Scaling
X_train, X_val, X_test = datafraud.feature_scaling(X_train, X_val, X_test, col_x)
# print(X_train.head())
####### ------------------------------------------ ############### -------------------------------------------------- #####


######## --------------------------------- Data Imbalance handling ------------------------------------------------------ #####
# Address imbalance using no sampling
X_train, y_train = datafraud.data_imbalance(X_train, y_train, "no sampling")
# Address imbalance using under sampling
X_train_us, y_train_us = datafraud.data_imbalance(X_train, y_train, "under sampling")
# Address imbalance using over sampling
X_train_ro, y_train_ro = datafraud.data_imbalance(X_train, y_train, "over sampling")
# Address imbalnce using SMOTE
X_train_sm, y_train_sm = datafraud.data_imbalance(X_train, y_train, "SMOTE")
# Address imbalnce using ADASYN
X_train_ada, y_train_ada = datafraud.data_imbalance(X_train, y_train, "ADASYN") 
######## --------------------------------- ################## ------------------------------------------------------------ #######


###### ----------------------------------- Training and Evaluation models ------------------------------------------------- ######
model_eval = model_eval.models(random_state=23)
##### -------------------------------------- ######################### ---------------------------------------------------- ######

###### ----------------------------------- Logistic Regression ------------------------------------------------- ##################
#Logistic Regression without sampling
# model_eval.model_fit_evaluation(params.params_LR, X_train, y_train, X_test, y_test, 'Logistic Regression', 'no sampling')
# # #Logistic Regression with under sampling
# model_eval.model_fit_evaluation(params.params_LR, X_train_us, y_train_us, X_test, y_test, 'Logistic Regression', 'under sampling')
# # #Logistic Regression with over sampling
model_eval.model_fit_evaluation(params.params_LR, X_train_ro, y_train_ro, X_test, y_test, 'Logistic Regression', 'over sampling')
# # #Logistic Regression with SMOTE
# model_eval.model_fit_evaluation( params.params_LR, X_train_sm, y_train_sm, X_val, y_val, 'Logistic Regression', 'SMOTE')
# #Logistic Regression with ADASYN
# model_eval.model_fit_evaluation(params.params_LR, X_train_ada, y_train_ada, X_val, y_val, 'Logistic Regression', 'ADASYN')
######### -------------------------------------- FIN ---------------------------------------------------------------------- ######


# ###### ------------------------------------- Decision Tree ------------------------------------------------------------------ ######
# #Decision Tree without  sampling
# model_eval.model_fit_evaluation(params.params_DT, X_train, y_train, X_test, y_test, 'Decision Tree', 'no sampling')
# #Decision Tree with under sampling
# model_eval.model_fit_evaluation(params.params_DT, X_train_us, y_train_us, X_test, y_test, 'Decision Tree', 'under sampling')
# #Decision Tree with over sampling
# model_eval.model_fit_evaluation(params.params_DT, X_train_ro, y_train_ro,  X_val, y_val, 'Decision Tree', 'over sampling')
# #Decision Tree with SMOTE
# model_eval.model_fit_evaluation(params.params_DT, X_train_ro, y_train_ro,  X_val, y_val, 'Decision Tree', 'SMOTE')
# #Decision Tree with ADASYN
# model_eval.model_fit_evaluation(params.params_DT, X_train_ro, y_train_ro,  X_val, y_val, 'Decision Tree', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Random Forest ---------------------------------------------------------------- ####
# #Random Forest without sampling
# model_eval.model_fit_evaluation(params.params_RF, X_train, y_train, X_test, y_test, 'Random Forest', 'no sampling')
# #Random Forest with under sampling
# model_eval.model_fit_evaluation(params.params_RF, X_train_us, y_train_us, X_test, y_test, 'Random Forest', 'under sampling')
# #Random Forest with over sampling
# model_eval.model_fit_evaluation(params.params_RF, X_train_ro, y_train_ro,  X_val, y_val, 'Random Forest', 'over sampling')
# #Random Forest with SMOTE
# model_eval.model_fit_evaluation(params.params_RF, X_train_sm, y_train_sm,  X_val, y_val, 'Random Forest', 'SMOTE')
# #Random Forest with ADASYAN
# model_eval.model_fit_evaluation(params.params_RF, X_train_ada, y_train_ada,  X_val, y_val, 'Random Forest', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- XG Boosting ---------------------------------------------------------------- ####
# # XGBoost  without  sampling
# model_eval.model_fit_evaluation(params.params_XGB, X_train, y_train, X_test, y_test, 'Xgboosting', 'no sampling')
# # XGBoost with under sampling
# model_eval.model_fit_evaluation(params.params_XGB, X_train_us, y_train_us, X_test, y_test, 'Xgboosting', 'Under sampling')
# # XGBoost with over sampling
# model_eval.model_fit_evaluation(params.params_XGB, X_train_us, y_train_us, X_test, y_test, 'Xgboosting', 'over sampling')
# # XGBoost with SMOTE
# model_eval.model_fit_evaluation(params.params_XGB, X_train_sm, y_train_sm,  X_val, y_val, 'Xgboosting', 'SMOTE')
# # XGBoost with ADASYAN
# model_eval.model_fit_evaluation(params.params_XGB, X_train_ada, y_train_ada, X_val, y_val, 'Xgboosting', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- ADA Boosting ------------------------------------------------------------------ ####
# # Adaboosting without sampling
# model_eval.model_fit_evaluation(params.params_ada, X_train, y_train, X_test, y_test, 'Adaboosting', 'no sampling')
# # Adaboosting with under sampling
# model_eval.model_fit_evaluation(params.params_ada, X_train_us, y_train_us, X_test, y_test, 'Adaboosting', 'under sampling')
# # Adaboosting with over sampling
# model_eval.model_fit_evaluation(params.params_ada, X_train_ro, y_train_ro, X_val, y_val, 'Adaboosting', 'over sampling')
# # Adaboosting with SMOTE
# model_eval.model_fit_evaluation(params.params_ada, X_train_sm, y_train_sm, X_val, y_val, 'Adaboosting', 'SMOTE')
# # Adaboosting with ADASYAN
# model_eval.model_fit_evaluation(params.params_ada, X_train_ada, y_train_ada, X_val, y_val, 'Adaboosting', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Gradient Boosting ------------------------------------------------------------ ####
# # Gradient Boosting without sampling
# model_eval.model_fit_evaluation(params.params_gb, X_train, y_train,X_test, y_test, 'Gradient Boosting', 'no sampling')
# # Gradient Boosting with under sampling
# model_eval.model_fit_evaluation(params.params_gb, X_train_us, y_train_us,X_test, y_test, 'Gradient Boosting', 'under sampling')
# # Gradient Boosting with over sampling
# model_eval.model_fit_evaluation(params.params_gb, X_train_ro, y_train_ro, X_val, y_val, 'Gradient Boosting', 'over sampling')
# # Gradient Boosting with SMOTE
# model_eval.model_fit_evaluation(params.params_gb, X_train_sm, y_train_sm, X_val, y_val, 'Gradient Boosting', 'SMOTE')
# # gradient Boosting with ADASYN
# model_eval.model_fit_evaluation(params.params_gb, X_train_ada, y_train_ada, X_val, y_val, 'Gradient Boosting', 'ADASYN')
# model_eval.get_params().keys()
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Light Gradient Boosting ------------------------------------------------------- ####
# # Light Gradient Boosting without no sampling
# model_eval.model_fit_evaluation(params.params_lgbm, X_train, y_train, X_val, y_val, 'Light Gradient Boosting', 'no sampling')
# # Light Gradient Boosting with under sampling
# model_eval.model_fit_evaluation(params.params_lgbm, X_train_us, y_train_us,X_test, y_test, 'Light Gradient Boosting', 'under sampling')
# # Light Gradient Boosting with over sampling
# model_eval.model_fit_evaluation(params.params_lgbm, X_train_ro, y_train_ro, X_val, y_val, 'Light Gradient Boosting', 'over sampling')
# # Light Gradient Boosting with SMOTE
# model_eval.model_fit_evaluation(params.params_lgbm, X_train_sm, y_train_sm, X_val, y_val, 'Light Gradient Boosting', 'SMOTE')
# # Light Gradient Boosting with ADASYN
# model_eval.model_fit_evaluation(params.params_lgbm, X_train_ada, y_train_ada, X_val, y_val, 'Light Gradient Boosting', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Cat Boosting ----------------------------------------------------------------- ####
# # Cat Boosting without no sampling
# model_eval.model_fit_evaluation(params.params_cat, X_train, y_train, X_val, y_val, 'Cat Boosting', 'no sampling')
# # Cat Boosting with under sampling
# model_eval.model_fit_evaluation(params.params_cat, X_train_us, y_train_us,X_test, y_test, 'Cat Boosting', 'under sampling')
# # Cat Boosting with over sampling
# model_eval.model_fit_evaluation(params.params_cat, X_train_ro, y_train_ro, X_val, y_val, 'Cat Boosting', 'over sampling')
# # Cat Boosting with SMOTE
# model_eval.model_fit_evaluation(params.params_cat, X_train_sm, y_train_sm, X_val, y_val, 'Cat Boosting', 'SMOTE')
# # Cat Boosting with ADASYN
# model_eval.model_fit_evaluation(params.params_cat, X_train_ada, y_train_ada, X_val, y_val, 'Cat Boosting', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Support Vector Machine ------------------------------------------------------- ####
# # SVM without sampling
# model_eval.model_fit_evaluation(params.params_svc, X_train, y_train, X_val, y_val, 'SVM', 'no sampling')
# # SVM with under Sampling
# model_eval.model_fit_evaluation(params.params_svc, X_train_us, y_train_us,X_test, y_test, 'SVM', 'under sampling')
# # SVM with over sampling
# model_eval.model_fit_evaluation(params.params_svc, X_train_ro, y_train_ro, X_val, y_val, 'SVM', 'over sampling')
# # SVM with SMOTE
# model_eval.model_fit_evaluation(params.params_svc, X_train_sm, y_train_sm, X_val, y_val, 'SVM', 'SMOTE')
# # SVM with ADSYN
# model_eval.model_fit_evaluation(params.params_svc, X_train_ada, y_train_ada, X_val, y_val, 'SVM', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- K-Nearest Neighbors ---------------------------------------------------------- ####
# # KNN without sampling
# model_eval.model_fit_evaluation(params.params_knn, X_train, y_train, X_val, y_val, 'KNN', 'no sampling')
# # KNN with under sampling
# model_eval.model_fit_evaluation(params.params_knn, X_train_us, y_train_us,X_test, y_test, 'KNN', 'under sampling')
# # KNN with over sampling
# model_eval.model_fit_evaluation(params.params_knn, X_train_ro, y_train_ro, X_val, y_val, 'KNN', 'over sampling')
# # KNN with SMOTE
# model_eval.model_fit_evaluation(params.params_knn, X_train_sm, y_train_sm, X_val, y_val, 'KNN', 'SMOTE')
# # KNN with ADASYN
# model_eval.model_fit_evaluation(params.params_knn, X_train_ada, y_train_ada, X_val, y_val, 'KNN', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####


# ######### -------------------------------------- Gaussian Naive Bayes ------------------------------------------------------- ####
# # Gaussian Naive Bayes without sampling
# model_eval.model_fit_evaluation(params.params_gnb, X_train, y_train, X_val, y_val, 'Gaussian Naive Bayes', 'no sampling')
# # Gaussian Naive Bayes with under sampling
# model_eval.model_fit_evaluation(params.params_gnb, X_train_us, y_train_us,X_test, y_test, 'Gaussian Naive Bayes', 'under sampling')
# # Gaussian Naive Bayes with over sampling
# model_eval.model_fit_evaluation(params.params_gnb, X_train_ro, y_train_ro, X_val, y_val, 'Gaussian Naive Bayes', 'over sampling')
# # Gaussian Naive Bayes with SMOTE
# model_eval.model_fit_evaluation(params.params_gnb, X_train_sm, y_train_sm, X_val, y_val, 'Gaussian Naive Bayes', 'SMOTE')
# # Gaussian Naive Bayes with ADASYN
# model_eval.model_fit_evaluation(params.params_gnb, X_train_ada, y_train_ada, X_val, y_val, 'Gaussian Naive Bayes', 'ADASYN')
# ######### -------------------------------------- FIN -------------------------------------------------------------------------- ####