from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#XGBoost
import xgboost as xgb
#LGBM
from lightgbm import LGBMClassifier
#Catboost
from catboost import CatBoostClassifier, Pool
#SVM
from sklearn.svm import SVC
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


class models():
    def __init__(self, random_state):
        self.random_state = random_state
    
   # Function to draw ROC curve    
    def roc_curve(self, actual, probs ):
        fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                                drop_intermediate = False )
        auc_score = metrics.roc_auc_score( actual, probs )
        plt.figure(figsize=(5, 5))
        plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        return None
    
    
    # Function to return various standard metrices for a model
    def model_metrics(self, r,a, p):
        confusion = confusion_matrix(a, p)
        TP = confusion[1,1] # true positive 
        TN = confusion[0,0] # true negatives
        FP = confusion[0,1] # false positives
        FN = confusion[1,0] # false negatives
        print ('Accuracy    : ', metrics.accuracy_score(a, p ))
        print ('Sensitivity : ', TP / float(TP+FN))
        print ('Specificity : ', TN / float(TN+FP))
        print ('Precision   : ', TP / float(TP + FP))
        print ('Recall      : ', TP / float(TP + FN))
        print('F1_score:',metrics.f1_score(a,p))
        print(confusion)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                    display_labels=r.classes_)
        disp.plot()
        plt.grid(False)
        plt.show()
        return None 
    
    def model_fit_evaluation(self, params, X_train, y_train, X_val, y_val, algo=None, sampling=None):
        #logistic regression
        if algo == "Logistic Regression":
            model = LogisticRegression()
        # Decision Tree
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier(random_state = 23)
        # Random Forest
        elif algo == "Random Forest":
            model = RandomForestClassifier(oob_score = True, random_state=23)
        # XGBoost: eXtreme Gradient Boosting
        elif algo == "XG Boosting":
            model = xgb.XGBClassifier()
        # ADA Boosting 
        elif algo == "ADA Boosting":
            model = AdaBoostClassifier()
        # Gradient Boosting
        elif algo == "Gradient Boosting":
            model = GradientBoostingClassifier()
        # Light Gradient Boosting
        elif algo == "Light Gradient Boosting":
            model = LGBMClassifier()
        # Cat Boosting
        elif algo == "Cat Boosting":
            model = CatBoostClassifier()
        # Support Vector Machine
        elif algo == "SVM":
            model = SVC()
        # K-Nearest Neighbors
        elif algo == "KNN":
            model = KNeighborsClassifier()
        # Gaussian Naive Bayes
        else:
            model = GaussianNB()   
            
        start_time = time.time()
        
        rcv = RandomizedSearchCV(model, params, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=23)
        rcv.fit(X_train, y_train)
        
        print('\n')
        print('best estimator : ', rcv.best_estimator_)
        print('best parameters: ', rcv.best_params_)
        print('best score: ', rcv.best_score_)
        print('\n')
        y_train_pred= (rcv.best_estimator_).predict(X_train)
        y_val_pred= (rcv.best_estimator_).predict(X_val)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.roc_curve(y_train, y_train_pred)
        print("Training set metrics")
        print ('AUC for the {} Model {} sampling technique'.format(algo,sampling), metrics.roc_auc_score( y_train, y_train_pred))
        self.model_metrics(rcv,y_train, y_train_pred)
        print('*'*50)
        print("Validation set metrics")
        self.roc_curve(y_val, y_val_pred)
        print ('AUC for the {} Model {} sampling technique'.format(algo,sampling), metrics.roc_auc_score( y_val, y_val_pred))
        self.model_metrics(rcv,y_val, y_val_pred)