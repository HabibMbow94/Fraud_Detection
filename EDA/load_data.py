import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split

#Information Gain code snippet
from sklearn.feature_selection import mutual_info_classif

# feature scaling
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

# Address imbalance using under sampling
from imblearn import under_sampling,  over_sampling

class transaction_data():
    def __init__(self, path_data, train_size, val_size, test_size) -> None:
        self.path_data = path_data
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        

        self.load_data()
        self.replace_median()
        self.drop_columns()
        
        self.corr = self.correlation()
        self.drop_columns_high_corr()
        # self.box_plot()

    #Get overview of the data
    def load_data(self, ):
        self.dataset=pd.read_csv(self.path_data, index_col='Index')
        
        return self.dataset
    
    def dataoveriew(self,):
        print('Data overview:\n')
        print('\nNombre de lignes: ', self.dataset.shape[0])
        print("\nNombre de variables", self.dataset.shape[1])
        print("\nNom des variables:")
        print(self.dataset.columns.tolist())
        print("\nValeurs manquantes:", self.dataset.isnull().sum().values.sum())
        print("\nValeurs uniques:")
        print(self.dataset.nunique())
    
    # Replace the NA for the median
    def replace_median(self, ):
         #cleaning the categorical feature - changing 0 values to null, cause a 0 value doesnt mean anything in categorical features
        # Replace missings of numerical variables with median
        replace = self.dataset.fillna(self.dataset.median(numeric_only=True), inplace=True)
        #cleaning the categorical feature - changing 0 values to null, cause a 0 value doesnt mean anything in categorical features
        replace =  self.dataset[' ERC20_most_rec_token_type'].replace({'0':np.NaN},inplace = True)
        replace =  self.dataset[' ERC20 most sent token type'].replace({'0':np.NaN},inplace = True)
        # cleaning the categorical feature - changing 0 values to null, cause a 0 value doesnt mean anything in categorical features
        return replace
    
    def drop_columns(self, ):
        # drop the Unnamed: 0 column as it is not needed for further analysis
        # drop_columns = self.dataset.drop(axis=1)
        #Drop the columns ' ERC20 most sent token type' and ' ERC20_most_rec_token_type' as they have many null ,0,None values
        drop_columns = self.dataset.drop(['Unnamed: 0', ' ERC20_most_rec_token_type', ' ERC20 most sent token type'],axis=1,inplace=True)
        
        #Drop the columns as they have no values
        drop_columns = self.dataset.drop([' ERC20 avg time between sent tnx',' ERC20 avg time between rec tnx',' ERC20 avg time between rec 2 tnx',' ERC20 avg time between contract tnx',' ERC20 min val sent contract',' ERC20 max val sent contract',' ERC20 avg val sent contract'],axis=1,inplace=True)
        return drop_columns
    
    def correlation(self):
        
        # Correlation matrix
        corr = self.dataset.corr()
        return corr
    
    # def plot_corr(self, ):
        
    #     #Scatter plot differnt attributes
    #     plt.subplots(figsize = (8, 6))
    #     sns.set(style = 'darkgrid')
    #     sns.scatterplot(data = self.dataset,x = 'Unique Received From Addresses', y= 'Received Tnx',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.set(style = 'whitegrid')
    #     sns.scatterplot(data = self.dataset,x = 'Unique Sent To Addresses', y= 'Sent tnx',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.scatterplot(data = self.dataset,x = ' ERC20 uniq sent addr.1', y= ' Total ERC20 tnxs',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.scatterplot(data = self.dataset,x = 'Unique Received From Addresses', y= 'Received Tnx',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.scatterplot(data = self.dataset,x = 'Sent tnx', y= 'Unique Sent To Addresses',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.scatterplot(data = self.dataset,x = ' ERC20 uniq rec addr', y= ' Total ERC20 tnxs',hue = 'FLAG' )
    #     plt.show()


    #     plt.subplots(figsize = (8, 6))
    #     sns.scatterplot(data = self.dataset,x = 'total transactions (including tnx to create contract', y= 'Received Tnx',hue = 'FLAG' )
    #     plt.show()
        
        
        #Heatmap of the numerical values

        # mask = np.zeros_like(self.corr)
        # mask[np.triu_indices_from(mask)]=True
        # with sns.axes_style('white'):
        #     fig, ax = plt.subplots(figsize=(30,20))
        #     sns.heatmap(self.corr,  mask=mask, annot=True, cmap='RdYlGn', center=0, square=True,fmt='.2g')

    def drop_columns_high_corr(self, ):
        
        # Upper triangle of correlations
        upper = self.corr.where(np.triu(np.ones(self.corr.shape), k=1).astype(bool))
        
        threshold=0.7
        # Select columns with correlations above threshold
        to_drop = [column for column in upper.columns if (any(upper[column] > threshold) or any(upper[column] < -(threshold)))]

        # print('There are %d columns to remove.' % (len(to_drop)))
        
        drop_high_corr = self.dataset.drop(to_drop, axis=1)
        
        #Heatmap of the numerical values
        # # Correlation matrix
        # corre = drop_high_corr.corr()
    
        # mask = np.zeros_like(corre)
        # mask[np.triu_indices_from(mask)]=True
        # with sns.axes_style('white'):
        #     fig, ax = plt.subplots(figsize=(30,20))
        #     sns.heatmap(corre,  mask=mask, annot=True, cmap='RdYlGn', center=0, square=True,fmt='.2g')
        
        return drop_high_corr
    
    # Univariate Analysis
    #Function for plotting box plot on  variable
    # def box_plot(self, ):
    #     plt.figure(figsize=(6,4))
    #     # columns
    #     self.columns=self.dataset.columns
        
    #     # Univariate Analysis
    #     for col in self.columns[2:]:
    #         sns.boxplot(y=self.dataset[col])
    #         plt.title("Boxplot for {}".format(col))
    #         plt.show()

    # # Bivariate Analysis
    # #Function for plotting box plot on  variable wrt to FLAG
    # def box_plot_y(self, ):
    #     plt.figure(figsize=(6,4))
    #     for col in self.columns[2:]:
    #         sns.boxplot(y=self.dataset[col],x=self.dataset['FLAG'])
    #         plt.title("Boxplot for {} wrt Flag".format(col))
    #         plt.show()
            
    # #Histogram of various attributes
    # def histogram(self, ):
    #     self.dataset.hist(figsize=(30,25))
        
        
    # Split dataset
    def train_val_test_split(self, ):
        # Putting response variable to y  and Putting feature variable to X
        y = self.dataset['FLAG']
        X = self.dataset.drop(['FLAG','Address'],axis=1)
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = self.test_size)
        relative_train_size = self.train_size / (self.val_size + self.train_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                        train_size = relative_train_size, test_size = 1-relative_train_size)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def importance_feature(self, Xtrain, Ytrain):
        #Information Gain code snippet
        from sklearn.feature_selection import mutual_info_classif
        importance=mutual_info_classif(Xtrain,Ytrain)
        feat_importances=pd.Series(importance,Xtrain.columns[0:len(Xtrain.columns)])
        # plt.figure(figsize=[30,15])


        feat_importances = feat_importances.nlargest(18)
        #list of 18 important features
        col_x=feat_importances.nlargest(18).index
        
        # plt.show()
        
        return feat_importances, col_x
    
    #Function for plotting box plot on  variable
    def box_plot_trans(self, Xtrain, variable):
        plt.figure(figsize=(6,4))
        sns.boxplot(y=Xtrain[variable])
        plt.title("Boxplot for {}".format(variable))
        plt.show()

    #Function for plotting box plot on  variable wrt to FLAG
    def box_plot__trans_y(Xtrain, variable):
        plt.figure(figsize=(6,4))
        sns.boxplot(y=Xtrain[variable],x=Xtrain['FLAG'])
        plt.title("Boxplot for {} wrt Flag".format(variable))
        plt.show()
        
        
    ## Step 4: Feature Scaling
    def feature_scaling(self, Xtrain, Xval, Xtest, col_x):
        
        #Normalisation using power transformer
        scaler = PowerTransformer()

        Xtrain[col_x] = scaler.fit_transform(Xtrain[col_x])
        Xval[col_x] = scaler.transform(Xval[col_x])
        Xtest[col_x] = scaler.transform(Xtest[col_x])
        
        return Xtrain, Xval, Xtest
    
    
    # Data Imbalance Handling
    def data_imbalance(self, Xtrain, Ytrain, name):
        # Address imbalance using under sampling
        us = under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=100)
        # Address imbalance using over sampling
        ro = over_sampling.RandomOverSampler(sampling_strategy='minority',random_state=100)
        #Address imbalnce using SMOTE
        sm = over_sampling.SMOTE(sampling_strategy='minority',random_state=100)
        #Address imbalnce using ADASYN
        ada = over_sampling.ADASYN(sampling_strategy='minority',random_state=100)
        if name == "under sampling":
            X_train_im, y_train_im = us.fit_resample(Xtrain, Ytrain)
        elif name == "over sampling":
           X_train_im, y_train_im = ro.fit_resample(Xtrain, Ytrain) 
        elif name == "SMOTE":
            X_train_im, y_train_im = sm.fit_resample(Xtrain, Ytrain) 
        elif name == "ADASYN":
            X_train_im, y_train_im = ada.fit_resample(Xtrain, Ytrain) 
        else:
            X_train_im, y_train_im = Xtrain, Ytrain
        # print (X_train_im.shape)
        # print (y_train_im.shape)
        # print (y_train_im.value_counts())

        return X_train_im, y_train_im