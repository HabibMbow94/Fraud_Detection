import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

def corr(dataset):
    
    # Correlation matrix
    corr = dataset.corr()
    
    # Upper triangle of correlations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    
    #Scatter plot differnt attributes
    plt.subplots(figsize = (8, 6))
    sns.set(style = 'darkgrid')
    sns.scatterplot(data = dataset,x = 'Unique Received From Addresses', y= 'Received Tnx',hue = 'FLAG' )
    plt.show()


    plt.subplots(figsize = (8, 6))
    sns.set(style = 'whitegrid')
    sns.scatterplot(data = dataset,x = 'Unique Sent To Addresses', y= 'Sent tnx',hue = 'FLAG' )
    plt.show()



    plt.subplots(figsize = (8, 6))
    sns.scatterplot(data = dataset,x = ' ERC20 uniq sent addr', y= ' Total ERC20 tnxs',hue = 'FLAG' )
    plt.show()



    plt.subplots(figsize = (8, 6))
    sns.scatterplot(data = dataset,x = 'Unique Received From Addresses', y= 'Received Tnx',hue = 'FLAG' )
    plt.show()


    plt.subplots(figsize = (8, 6))
    sns.scatterplot(data = dataset,x = 'Sent tnx', y= 'Unique Sent To Addresses',hue = 'FLAG' )
    plt.show()


    plt.subplots(figsize = (8, 6))
    sns.scatterplot(data = dataset,x = ' ERC20 uniq rec addr', y= ' Total ERC20 tnxs',hue = 'FLAG' )
    plt.show()


    plt.subplots(figsize = (8, 6))
    sns.scatterplot(data = dataset,x = 'total transactions (including tnx to create contract', y= 'Received Tnx',hue = 'FLAG' )
    plt.show()
    
    #Heatmap of the numerical values

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)]=True
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(30,20))
        sns.heatmap(corr,  mask=mask, annot=True, cmap='RdYlGn', center=0, square=True,fmt='.2g')

   
    return corr, upper