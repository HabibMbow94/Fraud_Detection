
import numpy as np
# Replace missings of numerical variables with median
def replace_median(dataset):
     #cleaning the categorical feature - changing 0 values to null, cause a 0 value doesnt mean anything in categorical features
    dataset.fillna(dataset.median(), inplace=True)
    dataset[' ERC20_most_rec_token_type'].replace({'0':np.NaN},inplace = True)
    dataset[' ERC20 most sent token type'].replace({'0':np.NaN},inplace = True)
    