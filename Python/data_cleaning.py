import pandas as pd
import matplotlib as plt
import numpy as np
from utils import dfs

for key,value in dfs.items():
    print(key)
    
    
#Check for NaN's    
for key, df in dfs.items():
    nas = df.isna().sum().replace(0,np.nan).dropna() / df.shape[0] * 100
    print(key, '\n\t', nas)

#Transactions
#Replace low NA % categorical with 'unknown'
dfs.get('transaction')['trans_spending_category'].fillna('unknown',inplace=True)
dfs.get('transaction')['transaction_type'].fillna('unknown',inplace=True)
#Dropping high NA % columns
dfs.get('transaction').drop(['merch_x_coord'], axis=1, inplace=True) 
dfs.get('transaction').drop(['merch_y_coord'], axis=1, inplace=True)

#Customer Demographics
#I'm not sure why fillna doens't work on this one, but this works.
median_val = np.nanmedian(dfs.get('customer_demog')['customer_income_level'])
dfs.get('customer_demog')['customer_income_level'] = dfs.get('customer_demog')['customer_income_level'].replace(np.nan,median_val)
#dfs.get('customer_demog')['customer_income_level'].fillna(median_val, inplace=True) #This doesn't work.










#print(dfs['customer_demog_1']['customer_income_level'].value_counts())

#dfs['customer_demog_1']['customer_income_level'].hist(bins=50,figsize=(20,15))
#save_fig("attribute_histogram_plots")
#plt.show()

