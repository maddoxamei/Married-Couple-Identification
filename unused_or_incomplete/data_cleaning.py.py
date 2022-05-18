#Transactions
#Replace low NA % categorical with 'unknown'
dfs.get('transaction')['trans_spending_category'].fillna('UNCATEGORIZED',inplace=True)
dfs.get('transaction')['transaction_type'].fillna('UNCATEGORIZED',inplace=True)
#Dropping high NA % columns
dfs.get('transaction').drop(['merch_x_coord'], axis=1, inplace=True) 
dfs.get('transaction').drop(['merch_y_coord'], axis=1, inplace=True)

median_val = np.nanmedian(dfs.get('customer_demog')['customer_income_level'])
dfs.get('customer_demog')['customer_income_level'] = dfs.get('customer_demog')['customer_income_level'].replace(np.nan,median_val)
