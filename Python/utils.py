import pandas as pd
import os

path = "../data"
schema = pd.read_excel(os.path.join(path, "schema.xlsx"), 1, engine='openpyxl')
schema.loc[:,'VARIABLE_NAME'] = schema.loc[:,'VARIABLE_NAME'].str.lower()

# Matches the file name with the table name in the schema
#	{filename : table_name}
table_reference = pd.Series(schema.loc[:,'table'].values, index=schema.loc[:,'.csv']).to_dict()
df_name_reference = pd.Series(schema.loc[:,'space'].values, index=schema.loc[:,'.csv']).to_dict() #WIP

# Matches variable names to their translations for each table, respectively (as tables can share variable names but have different meanings)
# 	{table_name: {variable_name : translation}}
translation_reference = schema.groupby('TABLE')[['VARIABLE_NAME','TRANSLATION']].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict() # Matches 

# Read all CSV's in the directory denoted by <path>
#	{filename : pandas.DataFrame}
dfs = dict([(csv, pd.read_csv( os.path.join(path, csv) ).rename(columns = translation_reference.get(table_reference.get(csv)))) for csv in os.listdir(path) if csv.endswith('.csv')])

for key,value in dfs.items():
    if key.endswith('.csv'):
        dfs[df_name_reference[key]] = dfs.pop(key)