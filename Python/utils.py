import pandas as pd
import os

_path = "../data"
_schema = pd.read_excel(os.path.join(_path, "schema.xlsx"), 1, engine='openpyxl')
_schema.loc[:,'VARIABLE_NAME'] = _schema.loc[:,'VARIABLE_NAME'].str.lower()

# Matches the file name with the table name in the schema
#	{filename : table_name}
_table_reference = pd.Series(_schema.loc[:,'table'].values, index=_schema.loc[:,'.csv']).to_dict()
_df_name_reference = pd.Series(_schema.loc[:,'space'].values, index=_schema.loc[:,'.csv']).to_dict() #WIP

# Matches variable names to their translations for each table, respectively (as tables can share variable names but have different meanings)
# 	{table_name: {variable_name : translation}}
_translation_reference = _schema.groupby('TABLE')[['VARIABLE_NAME','TRANSLATION']].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict() # Matches 

# Read all CSV's in the directory denoted by <path>
#	{filename : pandas.DataFrame}
dfs = dict([(_df_name_reference.get(csv), pd.read_csv( os.path.join(_path, csv) ).rename(columns = _translation_reference.get(_table_reference.get(csv)))) for csv in os.listdir(_path) if csv.endswith('.csv')])
