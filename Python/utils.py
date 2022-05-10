import pandas as pd
import numpy as np
import os
from global_variables import *

data_path = "../data"
_schema = pd.read_excel(os.path.join(data_path, "schema.xlsx"), 1, engine='openpyxl')
_schema.loc[:,'VARIABLE_NAME'] = _schema.loc[:,'VARIABLE_NAME'].str.upper()

# Matches the file name with the table name in the schema
#	{filename : table_name}
_table_reference = pd.Series(_schema.loc[:,'table'].values, index=_schema.loc[:,'.csv']).to_dict()
_df_name_reference = pd.Series(_schema.loc[:,'space'].values, index=_schema.loc[:,'.csv']).to_dict() #WIP

# Matches variable names to their translations for each table, respectively (as tables can share variable names but have different meanings)
# 	{table_name: {variable_name : translation}}
_translation_reference = _schema.groupby('TABLE')[['VARIABLE_NAME','TRANSLATION']].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict() # Matches 

# Read all CSV's in the directory denoted by <path>
#	{filename : pandas.DataFrame}
dfs = dict([(_df_name_reference.get(csv), pd.read_csv( os.path.join(data_path, csv) ).rename(columns = _translation_reference.get(_table_reference.get(csv)))) for csv in os.listdir(data_path) if csv.endswith('.txt')])


def progressbar(it, prefix="", size=60, file=sys.stdout):
    """ Create a visualization of a progress bar updates according to completion status
    :param it: job you are trying to create a progress bar for
    :type obj (sequence or collection)
    :param prefix: The text to display to the left of the status bar
    :type str
    :param size: total length of the progress bar
    :type int
    :param file: what to display/write the progress bar to
    :type output stream
    :return: job you are trying to create a progress bar for
    :rtype: obj (sequence or collection)
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()