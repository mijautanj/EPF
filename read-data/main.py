import pandas as pd


NPdata = pd.read_excel("./readNordPoolData/NPdataSetLag.xlsx", verbose=True, engine='openpyxl', index_col=0)
SMHIdata = pd.read_excel("./readSMHIdata/SMHIdataSet.xlsx", verbose=True, engine='openpyxl', index_col=0)

df_merged = NPdata.merge(SMHIdata, how='outer', on=['DateTime'])

df_merged['DateTime'] = df_merged['DateTime'].dt.round("H")
df_merged.drop_duplicates(subset=['DateTime'], keep='first', inplace=True, ignore_index=False)


print(df_merged.info())
df_merged.to_excel('fullDataSetLag.xlsx', engine='xlsxwriter')

h5File = "fullDataSet.h5"
df_merged.to_hdf(h5File, "/data/d1")
