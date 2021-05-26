import pandas as pd


NPdata = pd.read_excel("./readNordPoolData/NPdataSetLag.xlsx", verbose=True, engine='openpyxl', index_col=0)
SMHIdata = pd.read_excel("./readSMHIdata/SMHIdataSet.xlsx", verbose=True, engine='openpyxl', index_col=0)

#Merge NPdata and SMHIdata
df_merged = NPdata.merge(SMHIdata, how='outer', on=['DateTime'])

#Round all timestaps to hoursly values
df_merged['DateTime'] = df_merged['DateTime'].dt.round("H")

#Drop duplicate rows obtained by the change in time-zone
df_merged.drop_duplicates(subset=['DateTime'], keep='first', inplace=True, ignore_index=False)

#Print information about the obtained dataframe
print(df_merged.info())

#Save as excel for visualization and h5 format for faster read
df_merged.to_excel('fullDataSet.xlsx', engine='xlsxwriter')
df_merged.to_hdf("fullDataSet.h5", "/data/d1")
