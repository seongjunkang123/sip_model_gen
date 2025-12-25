import pandas as pd

# loading dataset (csv files)
asthma_data = pd.read_csv('../sip_data/res1/Asthma_peaktable_ver3.csv', index_col='pubchem_CID')
bronchi_data = pd.read_csv('../sip_data/res1/Bronchi_peaktable_ver3.csv', index_col='pubchem_CID')
copd_data = pd.read_csv('../sip_data/res1/COPD_peaktable_ver3.csv', index_col='pubchem_CID')

# sort by their chemical IDs (use index)
asthma_sorted_data = asthma_data.sort_index()
bronchi_sorted_data = bronchi_data.sort_index()
copd_sorted_data = copd_data.sort_index()

# load intersection data
intersection = pd.read_excel('../sip_data/res1/intersection_of_detected_compunds.xlsx', index_col='pubchem_CID')

# remove duplicated index entries (keep first) to avoid drop issues, then keep only rows in the intersection
asthma_sorted_data = asthma_sorted_data[~asthma_sorted_data.index.duplicated(keep='first')]
bronchi_sorted_data = bronchi_sorted_data[~bronchi_sorted_data.index.duplicated(keep='first')]
copd_sorted_data = copd_sorted_data[~copd_sorted_data.index.duplicated(keep='first')]

common_idx = intersection.index
asthma_sorted_data = asthma_sorted_data[asthma_sorted_data.index.isin(common_idx)]
bronchi_sorted_data = bronchi_sorted_data[bronchi_sorted_data.index.isin(common_idx)]
copd_sorted_data = copd_sorted_data[copd_sorted_data.index.isin(common_idx)]

# combine dataframes
combined_data = pd.DataFrame(columns=['Patient ID', 'Disease'] + intersection.index.tolist())

for i in range(len(asthma_sorted_data.columns) - 1):
    combined_data.loc[len(combined_data)] = [len(combined_data) + 1, 'asthma'] + asthma_sorted_data[f"{i+1:02d}"].tolist()

for i in range(len(bronchi_sorted_data.columns) - 1):
    combined_data.loc[len(combined_data)] = [len(combined_data) + 1, 'bronchi'] + bronchi_sorted_data[f"{i+1}"].tolist()

for i in range(len(copd_sorted_data.columns) - 1):
    combined_data.loc[len(combined_data)] = [len(combined_data) + 1, 'copd'] + copd_sorted_data[f"{i+1}"].tolist()

combined_data.set_index('Patient ID', inplace=True)
print(combined_data)

# save to csv file
combined_data.to_csv('../sip_data/res1/combined_data.csv')