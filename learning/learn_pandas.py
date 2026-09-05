import pandas as pd

asthma_df = pd.read_csv('../../sip_data/res1/Asthma_peaktable_ver3.csv')
asthma_df.set_index('pubchem_CID', inplace=True)

print(asthma_df.index)