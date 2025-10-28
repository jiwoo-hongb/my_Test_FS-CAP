import pandas as pd
df = pd.read_csv("Bindingdb_All.tsv", sep='\t', on_bad_lines='skip', low_memory=False)
print(df['Target Name Assigned by Curator or DataSource'].unique()[:50])
