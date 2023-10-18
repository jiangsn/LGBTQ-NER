# %%
### load libraries
import spacy
import pandas as pd
from collections import Counter


# %% Load in KB terms extracted from tobacco papers
df_mesh = pd.read_csv('./MeSH/alldata_by_mesh.csv').fillna('')
df_umls = pd.read_csv('./MeSH/alldata_by_umls.csv').fillna('')
print(len(df_mesh), len(df_umls))


# %% 
### 
df_mesh_cui = df_mesh[df_mesh['score']>0]
df_umls_cui = df_umls[df_umls['score']>0]
df_umls_cui


# %%
### Merge two databases
df_merge = pd.merge(df_mesh_cui, df_umls_cui, how='outer', on='text').fillna('')
df_merge


# %%
### Merge terms from two KBs: 1) higher score or 2) non-empty ones
df_merge_clean = df_merge[['text']]
df_merge_clean['kb'] = ''
df_merge_clean['cui'] = ''
df_merge_clean['score'] = 0
df_merge_clean['name'] = ''
df_merge_clean['aliases'] = ''
df_merge_clean = df_merge_clean[df_merge_clean['text']!='']


# %%
for idx, row in df_merge.iterrows():
    if row['score_x'] == '':
        df_merge_clean.loc[idx,'kb'] = 'umls'
        df_merge_clean.loc[idx,'cui'] = row['cui_y']
        df_merge_clean.loc[idx,'score'] = row['score_y']
        df_merge_clean.loc[idx,'name'] = row['name_y']
        df_merge_clean.loc[idx,'aliases'] = row['aliases_y']
    elif row['score_y'] == '':
        df_merge_clean.loc[idx,'kb'] = 'mesh'
        df_merge_clean.loc[idx,'cui'] = row['cui_x']
        df_merge_clean.loc[idx,'score'] = row['score_x']
        df_merge_clean.loc[idx,'name'] = row['name_x']
        df_merge_clean.loc[idx,'aliases'] = row['aliases_x']
    elif row['score_y'] > row['score_x']:
        df_merge_clean.loc[idx,'kb'] = 'umls'
        df_merge_clean.loc[idx,'cui'] = row['cui_y']
        df_merge_clean.loc[idx,'score'] = row['score_y']
        df_merge_clean.loc[idx,'name'] = row['name_y']
        df_merge_clean.loc[idx,'aliases'] = row['aliases_y']
    elif row['score_x'] > row['score_y']:
        df_merge_clean.loc[idx,'kb'] = 'mesh'
        df_merge_clean.loc[idx,'cui'] = row['cui_x']
        df_merge_clean.loc[idx,'score'] = row['score_x']
        df_merge_clean.loc[idx,'name'] = row['name_x']
        df_merge_clean.loc[idx,'aliases'] = row['aliases_x']
    else:
        df_merge_clean.loc[idx,'kb'] = 'mesh+umls'
        df_merge_clean.loc[idx,'cui'] = row['cui_x'] + ', ' + row['cui_y']
        df_merge_clean.loc[idx,'score'] = row['score_x']
        txt = row['name_x'] + ', ' + row['name_y']
        txt = ', '.join(list(Counter(txt.split(', '))))
        df_merge_clean.loc[idx,'name'] = txt
        txt = row['aliases_x'] + ', ' + row['aliases_y']
        txt = ', '.join(list(Counter(txt.split(', '))))
        df_merge_clean.loc[idx,'aliases'] = txt
 
df_merge_clean.reset_index(drop=True, inplace=True)
df_merge_clean = df_merge_clean.sort_values('text')
df_merge_clean = df_merge_clean[['text','aliases','kb','cui','score','name']]
df_merge_clean.to_csv('./MeSH/mesh_umls_merged.csv', index=False)
df_merge_clean




# %% 
################# testing
df_merge[df_merge['text'].str.contains('abnormal')]



# %%
