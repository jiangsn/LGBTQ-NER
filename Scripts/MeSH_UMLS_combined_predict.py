# %% 
# Load libraries
#
import pandas as pd
from collections import Counter
import spacy

# load the CPU-trained model / make sure use the right model!
nlp_output = spacy.load("./NER/model_all_expand/model-best")


# %%
### read in data
df_merge = pd.read_csv('./MeSH/mesh_umls_merged.csv').fillna('')
df_merge['NER_tag'] = ''
df_merge


# %% NER prediction
###
for idx, row in df_merge.iterrows():
    doc = nlp_output(row['text'])
    if doc is not None:
        ents = [e.label_ for e in doc.ents]
        df_merge.loc[idx,'NER_tag'] = ','.join(ents)

df_merge.to_csv('./MeSH/mesh_umls_merged_NER_tag.csv', index=False)
df_merge.head(20)




# %% 
################# testing
df_merge[df_merge['text'].str.contains('abnormal')]



# %%
