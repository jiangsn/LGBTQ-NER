# %%
import os
import os.path
import pandas as pd
from collections import Counter
import re
from collections import Counter

outPath = './TobaccoControl/NLP_results/'
tablesTC = 'TableFullRecordsFlattenUpd.csv'

search_keys = ['heterosexual', 'lgb', 'lesbian', 'gay', 'bisexual',
                'transgender', 'queer', 'sexual orientation',
                'sextual identity']


# %% ######################################################################################
#
# Select records/tables with key terms in them
#
with open(outPath + tablesTC, 'r') as f:
    lines = f.readlines()
records = []
for i in range(1, len(lines)):
    line = lines[i].lower()
    if any([(x in line) for x in search_keys]):
        records.append(i-1)
print('records containing keywords', len(records))


# %%
### get the whole tables containing the above keywords in the table
df_tc = pd.read_csv(outPath + tablesTC, encoding='utf-8').fillna('')
df_table_list = df_tc.iloc[records][['paperLink','tableLink']]
df_table_list = df_table_list.drop_duplicates()
print('tables containing keywords', len(df_table_list))

df_tables = pd.merge(df_tc, df_table_list, on=['paperLink','tableLink'],how='inner').fillna('')
df_tables['Conference'] = 'Tobacco Control'
df_tables['semantic_1'] = ''

### to make consistent with 3j
columns = "Conference,paperIdx,tableIdx,rowIdx,colIdx,type,semantic_0,sc_ch_raw,sc_ch_raw_type,human_Check,obj_cell_values,tableLink,paperLink,semantic_0_clean,semantic_1,Year,Location"
df_tables = df_tables[columns.split(',')]

df_tables.to_csv(outPath + 'LGBTQonlyTablesTC.csv', encoding='utf-8', index=False)
print('all records from tables containing keywords', len(df_tables))


########################### Pre-fill sample size info ################################

# %% 
### Extract sample size and % or 95% ci
df_N = df_tables[df_tables['sc_ch_raw'].str.contains('n=') | 
                    df_tables['sc_ch_raw'].str.contains('n =') |
                    df_tables['sc_ch_raw'].str.contains('N=') | 
                    df_tables['sc_ch_raw'].str.contains('N =')]
for index, row in df_N.iterrows():
    raw1 = row['sc_ch_raw'].lower()
    if 'n=' in raw1:
        raw2 = raw1.split('n=')[1]
    else:
        raw2 = raw1.split('n =')[1]
    raw3 = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,.<>?‡§†—]', ' ', raw2).strip()
    sampleN = raw3.split(' ')[0].strip()   
    ######### Use r'[^\d.] to remove all except non-digit, ., amd / #########
    sampleN = re.sub(r'[^\d./]+', '', sampleN).strip()
    #########################################################################
    # print(row['sc_ch_raw'], slower, '-----', sampleN)
    if '95% ci' in raw1:
        line = 'n=' + sampleN + '; ' + '95% ci'
    elif 'p Value' in raw1:
        line = 'n=' + sampleN + '; ' + 'p Value'
    elif '%' in raw1:
        line = 'n=' + sampleN + '; ' + '%'
    else:
        line = 'n=' + sampleN + '; n'
    # print(raw1, '-----', line)
    df_tables.loc[index,'semantic_0_clean'] = 'sample size'
    df_tables.loc[index,'semantic_1'] = line
# df_tables[df_tables['semantic_1'].str.contains('n=')]

# save the new table
df_tables.to_csv(outPath + 'LGBTQonlyTablesTCPrefill.csv', encoding='utf-8', index=False)


# %%
