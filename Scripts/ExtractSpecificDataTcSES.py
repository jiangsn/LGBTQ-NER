# %%
import os
import os.path
import pandas as pd
from collections import Counter
import re
from collections import Counter

outPath = './TobaccoControl/NLP_results/'
tablesTC = 'TableFullRecordsFlattenUpd.csv'


# %% ######################################################################################
#
# Select only tables with "Income/SES" info in it
#
df_tc = pd.read_csv(outPath + tablesTC, encoding='utf-8').fillna('')
df_table_list = df_tc[df_tc['semantic_0']=='Income/SES']
print(len(df_table_list))
df_table_list = df_table_list[['paperLink','tableLink']].drop_duplicates()
df_table_list


# %%
### get the whole tables containing "Income/SES" in the table
#
df_tables = pd.merge(df_tc, df_table_list, on=['paperLink','tableLink'],how='inner').fillna('')
df_tables['Conference'] = 'Tobacco Control'
df_tables['semantic_1'] = ''


# %% 
### to make consistent with 3j
#
columns = "Conference,paperIdx,tableIdx,rowIdx,colIdx,type,semantic_0,sc_ch_raw,sc_ch_raw_type,human_Check,obj_cell_values,tableLink,paperLink,semantic_0_clean,semantic_1,Year,Location"
df_tables = df_tables[columns.split(',')]
df_tables.to_csv(outPath + 'SESonlytablesTC.csv', encoding='utf-8', index=False)
df_tables


# %% 
########################### Pre-fill sample size info ################################
#
df_tables = pd.read_csv(outPath + 'SESonlytablesTC.csv', encoding='utf-8').fillna('')
df_tables.head(3)   # load in data if not already

# %% Extract sample size and % or 95% ci
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

df_tables[df_tables['semantic_1'].str.contains('n=')]


# %%  save the new table
df_tables.to_csv(outPath + 'SESonlytablesTCPrefill.csv', encoding='utf-8', index=False)




# %%
