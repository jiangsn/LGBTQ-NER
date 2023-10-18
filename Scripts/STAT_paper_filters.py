# %% 
# Load libraries
#
import os
import os.path
import pandas as pd
import re
from collections import Counter

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataPath = './STAT/'
# df_paper = pd.read_csv(dataPath + 'paperAbstractsAll.csv',encoding='utf8')


# %%
# combine paperData.csv from 4 journals
df_tc = pd.read_csv('./TobaccoControl/paperData.csv',encoding='utf8')
df_3j = pd.read_csv('./NLP_results_3j/paperDataOthers.csv',encoding='utf8')
df_all = df_tc.append(df_3j)
df_all.groupby(['Conference'])['PaperType'].count()


# %% 
# List type of papers
#
df_all.groupby(['PaperType'])['PaperType'].count()


# %%
# List number of papers with tables

# df_tbl = pd.read_csv('./TobaccoControl/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./NicotineTobaccoResearch/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./TobaccoPreventionCessation/paperData_w_figtbl_upd.csv',encoding='utf8')
# df_tbl[df_tbl['Num of Tables']>0]
df_tbl = pd.read_csv('./NLP_results_3j/TableFullRecordsFlatten3jUpd.csv',encoding='utf8')
df_tbl = df_tbl.groupby(['Conference','paperIdx'])['paperIdx'].size().to_frame('count').reset_index()
df_tbl.groupby('Conference')['paperIdx'].count()


# %%
# List number of papers with figures

# df_tbl = pd.read_csv('./TobaccoControl/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./NicotineTobaccoResearch/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./TobaccoInducedDiseases/paperData_w_figtbl_upd_corrected.csv',encoding='utf8')
df_tbl = pd.read_csv('./TobaccoPreventionCessation/paperData_w_figtbl_upd.csv',encoding='utf8')
df_tbl[df_tbl['Num of Figures']>0]


# %%
# Calculate total number of tables/figures for each journal

# df_tbl = pd.read_csv('./TobaccoControl/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./NicotineTobaccoResearch/paperData_w_figtbl.csv',encoding='utf8')
# df_tbl = pd.read_csv('./TobaccoPreventionCessation/paperData_w_figtbl_upd.csv',encoding='utf8')
df_tbl = pd.read_csv('./TobaccoInducedDiseases/paperData_w_figtbl_upd_corrected.csv',encoding='utf8')
df_tbl.groupby('Conference').agg({'Num of Tables':'sum','Num of Figures':'sum'}).reset_index() # num



# %%
# List number of papers with figures

# %%
# Count papers with race data # 727 for 3j, 176 for TC

df_race = pd.read_csv('./NLP_results_3j/RaceOnlyTables3j.csv',encoding='utf8')
# df_race = pd.read_csv('./TobaccoControl/NLP_results/RaceOnlyTables20210922.csv',encoding='utf8')
# df_race.groupby(['paperIdx']).count()
len(df_race) # 9835 for TC, 41263 for 3j


# %%
# Count paper/rows with both ethc/sample size info
df_race_n = pd.read_csv('./STAT/DataCleaningResult.csv',encoding='utf8')
df_race_n.groupby(['tableLink']).count()



# %%
# Find paper type within the 300 test set
df_test = pd.read_csv('./NER/' + 'testDataset.csv',encoding='utf8')
df_test = df_test[['Conference','Year','Title','PaperType','Link']]
df_test.to_csv(dataPath + 'testSetPaperType.csv')

# %%
