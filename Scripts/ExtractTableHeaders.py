# %%
import os
import os.path
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_colwidth', 30)
dataPath = './TobaccoControl/tableData/tables/'
outPath = './TobaccoControl/NLP_results/'


# %% 
################################### sub functions ##################################
#

### Function to find num of header lines
def numHeaderLines(inputDataFile):
    header = []
    i=0
    with open(inputDataFile, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.split(',')[0] == '':
            header.append(i)
            i += 1
        else:
            break
    return header

### Function to extract column headers & subject column items
def extractFromTable(fname):

    inputDataFile = dataPath + fname
    # find num of header lines
    header = numHeaderLines(inputDataFile)

    # open the saved table
    df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
        # full_data = main_data.iloc[:, 1:] # if need to drop the first column
    
    column_headers = []
    subject_column_rows = []
    num_rows = len(df.iloc[:,0])
    num_cols = len(df.iloc[0,:])
    if num_cols <= 1: # in the case of table 'TC.2017.338.153143.csv'
        return [], []

    # find paper info
    matchName = fname.strip('.csv')
    df_match = df_figtbl[df_figtbl['filename']==matchName].iloc[0]
    # jName = df_match['Conference']
    # pTitle = df_match['Paper Title']
    # pDOI = df_match['Paper DOI']
    # authors = df_match['Author']
    pLink = df_match['paper_url']
    pCaption = df_match['cap_url'].strip()
    tblLink = df_match['url']

    # find image link on AWS
    matchNameAWS = fname.replace('.csv', '_1.png')
    df_match_img = df_figtbl_AWS[df_figtbl_AWS['filename']==matchNameAWS].iloc[0]
    imgLink = df_match_img['url']

    # get dicts from table caption first
    dicts = build_Dicts(pCaption)

    # Loop through first column for all the rows
    for i in range(0,num_rows):
        sc_row = str(df.iloc[i,0])
        if sc_row != '':
            subject_column_rows.append(sc_row)
            # get dict if any
            dict = build_Dicts(sc_row)
            for key, value in dict.items():
                if not (key in dicts):
                    dicts[key] = value

    # Loop through column headers
    for i in range(1, num_cols):
        hd = df.columns[i]
        hdtxt = []
        if isinstance(hd, tuple):
            for subhd in hd:
                hdtxt.append(str(subhd))
            hdtxt = ' '.join(hdtxt)
        else:
            hdtxt = str(hd)
        column_headers.append(hdtxt)
        # get dict if any
        dict = build_Dicts(hdtxt)
        for key, value in dict.items():
            if not (key in dicts):
                dicts[key] = value

    df_data = []
    for sc_row in subject_column_rows:
        df_data.append(['SubjectColumn', sc_row, imgLink, tblLink, pLink])
        # df_data.append(['SubjectColumn', sc_row.strip()])
    for col_hd in column_headers:
        df_data.append(['ColumnHeader', col_hd, imgLink, tblLink, pLink])
        # df_data.append(['ColumnHeader', col_hd.strip()])
    df_value = pd.DataFrame(data=df_data, columns = ['Type', 'Value',
                            'imageLink', 'TableLink',  'PaperLink'])
    # df_value = pd.DataFrame(data=df_data, columns = ['Type', 'Value'])

    return df_value, dicts


### Function to populate a dictionary for abbreviations and the actuals
def build_Dicts(txt):
    # get all capital abbreviations
    ABBVs = re.findall(r"\b[A-Z]{2,}\b", txt)
    keys = Counter(ABBVs).keys()
    ABBVs = [key for key in keys] # get the first occurrence of ABBVs
    # print(txt, '\n', ABBVs, '\n')
    dicts = {}
    keys = []
    values = []
    # find matching values
    for ABBV in ABBVs:
        tclean = re.sub(r"[*#$^&@|()';]", '', txt.split(ABBV)[0]).strip()
        if tclean=='':
            continue
        tclean = tclean.replace(' of ', ' ') # remove 'of' from the string
        tclean = re.split('[ -]', tclean) # split also by '-'
        tclean = [x.lower() for x in tclean if x != '']
        tclean = tclean[-len(ABBV):]
        if len(tclean) < len(ABBV):
            continue
        ismatch = True
        print(ABBV, ':', tclean)
        for i in range(0,len(ABBV)):
            if ABBV[i] != tclean[i][0].upper():
                ismatch = False
                break
        if ismatch:
            keys.append(ABBV)
            tclean = ' '.join(tclean)
            values.append(tclean)
    # populate the dictionary
    for i in range(0,len(keys)):
        if not (keys[i] in dicts): # to avoid duplicates
            dicts[keys[i]] = values[i]
    return dicts


### Function to consult a dictionary to return the actuals of any abbreviations
def lookup_Dicts(txt, dicts):
    # get all capital abbreviations
    ABBVs = re.findall(r"\b[A-Z]{2,}\b", txt)
    keys = []
    values = []
    # fine item in the dicts
    for ABBV in ABBVs:
        if ABBV in dicts:
            keys.append(ABBV)
            values.append(dicts[ABBV])
    # return a two column results
    return [';'.join(keys), ';'.join(values)]



# %% 
######## Loop through all tables to extract SC rows and column headers
# 
# get links to paper and table image
df_figtbl = pd.read_csv('./TobaccoControl/figtblData.csv', encoding='utf-8', index_col=0).fillna('')
df_figtbl_AWS = pd.read_csv('./TobaccoControl/figtblDataAWS.csv', encoding='utf-8', index_col=0).fillna('')

df_paper = pd.read_csv('./TobaccoControl/paperData.csv', encoding='utf-8', index_col=0).fillna('')

# %%
df_table = pd.DataFrame()
df_dicts = pd.DataFrame()
old_paper = ''
dict_abs = {}
for i, fname in enumerate(os.listdir(dataPath)):
    if i < 0: 
        continue
    # if (i%100)==0:
    print(i)
    matchName = fname.strip('.csv')
    df_match = df_figtbl[df_figtbl['filename']==matchName].iloc[0]
    if df_match['Paper DOI'] != old_paper:
        old_paper = df_match['Paper DOI']
        if dict_abs: # save old paper dicts first
            df_dict = pd.DataFrame(dict_abs.items())
            df_dicts = df_dicts.append(df_dict)
        # get dicts from paper abstract first
        pAbstract = df_paper[df_paper['DOI'] == df_match['Paper DOI']].iloc[0]['Abstract'].strip()
        dict_abs = build_Dicts(pAbstract)
    # get SC/Col headers and dicts from table caption and headers
    df, dicts = extractFromTable(fname)
    if dicts:
        for key, value in dicts.items():
            if not (key in dict_abs):
                dict_abs[key] = value       
    # save the table headers 
    if len(df)>0:
        df = df.drop_duplicates(keep='first')
        df_table = df_table.append(df)

# save the dict from last paper
if len(df_dict)>0:
    df_dicts = df_dicts.append(df_dict)

# keep the first and drop all other duplicates 
df_dicts = df_dicts.drop_duplicates(keep='first')
df_dicts.columns = [['Abbr', 'Actual']]

# df_table = df_table.reset_index(drop=True) 
df_table.to_csv(outPath+'TableSubjectAndColumnHeadersAll.csv', encoding='utf-8', index=False)
# df_dicts.to_csv(outPath+'DictionaryFromAbstractCaptionTable.csv', encoding='utf-8', index=False)
print('Done!')

# %%
df_table.head(5)

# %%
df_dicts

# %% Count unique terms and join strings of the table link and image link cells
###
#
df_imglink = df_table.groupby(['Type','Value'])['imageLink'].apply(';'.join).reset_index()
df_tbllink = df_table.groupby(['Type','Value'])['TableLink'].apply(';'.join).reset_index()
df_plink = df_table.groupby(['Type','Value'])['PaperLink'].apply(';'.join).reset_index()

# %% combine the two columns
###
#
df_tmp = pd.merge(df_imglink, df_tbllink, on=['Type','Value'], how='outer')
df_unique = pd.merge(df_tmp, df_plink, on=['Type','Value'], how='outer')
df_unique['ValueClean'] = ''
for index, row in df_unique.iterrows():
    row['ValueClean'] = re.sub('[‡§†‡*¶]', '', row['Value'])
df_unique.to_csv(outPath+'HeadersRowsUnique.csv', encoding='utf-8')
df_unique


# %% count number of unique terms in Subject Column Rows
##################################### Deprecated ##################################
pd.set_option('display.max_colwidth', 100)
df_sc_hd = df_table[df_table['Type']=='SubjectColumn']
df_sc_hd_nunique = pd.DataFrame(data=pd.unique(df_sc_hd['Value']))
df_sc_hd_nunique.columns = ['Value']
df_sc_hd_nunique['Type']='SubjectColumn'
df_sc_hd_nunique = df_sc_hd_nunique[['Type','Value']]

# count number of unique terms in Column Headers
df_col_hd = df_table[df_table['Type']=='ColumnHeader']
df_col_hd_nunique = pd.DataFrame(data=pd.unique(df_col_hd['Value']))
df_col_hd_nunique.columns = ['Value']
df_col_hd_nunique['Type']='ColumnHeader'
df_col_hd_nunique = df_col_hd_nunique[['Type','Value']]

# %% append two dfs
df_hd_unique = df_col_hd_nunique.append(df_sc_hd_nunique)
df_hd_unique['ValueClean'] = ''
for index, row in df_hd_unique.iterrows():
    row['ValueClean'] = re.sub('[‡§†‡*¶]', '', row['Value'])
df_hd_unique.to_csv(outPath+'HeadersRowsUnique.csv', encoding='utf-8')
df_hd_unique
##################################### Deprecated ##################################


# %% 
### show a specific table
fname = 'TC.2017.338.153143.csv'
inputDataFile = dataPath+fname
header = numHeaderLines(inputDataFile)
df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
df



# %% #################### for testing #######################
s1 = 'Table 1 covariates among continuing smokers (CS)* and among baseline smokers (BS)†, overall and by plain packaging phase decond hand smoking (SHS)'
s2 = 'ddd is SHS as BS random we know CS true'

dicts = build_Dicts(s1)
result = lookup_Dicts(s2, dicts)
result

# %%

string= '#$#&^&#$@|||| >=1235 ***15 abstr'
ss = []
for s in string.split(' '):
    # s = re.sub('[\W\_]','',s)
    s = re.sub('[§*#$^&@|]','',s)
    ss.append(s)
' '.join(ss).strip()

# %%
dd = ['as', '', 'sds']
[x for x in dd if x != '']


# %%
s = 'asd fdgf-ddf-asas pls'
re.split('[ -]', s)

# %%
