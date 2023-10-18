# %%
import os
import os.path
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_colwidth', 30)
nlp_path = '/home/jian/mengling/TobaccoResearch/TobaccoControl/NLP_results/'
dataPath = './TobaccoControl/tableData/tables/'

# %% 
############## Function to find num of header lines
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


# %% 
########################################## this enclosded part is done ############
f_HeaderRowUnique = 'TableSubjectAndColumnHeadersAll.csv'

df_HR = pd.read_csv(nlp_path+f_HeaderRowUnique, encoding='utf-8')
df_HR

# %%
# search all values with the pop_words
pop_words = ['Race', 'race', 'RACE', 'races', 'Races', 'multirace', 'Multirace',
            'Ethnicity', 'ethnicity', 'ETHNICITY']
            # 'Age', 'age', 'AGE', 'ages', 'gender', 'Gender', 'GENDER']
any_in = lambda a, b: any(i in b for i in a) # function to check inclusion

cnt = 0
tables = []
for index, row in df_HR.iterrows():
    # in re expression '/s' means space
    txt = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', str(row['Value'])) 
    if any_in(pop_words, txt):
        tables.append([txt, row['Type'], row['TableLink'], row['imageLink'], row['PaperLink']])
        # append to list
        # values.append(item_Clr)

df = pd.DataFrame(data=tables, columns=['Value', 'Type', 'TableLink', 'imageLink', 'PaperLink'])
df.to_csv(nlp_path + 'HeaderRows_with_RaceEthnicWords.csv', encoding='utf-8')
df

# %% single out all the tables
df_clean = pd.read_csv(nlp_path+'HeaderRows_with_RaceEthnicWords.csv', index_col=0, encoding='utf-8')
df_uniqueTbl = df_clean.groupby(['TableLink'])['Value'].count().reset_index()
df_uniqueTbl

# %%
# Query out all column headers and subject column rows
lines = []
df_all = pd.merge(df_HR, df_uniqueTbl[['TableLink']], on='TableLink', how='right')
df_all.sort_values(['PaperLink', 'TableLink'], ascending=[True, False])
df_all.to_csv(nlp_path+'HeaderRowsRaceEthnicAllRecords.csv', encoding='utf-8')

# %%
# 
# Open the data and add columns to help labeling
#
df_all = pd.read_csv(nlp_path+'HeaderRowsRaceEthnicAllRecords.csv', index_col=0, encoding='utf-8')
df_all['tblIdx'] = ''
df_all['race?'] = ''
df_all = df_all[['tblIdx', 'Type', 'race?', 'Value', 'TableLink', 'PaperLink', 'imageLink']]
df_all

# %%
# fill in the new columns

race_words = ['Hispanic', 'hispanic', 'White', 'white', 'Caucasian', 'caucasian',
        'Black', 'black', 'African', 'african', 'American', 'american', 'Other', 'other', 
        'Black', 'black', 'Race', 'race', 'races', 'ethnicity', 'Ethnicity', 'Asian', 'asian',
        'Pacific', 'pacific', 'Islander', 'Chinese', 'Han', 'Malay', 'Latino', 'Mexican', 'Arab', 
        'Chaldean', 'Persian', 'Kurdish', 'Turkmen', 'Multiethnic', 'Indian', 'Alaska',
        'Hawaiian', 'Māori', 'Samoan', 'European', 'Zealand', 'Minority']

any_in = lambda a, b: any(i in b for i in a) # function to check inclusion

old_table = ''
tblIdx = 0
for index, row in df_all.iterrows():
    tbl = row['TableLink']
    if tbl==old_table:
        row['tblIdx'] = tblIdx
    else:
        tblIdx += 1
        row['tblIdx'] = tblIdx
        old_table = tbl
    txt = str(row['Value'])
    txt = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt) # '\s' means space
    if any_in(race_words, txt):
        row['race?'] = 'yes'

df_all.to_csv(nlp_path+'HeaderRowsRaceEthnicAllRecords.csv', encoding='utf-8')
df_all[0:20]

########################################## the above enclosded part is done ############



# %%
# Open the race table and add sample size columns
#
df_race = pd.read_csv(nlp_path+'HeaderRowsRaceEthnicAllRecords.csv', index_col=0, encoding='utf-8').fillna('')
df_race['n_size'], df_race['n_note'] = '', ''
df_race = df_race[['tblIdx', 'Type', 'race?', 'Value', 'n_size', 'n_note', 'TableLink', 
                    'PaperLink', 'imageLink']]
df_race[5:13]


# %%
# extract from saved tables and fill the race table with sample size info
#
race_cell_idx = df_race[df_race['race?']=='yes'].index

tblidx_old = -1
n_note = ''
for index in race_cell_idx:
    tblidx = df_race.iloc[index]['tblIdx']
    if tblidx<=-1 or tblidx>500:
        continue
    
    if tblidx != tblidx_old:
        tblidx_old = tblidx
        matchfname = df_race.iloc[index]['imageLink'].split('/')[-1].split('_')[0] + '.csv'
        inputDataFile = dataPath + matchfname
        # find num of header lines
        header = numHeaderLines(inputDataFile)
        # open the saved table
        df_tbl = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
        print('table index =', tblidx)

    value_type = df_race.iloc[index]['Type']
    value_txt = df_race.iloc[index]['Value']
    if value_type == 'SubjectColumn':
        match_idx = df_tbl[df_tbl.iloc[:,0]==value_txt].index[0]
        right_cell = df_tbl.iloc[match_idx,1]
        # when the row is a section header, use column header
        if right_cell==value_txt or right_cell=='': 
            # check the heading cell
            header_cell = df_tbl.columns[1]
            if isinstance(header_cell, tuple):
                header_cell = ' '.join(header_cell)
            else:
                header_cell = str(header_cell)
            print('   ', 'header cell:', header_cell)
            # analyze header cell
            if 'n=' in header_cell:
                txt = header_cell.split('n=')[1]
                # n_size = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt)
                n_size = re.split(r'[`\-=~!@#$^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt)
                n_size = [x for x in n_size if (x!='' and x.isnumeric())]
                n_size = '; '.join(n_size)
            elif 'N=' in header_cell:
                txt = header_cell.split('N=')[1]
                n_size = re.split(r'[`\-=~!@#$^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt)
                n_size = [x for x in n_size if (x!='' and x.isnumeric())]
                n_size = '; '.join(n_size)
            else:
                n_size = ''
            
            if '%' in header_cell:
                n_note = '%'
            else:
                n_note = ''
            # sometimes '%' is in SC header
            if '%' in value_txt:
                print('   ', 'SC cell:', value_txt)
                n_note = '%'
            print('      ', value_txt, n_size)
            # save the result
            df_race.at[index, 'n_size'] = n_size

        else:
            # first check if 'n=' is in the subject column value_txt
            if 'n=' in value_txt:
                print('   ', 'SC cell:', value_txt)
                txt = value_txt.split('n=')[1]
                n_size = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt)
                n_size = n_size[0]
                if '%' in value_txt.split(n_size)[1]:
                    n_note = '%'
                else:
                    n_note = ''
            else:
                print('   ', 'SC right cell:', right_cell)
                n_size = str(right_cell).split()[0]
                if '%' in n_size:
                    n_size = n_size.split('%')[0]
                    n_note = '%'
            print('      ', value_txt, n_size, n_note)
            # save the result
            df_race.at[index, 'n_size'] = n_size
            df_race.at[index, 'n_note'] = n_note

    elif value_type=='ColumnHeader':
        print('ColumnHeader:', value_txt)
        df_race.at[index, 'n_note'] = value_txt
    else:
        print('Type not right!!!')

# save the results to file
df_race.to_csv(nlp_path+'RaceEthnicAllRecordsAutoFilled.csv', encoding='utf-8')
    


    
############# special tables for further review:
# table index = 57
#
#

# %% check if the results have been saved in table
df_race[5:50]








# %% #################### for testing #######################
match_idx = df_tbl[df_tbl.iloc[:,0]=='Race/ethnicity'].index[0]
right_cell = df_tbl.iloc[match_idx,1]
# when the row is a section header, use column header
if right_cell=='Race/ethnicity' or right_cell=='': 
    # check the heading cell
    header_cell = df_tbl.columns[1]
    if isinstance(header_cell, tuple):
        header_cell = str(header_cell[-1])
    else:
        header_cell = str(header_cell)
    # analyze header cell
    txt = header_cell.split('n=')[1]
    n_size = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt)
    if '%' in header_cell.split(' '): 
        n_note = '%'
    else:
        n_note = 'N/A'
    print(n_size, n_note)
else:
   n_size = right_cell.split()[0]
   print(n_size)


# %%
