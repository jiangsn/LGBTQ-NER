# %%
import os
import os.path
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 50)
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

### Function to extract the entire table
def extractFromTable(fname, paperIdx, tableIdx):

    inputDataFile = dataPath + fname
    # find num of header lines
    header = numHeaderLines(inputDataFile)

    # open the saved table
    df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
        # full_data = main_data.iloc[:, 1:] # if need to drop the first column
    
    table_records = []
    num_rows = len(df.iloc[:,0])
    num_cols = len(df.iloc[0,:])

    # find paper link and table link
    matchName = fname.strip('.csv')
    df_match = df_figtbl[df_figtbl['filename']==matchName].iloc[0]
    pLink = df_match['paper_url']
    tblLink = df_match['url']
    # find image link on AWS
    if False:
        matchNameAWS = fname.replace('.csv', '_1.png')
        df_match_img = df_figtbl_AWS[df_figtbl_AWS['filename']==matchNameAWS].iloc[0]
        imgLink = df_match_img['url']

    if num_cols < 2: # in the case of table 'TC.2017.338.153143.csv'
        for i in range(0,num_rows):
            sc_row = str(df.iloc[i,0])
            # category, sc, colhd, 
            table_records.append([i+1, '', sc_row, 0, '', sc_row])

    # Loop through all columns
    for i in range(1, num_cols):
        hd = df.columns[i]
        hdtxt = []
        if isinstance(hd, tuple):
            for subhd in hd:
                hdtxt.append(str(subhd))
            hdtxt = ' '.join(hdtxt)
        else:
            hdtxt = str(hd)
        # loop through subject column rows
        sc_hd_type = 0
        sc_hd_old = ''
        sc_hd = ''
        for j in range(0,num_rows):
            sc_row = str(df.iloc[j,0])
            sc_row_right = str(df.iloc[j,1])
            if (sc_row_right==sc_row or sc_row_right=='') and (sc_row!=''):
                sc_hd_type = 1 # an almost empty header line
                sc_hd = sc_row
                if sc_hd_old!='' and str(df.iloc[j-1,0])==sc_hd_old:
                    sc_hd = sc_hd_old + ' ' + sc_hd
                sc_hd_old = sc_hd
                # this is needed for some tables
                if str(df.iloc[j,i])==str(df.iloc[j,i-1]):
                    table_records.append([j+1, sc_hd, sc_row, i, hdtxt, '']) # str(df.iloc[j,i])
                else:
                    table_records.append([j+1, sc_hd, sc_row, i, hdtxt, str(df.iloc[j,i])])
            else:
                if sc_row!='' and sc_hd_type!=1:
                    sc_hd = sc_row
                    sc_hd_type = 2 # a single line or header followed by empty sc rows
                table_records.append([j+1, sc_hd, sc_row, i, hdtxt, str(df.iloc[j,i])])

    df_data = []
    for table_record in table_records:
        df_data.append([paperIdx, tableIdx, table_record[0], table_record[1], table_record[2], 
                    table_record[3], table_record[4], table_record[5], tblLink, pLink])
    df_value = pd.DataFrame(data=df_data, columns = ['paperIdx', 'tableIdx', 'rowIdx', 
                    'SC_level_0', 'subjectColumn', 'colIdx', 'columnHeader', 'Value', 
                    'tableLink',  'paperLink'])

    return df_value



# %% 
#
##################### Loop through all tables to extract SC rows and column headers #####################
# 
# get links to paper and table image
df_figtbl = pd.read_csv('./TobaccoControl/figtblData.csv', encoding='utf-8', index_col=0).fillna('')
df_figtbl_AWS = pd.read_csv('./TobaccoControl/figtblDataAWS.csv', encoding='utf-8', index_col=0).fillna('')
df_paper = pd.read_csv('./TobaccoControl/paperData.csv', encoding='utf-8', index_col=0).fillna('')

# %%
df_table = pd.DataFrame()
old_paper = ''
paperIdx = 0
for i, fname in enumerate(sorted(os.listdir(dataPath))):
    # if fname != 'TC.2015.ii90.145483.csv' and fname != 'TC.2015.190.145045.csv':
               # Type 2 and 1 sc headers                # Type 2 sc header
    #     continue
    matchName = fname.strip('.csv')
    df_match = df_figtbl[df_figtbl['filename']==matchName].iloc[0]
    if df_match['Paper DOI'] != old_paper:
        old_paper = df_match['Paper DOI']
        paperIdx += 1
    
    # if paperIdx < 0: 
    #     break

    if df_match['cap_url'] == '': # table no. 1171 has no title
        tableIdx = ''
    else:
        tableIdx = df_match['cap_url'].split(' ')[1]
    print(i, paperIdx, tableIdx)
    # get subject column rows and corresponding results for each column in a table
    df = extractFromTable(fname, paperIdx, tableIdx)
    if len(df)>0:
        df_table = df_table.append(df)

df_table.sort_values(['paperIdx', 'tableIdx'])
df_table.to_csv(outPath+'TableFullRecords.csv', encoding='utf-8', index=False)
print('Done!')

# %% show some results for checking
df_table[df_table['colIdx']==1]


# %% 
### show a specific table
#
fname = 'TC.2017.338.153143.csv'
inputDataFile = dataPath+fname
header = numHeaderLines(inputDataFile)
df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
df


# %%
#
###################### Extract SC rows and get the unique ones for SC_level_0 #########################
#
#
df_table = pd.read_csv(outPath+'TableFullRecords.csv', encoding='utf-8').fillna('')
# df_col = df_table[(df_table['colIdx']==1) & (df_table['SC_level_0']!='')]
df_col = df_table[df_table['colIdx']==1]
df_col = df_col['SC_level_0'].drop_duplicates().sort_values().reset_index(drop=True)
df_col = df_col.reset_index() # turning a series back into a dataframe with new index
df_col['SC_level_0_clean'] = ''
df_col


# %% 
############################ Find race SC_level_0 ############################
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_rows', 300)
select_rows = df_col['SC_level_0'].str.contains(r"\b(Race|race|Racial|racial|Multirace|ethnicity|Ethnicity)\b")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Race/Ethnicity'
df_col[df_col['SC_level_0_clean']=='Race/Ethnicity'].reset_index(drop=True)


# %% 
############################ Find AGE SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r"\b(Age|ages|Ages)\b|year-old|years old|year olds|years")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Age'
df_col[df_col['SC_level_0_clean']=='Age'].reset_index(drop=True)


# %% 
############################ Find GNEDER SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r"\b(Gender|Sex|Male|Female|LGB)\b")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Gender'
df_col[df_col['SC_level_0_clean']=='Gender'].reset_index(drop=True)


# %% 
############################ Find EDUCATION SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r"\b(Education|education)\b|Level of education")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Education'
df_col[df_col['SC_level_0_clean']=='Education'].reset_index(drop=True)


# %% 
############################ Find Income SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r'\b(Income|income|Wealth|SES)\b')
                        # use r for raw string and \b for word boundary
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Income/SES'
df_col[df_col['SC_level_0_clean']=='Income/SES'].reset_index(drop=True)


# %% 
############################ Find Marital Status SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r"\b(Marital|marital|Marriage|marriage)\b")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Marital status'
df_col[df_col['SC_level_0_clean']=='Marital status'].reset_index(drop=True)


# %% 
############################ Find Smoking status SC_level_0 ############################
pd.set_option('display.max_rows', 350)
select_rows = df_col['SC_level_0'].str.contains(r"\b(Smoking status|smoking status)\b")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Smoking status'
df_col[df_col['SC_level_0_clean']=='Smoking status'].reset_index(drop=True)


# %% 
############################ Find Employment SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains("Employment|employment|job")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Employment status'
df_col[df_col['SC_level_0_clean']=='Employment status'].reset_index(drop=True)


# %% 
############################ Find Knowledge of Smoking SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains("Knowledge|knowledge|health|Health")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Health Knowledge'
df_col[df_col['SC_level_0_clean']=='Health Knowledge'].reset_index(drop=True)


# %% 
############################ Find Year SC_level_0 ############################
select_rows = df_col['SC_level_0'].str.contains(r"\b(Year|Years|\-years|\(years\)|Time|time)\b")
df_col[select_rows].reset_index(drop=True)

# %% replace SC_level_0
df_col.loc[select_rows, 'SC_level_0_clean'] = 'Time'
df_col[df_col['SC_level_0_clean']=='Time'].reset_index(drop=True)


# %% 
############################ Find Location SC_level_0 ############################
#
# Load NLP modules
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
location_list = []
for index, row in df_col.iterrows():
    if index % 100 == 0:
        print(index)
    txt = row['SC_level_0']
    tokens = nlp(txt)
    loc_list = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE': # countries, cities, states; 'LOC': non-GPE locations, mountain ranges, bodies of water
            loc_list.append(ent.text)
    loc_counts = Counter(loc_list).most_common(10)
    locs = ','.join([(x[0]) for x in loc_counts])
    if locs:
        location_list.append([index,locs,txt, row['SC_level_0_clean']])
        # print(index, locs)

# %% save the results in dataframe
pd.set_option('display.max_rows', 620)
df_location = pd.DataFrame(location_list, columns=['index','GPE','SC_level_0','SC_level_0_clean'])
df_location

# %% clean up the results
remove_list = [0,1,2,102,103,114,142,143,195,307,343,344,345,449,463,539]
rows = df_location.index[remove_list]
df_location.drop(rows, inplace=True) # remove the rows mislabeled as location
df_location.reset_index(inplace=True, drop=True)
df_location

# %% remove those already assigned 
rows = df_location[df_location['SC_level_0_clean']!=''].index
df_location.drop(rows, inplace=True)
df_location.reset_index(inplace=True, drop=True)
df_location

# %% save the results to a table
df_location.to_csv(outPath + 'TableLocationCategory.csv', encoding='utf-8', index=False)
df_location


# %% update the main table
df_col.loc[df_location['index'],'SC_level_0_clean'] = 'Location'
df_col[df_col['SC_level_0_clean']=='Location'].reset_index(drop=True)

# %%
############### Save the updated main table ###############
df_col.to_csv(outPath + 'TableSubjectColumnLevel_0_Update.csv', encoding='utf-8', index=False)


# %%
#
############### Map cleaned category back to FullTablesFirstCol ###############
#
#
df_table = pd.read_csv(outPath+'TableFullRecords.csv', encoding='utf-8').fillna('') 
pd.set_option('display.max_rows', 30)
df_table_1stcol = df_table[df_table['colIdx']==1]
df_table_1stcol

# %%
df_sc_upd = pd.read_csv(outPath + 'TableSubjectColumnLevel_0_Update.csv', index_col=0, encoding='utf-8').fillna('') 
df_sc_upd

# %% merge two tables
df_table_1stcol_upd = pd.merge(df_table_1stcol, df_sc_upd, on='SC_level_0', how='left').fillna('') 
df_table_1stcol_upd = df_table_1stcol_upd[['paperIdx','tableIdx','rowIdx','SC_level_0',
                    'SC_level_0_clean','subjectColumn','colIdx',
                    'columnHeader','Value','tableLink','paperLink']]
df_table_1stcol_upd.to_csv(outPath + 'FullTablesFirstColumn.csv', encoding='utf-8', index=False)
df_table_1stcol_upd

# %% 
# check race/ethnic terms in subjectColumn that were missed in SC_level_0_clean
#
race_words = ['Hispanic', 'hispanic', 'White', 'white', 'Caucasian', 'caucasian',
        'African', 'african', 'American', 'american',  
        'Black', 'black', 'Race', 'race', 'races', 'ethnicity', 'Ethnicity', 'Asian', 'asian',
        'Pacific', 'pacific', 'Islander', 'Chinese', 'Han', 'Malay', 'Latino', 'Mexican', 'Arab', 
        'Chaldean', 'Persian', 'Kurdish', 'Turkmen', 'Multiethnic', 'Indian', 'Alaska',
        'Hawaiian', 'Māori', 'Samoan', 'European', 'Zealand', 'Minority']

any_in = lambda a, b: any(i in b for i in a) # function to check inclusion

cnt = 0
for index, row in df_table_1stcol_upd.iterrows():
    if row['SC_level_0_clean']=='':
        txt = str(row['subjectColumn'])
        txt = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt) # '\s' means space
        if any_in(race_words, txt):
            cnt += 1
            print(index, row['subjectColumn'])
            df_table_1stcol_upd.loc[index,'SC_level_0_clean'] = 'Race/Ethnicity'
print('total =', cnt)
# update FullTablesFirstColumn.csv
df_table_1stcol_upd.sort_values(['paperIdx', 'tableIdx'])
df_table_1stcol_upd.to_csv(outPath + 'FullTablesFirstColumn.csv', encoding='utf-8', index=False)



# %%
#
# Open cleaned firstColumn race data and select those tables with race in it
#
#
df_1stcol_race_clean = pd.read_csv(outPath+'FullRecordsFirstColumnCleanedRace.csv', encoding='utf-8').fillna('') 
pd.set_option('display.max_rows', 30)
df_1stcol_race_clean_race = df_1stcol_race_clean[df_1stcol_race_clean['SC_level_0_clean']=='Race/Ethnicity']
race_table_list = df_1stcol_race_clean_race['paperIdx'].unique()
df_race_tables_1stcol = df_1stcol_race_clean.loc[df_1stcol_race_clean['paperIdx'].isin(race_table_list)]
df_race_tables_1stcol = df_race_tables_1stcol.sort_values(['paperIdx', 'tableIdx','rowIdx'])
df_race_tables_1stcol.to_csv(outPath + 'RaceEthnicTablesFirstColumn.csv', encoding='utf-8', index=False)
df_race_tables_1stcol

# %%
len(race_table_list)


# %% ########################################################
#
#############  For Testing
#
#############################################################

s1 = 'Table 1 covariates among continuing smokers (CS)* and among baseline smokers (BS)†, overall and by plain packaging phase decond hand smoking (SHS)'
s2 = 'ddd is SHS as BS random we know CS true'

dicts = build_Dicts(s1)
result = lookup_Dicts(s2, dicts)
result
