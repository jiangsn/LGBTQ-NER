# %%
import os
import os.path
from numpy import True_
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
import csv

import spacy
nlp = spacy.load('en_core_web_lg')
nlp_sentence = spacy.load('en_use_md') # https://github.com/MartinoMensio/spacy-universal-sentence-encoder 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 20)
outPath = './NLP_results_3j/'


# %% 
################################### sub functions ##################################
#

### Function to see if there is a noun
def hasNoun(cellValue):
    cellValueR = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—–]', ' ', cellValue)
    sentence = nlp(cellValueR.strip())
    # get all Parts of Speech (pos)
    for word in sentence:
        # print(word.pos_)
        if word.pos_ == 'NOUN' or word.pos_ == 'ADJ' or word.pos_ == 'VERB':
            return True
    return False


### Function to classify a string into sentence, phrase, or non-semantic
def stringType(cellValue):
    cellValueR = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—–]', ' ', cellValue)
    sentence = nlp(cellValueR.strip())
    # get all Parts of Speech (pos)
    have_subj = False
    have_obj = False
    have_root = False
    have_comb = False
    have_intj = False
    have_NUM = False
    count = 0
    for word in sentence:
        count += 1
        if word.dep_ == 'nsubj':
            have_subj = True
        elif word.dep_ == 'dobj':
            have_obj = True
        elif word.dep_ == 'ROOT':
            have_root = True
        elif word.dep_ == 'xcomp':
            have_comb = True

        if word.pos_ == 'INTJ':
            have_intj = True
        elif word.pos_ == 'NUM':
            have_NUM = True
        # print(word.text,  word.pos_, word.dep_)
    # print(count, have_intj, have_NUM)

    if have_subj and (have_obj or have_comb) and have_root:
        return 'sentence'
    elif count>1:
        return '' # phrase no need to show
    elif count==1 and have_intj:
        return 'no-meaning'
    elif count==1 and have_NUM and (not ('+' in cellValue)):
        return 'no-meaning'
    else:
        return '' # phrase no need to show

### Function to extract root of a sentence
def stringRoot(cellValue):
    if cellValue == '':
        return ''
    cellValue = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—–]', ' ', cellValue)
    cellValue = cellValue.strip()
    sentence = nlp(cellValue)
    # get all dependency (dep)
    roots = []
    for word in sentence:
        if word.dep_ == 'ROOT':
            roots.append(word.text)
        # print(word.text,  word.pos_, word.dep_)
    if any(roots):
        return ' '.join(roots)
    else:
        return ''


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
def extractFromTable(conference, fname, paperIdx, tableIdx):

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
        # get column header 
        hd = df.columns[0]
        hdtxt = []
        if isinstance(hd, tuple):
            for subhd in hd:
                hdtxt.append(str(subhd))
            hdtxt = ' '.join(hdtxt)
        else:
            hdtxt = str(hd)
        hd_type = stringType(hdtxt)
        coltxt = ''
        for i in range(0,num_rows):
            sc_row = str(df.iloc[i,0]).strip()
            sc_row_type = stringType(sc_row)
            table_records.append([str(i+1), '0', 'sbjCol', hdtxt, sc_row, sc_row_type, '', sc_row])
            coltxt += sc_row + '; '
        hd_semantic = stringRoot(hdtxt) # use Root of string to fill header
        table_records.append(['0', '0', 'colHd', hd_semantic, hdtxt, hd_type, '', coltxt.strip('; ')])

    else:
        sc_hd_type = 0
        sc_hd_old = ''
        sc_hd = ''
        # loop through subject column rows
        for j in range(0,num_rows):
            # Loop through all columns to merger cell values
            CellValues = []
            for i in range(1, num_cols):
                if str(df.iloc[j,i]).strip()=='':
                    CellValues.append('')
                else:
                    # CellValues.append('['+str(i)+','+str(j+1)+']: ' + str(df.iloc[j,i]))
                    CellValues.append(str(df.iloc[j,i]))
            if any(CellValues)==False:
                CellValues = ''
            else:
                CellValues = '; '.join(CellValues)

            sc_row = str(df.iloc[j,0]).strip()
            sc_row_type = stringType(sc_row)
            sc_row_right = str(df.iloc[j,1])
            if (sc_row_right==sc_row or sc_row_right=='') and (sc_row!=''):
                sc_hd_type = 1 # an almost empty header line
                sc_hd = sc_row
                if sc_hd_old!='' and str(df.iloc[j-1,0])==sc_hd_old:
                    sc_hd = sc_hd_old + ' ' + sc_hd
                sc_hd_old = sc_hd
                # remove repeated obj_cell_values if equal to sc_hd
                unq = set(CellValues.split('; '))
                if len(unq)==1:
                    if list(unq)[0]==sc_hd:
                        CellValues = ''
                table_records.append([str(j+1), '0', 'sbjCol', sc_hd, sc_row, sc_row_type, '', CellValues])
            else:
                if sc_row!='' and sc_hd_type!=1:
                    sc_hd = stringRoot(sc_row) # use Root of string to fill header
                    sc_hd_type = 2 # a single line OR header followed by empty sc rows
                if sc_row=='':
                    humanChk = '1'
                else:
                    humanChk = ''
                table_records.append([str(j+1), '0', 'sbjCol', sc_hd, sc_row, sc_row_type, humanChk, CellValues])

        # loop through columns
        for i in range(0, num_cols):
            # get column header
            hd = df.columns[i]
            hdtxt = []
            if isinstance(hd, tuple):
                for subhd in hd:
                    if not ('Unnamed' in str(subhd)):
                        hdtxt.append(str(subhd))
                if any(hdtxt):
                    hdtxt = ' '.join(hdtxt)
                else:
                    hdtxt = ''
            else:
                if 'Unnamed' in str(hd):
                    hdtxt = ''
                else:
                    hdtxt = str(hd)
            hd_type = stringType(hdtxt)
            hd_semantic = stringRoot(hdtxt) # use Root of string to fill header
            CellValues = []
            for j in range(0,num_rows):
                if str(df.iloc[j,i]).strip()=='':
                    CellValues.append('')
                else:
                    CellValues.append(str(df.iloc[j,i]))
            CellValues = '; '.join(CellValues)
            table_records.append(['0', str(i), 'colHd', hd_semantic, hdtxt, hd_type, '', CellValues])
        
    df_data = []
    old_sc = '' 
    for table_record in table_records:
        if table_record[4]==old_sc:
            if table_record[2]=='sbjCol':
                table_record[6] = '1'
        else:
            old_sc = table_record[4]
        df_data.append([conference, paperIdx, tableIdx, table_record[0], table_record[1], table_record[2], 
                table_record[3], table_record[4], table_record[5], table_record[6], 
                table_record[7], tblLink, pLink])
    df_value = pd.DataFrame(data=df_data, columns = ['Conference', 'paperIdx', 'tableIdx', 'rowIdx', 'colIdx',
                    'type', 'sc_ch_semantic_0', 'sc_ch_raw', 'sc_ch_raw_type', 'human_Check', 'obj_cell_values', 
                    'tableLink',  'paperLink'])

    return df_value


# %% 
#
##################### Loop through all tables to extract SC rows and column headers #####################
# 
# get links to paper and table image
metafiles = [['./TobaccoInducedDiseases/figtblData.csv',
                './TobaccoInducedDiseases/paperData.csv',
                './TobaccoInducedDiseases/tableData/tables/'],
            ['./TobaccoPreventionCessation/figtblData.csv',
                './TobaccoPreventionCessation/paperData.csv',
                './TobaccoPreventionCessation/tableData/tables/'],
            ['./NicotineTobaccoResearch/figtblData.csv', 
                './NicotineTobaccoResearch/paperData.csv',
                './NicotineTobaccoResearch/tableData/tables/']]

metafiles = [['./TobaccoPreventionCessation/figtblData.csv',
                './TobaccoPreventionCessation/paperData.csv',
                './TobaccoPreventionCessation/tableData/tables/']]

df_table = pd.DataFrame()

for metafile in metafiles:
    df_figtbl = pd.read_csv(metafile[0], encoding='utf-8').fillna('')
    df_paper = pd.read_csv(metafile[1], encoding='utf-8', index_col=0).fillna('')
    dataPath = metafile[2]
    print(len(df_figtbl), len(df_paper), dataPath)
    if True: 
        old_paper = ''
        paperIdx = 0
        paperIdx_old = -1
        for i, fname in enumerate(sorted(os.listdir(dataPath))):
            # if fname != 'TC.2015.ii90.145483.csv' and fname != 'TC.2015.190.145045.csv':
                    # Type 2 and 1 sc headers                # Type 2 sc header
            #     continue
            matchName = fname.strip('.csv')
            df_match = df_figtbl[df_figtbl['filename']==matchName]
            if len(df_match)==0:
                continue
            else:
                df_match = df_match.iloc[0]
            if df_match['Paper DOI'] != old_paper:
                old_paper = df_match['Paper DOI']
                paperIdx += 1
                tableIdx_na = 1 # assign to those without table no.
            
            if paperIdx < 0:
                break
            
            if df_match['cap_url'] == '': # table no. 1171 has no title
                tableIdx = tableIdx_na
                tableIdx_na += 1
            else:
                if 'NicotineTobaccoResearch' in metafile[0]:
                    tableIdx = df_match['cap_url'].split(' ')[1].strip('.')
                else:
                    tableIdx = df_match['filename'].split('.')[-1]
            print(metafile[0].split('/')[1], i, paperIdx, tableIdx)
            # get subject column rows and corresponding results for each column in a table
            df = extractFromTable(df_match['Conference'], fname, paperIdx, tableIdx)
            if len(df)>0:
                df_table = df_table.append(df)

df_table = df_table.sort_values(['Conference', 'paperIdx', 'tableIdx']).reset_index(drop=True)
# df_table.to_csv(outPath+'TableFullRecordsFlatten3j.csv', encoding='utf-8', index=False)
df_table.to_csv(outPath+'TableFullRecordsFlatten_TPC.csv', encoding='utf-8', index=False)
print('Done!')


# %% 
##### show some results for checking
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 30)
df_table = pd.read_csv(outPath+'TableFullRecordsFlatten3j.csv', encoding='utf-8').fillna('')
df_table


# %% 
################ Tool: show a specific table ################
#
fname = 'TC.2019.s20.157264.csv'
inputDataFile = dataPath+fname
header = numHeaderLines(inputDataFile)
df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
df

# %% or using index to get the table link
tblLink = df_table[df_table.index==52786]['tableLink']
tblLink.values[0]


# %% 
############################ Find and autofill key semantic_level_0 ############################
# pd.set_option('display.max_colwidth', 120)
# pd.set_option('display.max_rows', 300)
select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Race|race|Racial|racial|Multirace|multiracial|ethnicity|Ethnicity)\b")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Race/Ethnicity'
# df_table[df_table['sc_ch_semantic_0']=='Race/Ethnicity'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Age|ages|Ages)\b|year-old|years old|year olds|years")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Age'
#df_table[df_table['sc_ch_semantic_0']=='Age'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Gender|Sex|Male|Female|LGB)\b")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Gender'
# df_table[df_table['sc_ch_semantic_0']=='Gender'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Education|education)\b|Level of education")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Education'
# df_table[df_table['sc_ch_semantic_0']=='Education'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r'\b(Income|income|Wealth|SES)\b')
                        # use r for raw string and \b for word boundary
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Income/SES'
# df_table[df_table['sc_ch_semantic_0']=='Income/SES'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Marital|marital|Marriage|marriage)\b")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Marital status'
# df_table[df_table['sc_ch_semantic_0']=='Marital status'].reset_index(drop=True)

pd.set_option('display.max_rows', 350)
select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Smoking status|smoking status)\b")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Smoking status'
# df_table[df_table['sc_ch_semantic_0']=='Smoking status'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains("Employment|employment|job")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Employment status'
# df_table[df_table['sc_ch_semantic_0']=='Employment status'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains("Knowledge|knowledge|health|Health")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Health knowledge'
# df_table[df_table['sc_ch_semantic_0']=='Health Knowledge'].reset_index(drop=True)

select_rows = df_table['sc_ch_semantic_0'].str.contains(r"\b(Year|Years|\-years|\(years\)|Time|time)\b")
# df_table[select_rows].reset_index(drop=True)
df_table.loc[select_rows, 'sc_ch_semantic_0'] = 'Datetime'
# df_table[df_table['sc_ch_semantic_0']=='Time'].reset_index(drop=True)

df_table.to_csv(outPath+'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8', index=False)


# %%
df_table = pd.read_csv(outPath+'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8').fillna('')
df_table


# %%
###################### Merge Location/Year into FullRecords ######################
#
df_location = pd.read_csv(outPath+'TobaccoPapersKeySummaryOthers.csv', encoding='utf-8').fillna('')
# remove space in paperLink to avoid error
df_location['paperLink'] = df_location['paperLink'].str.strip()
df_table_upd = pd.merge(df_table, df_location[['Year','Location','paperLink']], 
            on='paperLink', how='left').fillna('')
df_table_upd            

# %%            
df_table_upd.to_csv(outPath+'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8', index=False)


# %% 
######## check race/ethnic terms in subjectColumn that were missed in sc_ch_semantic_0
#
race_words = ['hispanic', 'caucasian', 'african', 'race', 'races', 'ethnicity', 'asian',
        'pacific', 'islander', 'chinese', 'han', 'malay', 'latino', 'mexican', 'arab', 
        'chaldean', 'persian', 'kurdish', 'turkmen', 'multiethnic', 'indian', 'alaska',
        'hawaiian', 'māori', 'samoan', 'european', 'zealand', 'minority', 'multiracial']

any_in = lambda a, b: any(i in b for i in a) # function to check inclusion

cnt = 0
for index, row in df_table.iterrows():
    if row['sc_ch_semantic_0']!='Race/Ethnicity':
        txt = str(row['sc_ch_raw']).lower()
        txt = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', txt) # '\s' means space
        if any_in(race_words, txt):
            cnt += 1
            print(index, row['sc_ch_raw'])
            df_table.loc[index,'sc_ch_semantic_0'] = 'Race/Ethnicity'
print('total =', cnt)
# save results
df_table.to_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8', index=False)


# %%
###################### get unique race/ethnicity results ######################
#
df_race_rows = df_table[df_table['sc_ch_semantic_0']=='Race/Ethnicity']
df_race_rows

# %% clean up race/ethnic words
# pd.set_option('display.max_rows', 330)
df_race = df_race_rows.groupby('sc_ch_raw')['tableLink'].first().to_frame(name = 'tableLbink').reset_index()
df_race['sc_ch_raw_clean'] = ''
for index, row in df_race.iterrows():
    df_race.loc[index,'sc_ch_raw_clean'] = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—]', ' ', row['sc_ch_raw']).strip().lower()
df_race.sort_values('sc_ch_raw_clean')
df_race['semantic_0_clean'] = ''
df_race['raceTerm'] = ''
df_race['ethnicTerm'] = ''
df_race.to_csv(outPath+'RaceEthnicityTermsClean3j.csv', encoding='utf-8', index=False)
df_race



#
#
#
#
#





# %%
################ Merge already-cleaned sc_level_0_clean into new FullRecords ################
#
df_race_cleaned = pd.read_csv(outPath+'TableRaceEthnic3jRecords.csv').fillna('')
df_race_cleaned

#%% change column type to match with TablesWithRaceRecordsCleaned 
df_table['rowIdx'] = df_table['rowIdx'].astype('int')

df_table_upd = pd.merge(df_table, df_race_cleaned[['Conference','paperIdx','tableIdx','rowIdx',
            'sc_ch_raw','semantic_0_clean','semantic_1']], on=['Conference','paperIdx','tableIdx',
            'rowIdx','sc_ch_raw'], how='left').fillna('')
df_table_upd

# %% 
df_table_upd.rename(columns={'sc_ch_semantic_0':'semantic_0'}, inplace=True)
# df_table_upd2.rename(columns={'SC_level_0_clean':'semantic_0_clean'}, inplace=True)
# df_table_upd[df_table_upd['semantic_0_clean']!='']
df_table_upd


# %% 
############################ Find Location ############################
#
# Load NLP modules
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
location_list = []
for index, row in df_table_upd.iterrows():
    if index % 100 == 0:
        print(index)
    txt = row['semantic_0']
    tokens = nlp(txt)
    loc_list = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE': # countries, cities, states; 'LOC': non-GPE locations, mountain ranges, bodies of water
            loc_list.append(ent.text)
    loc_counts = Counter(loc_list).most_common(10)
    locs = ','.join([(x[0]) for x in loc_counts])
    if locs:
        location_list.append([index,locs,txt])
        # print(index, locs)

# %% save the results in dataframe
pd.set_option('display.max_rows', 30)
df_location = pd.DataFrame(location_list, columns=['index','GPE','semantic_0'])
df_location

# %% clean up the results
if False:
    remove_list = [0,1,2,102,103,114,142,143,195,307,343,344,345,449,463,539]
    rows = df_location.index[remove_list]
    df_location.drop(rows, inplace=True) # remove the rows mislabeled as location
    df_location.reset_index(inplace=True, drop=True)
    df_location

# %% remove those already assigned 
if False:
    rows = df_location[df_location['sc_ch_semantic_0']!=''].index
    df_location.drop(rows, inplace=True)
    df_location.reset_index(inplace=True, drop=True)
    df_location
    # save the results to a table
    df_location.to_csv(outPath + 'TableLocationCategory.csv', encoding='utf-8', index=False)
    df_location

# %% update and save the main table
df_table_upd.loc[df_location['index'],'semantic_0'] = 'Location'
# df_table[df_table['sc_ch_semantic_0']=='Location'].reset_index(drop=True)
df_table_upd.to_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8', index=False)


# %% #############################################################
#
# Find human_Check=1 records and move the first cell value to combine with sc_ch_raw
# Also check the cell above 1 - merge if need to
#
df_table = pd.read_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8').fillna('')
df_table[df_table['human_Check']==1]

# %% first clean obj_cell_values that have the same terms - keep one only
for index, row in df_table.iterrows():
    cellV = row['obj_cell_values'].split('; ')
    cellV = [x for x in cellV if x!='']
    if len(cellV) > 1:
        if len(set(cellV))==1:
            df_table.loc[index,'obj_cell_values'] = cellV[0]
df_table[df_table['human_Check']==1]

# %% do not run this alone without running the above block
for index, row in df_table[df_table['human_Check']==1].iterrows():
    cellV = row['obj_cell_values'].split('; ')
    if row['sc_ch_raw']!='': # move it to become semantic_0
        df_table.loc[index,'semantic_0'] = row['sc_ch_raw']
    df_table.loc[index,'sc_ch_raw'] = cellV[0]
    # remove it from obj_cell_values
    if len(cellV)==1:
        df_table.loc[index,'obj_cell_values'] = ''
    else:    
        df_table.loc[index,'obj_cell_values'] = '; ' + '; '.join(cellV[1:])
    # do the same to the row above, if contains a phrase/noun
    if df_table.loc[index-1,'human_Check'] == '':
        cellV = df_table.loc[index-1,'obj_cell_values'].split('; ')
        if hasNoun(cellV[0]):
            if df_table.loc[index-1,'sc_ch_raw'] != '': # move it to become semantic_0
                df_table.loc[index-1,'semantic_0'] = df_table.loc[index-1,'sc_ch_raw']
            df_table.loc[index-1,'sc_ch_raw'] = cellV[0]
            if len(cellV)==1:
                df_table.loc[index-1,'obj_cell_values'] = ''
            else:
                df_table.loc[index-1,'obj_cell_values'] = '; ' + '; '.join(cellV[1:])
            df_table.loc[index-1,'human_Check'] = 1

df_table[df_table['human_Check']==1]

# %% save the updated df_table
df_table.to_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8', index=False)


# %% ########################################################################################
#
# Get unique sc_ch_raw terms and assign semantics to those easy ones
#
df_table = pd.read_csv(outPath + 'TableFullRecordsFlattenUpd.csv', encoding='utf-8').fillna('')
df_sc_ch_unique = df_table.groupby(['sc_ch_raw'])['tableLink'].first().to_frame(name='tableLink').reset_index()
df_sc_ch_unique['clean0'] = ''
df_sc_ch_unique = df_sc_ch_unique[['sc_ch_raw', 'clean0', 'tableLink']]
df_sc_ch_unique

# %%
for index, row in df_sc_ch_unique.iterrows():
    # clean0 = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—–▸]', '', str(row['sc_ch_raw']))
    clean0 = re.sub(r'[`\=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?‡§†▸\t¶]', '', str(row['sc_ch_raw']))
    df_sc_ch_unique.loc[index,'clean0'] = clean0.strip().lower()        
df_sc_ch_unique.to_csv(outPath + 'Table_SB_CH_clean0.csv', encoding='utf-8', index=False) # quoting=csv.QUOTE_NONNUMERIC
####### Note: import rather than open the csv file in Google Sheet #######
df_sc_ch_unique

# %% get unique records again
df_sc_ch_clean0 = df_sc_ch_unique[df_sc_ch_unique['clean0']!='']
df_sc_ch_clean1 = df_sc_ch_clean0.groupby('clean0')[['sc_ch_raw', 'tableLink']].first().reset_index()
df_sc_ch_clean1.to_csv(outPath + 'Table_SB_CH_clean1.csv', encoding='utf-8', index=False) # , quoting=csv.QUOTE_NONNUMERIC
####### Note: import rather than open the csv file in Google Sheet #######
df_sc_ch_clean1



# %% ####################################### Update master ################################
#
# Put assigned semantic_0 in Table_SB_CH_clean1 back to TableFullRecords
#
df_sc_ch_cleaned = pd.read_csv(outPath + 'Table_SB_CH_clean1_upd.csv', encoding='utf-8').fillna('')
df_sc_ch_cleaned_only = df_sc_ch_cleaned[df_sc_ch_cleaned['sc_ch_semantic_0']!='']
df_sc_ch_cleaned_only = df_sc_ch_cleaned_only[['clean0','sc_ch_semantic_0']]
df_sc_ch_cleaned_only

# %%
df_sc_ch_clean0 = pd.read_csv(outPath + 'Table_SB_CH_clean0.csv', encoding='utf-8').fillna('')

# %% match to sc_ch_raw
df_sc_ch_match = pd.merge(df_sc_ch_clean0, df_sc_ch_cleaned_only, on='clean0', how='inner')
df_sc_ch_match

# %% open the full table and replace 'Time' by 'Datetime'
df_full_upd = pd.read_csv(outPath + 'TableFullRecordsFlattenUpd.csv', encoding='utf-8').fillna('')
time_records = df_full_upd[df_full_upd['semantic_0']=='Time']
df_full_upd.loc[time_records.index, 'semantic_0'] = 'Datetime'
df_full_upd[df_full_upd['semantic_0']=='Datetime']

# %% update full table with the assigned semantic_0 to the sc_ch_unique
df_sc_ch_match_only = df_sc_ch_match[['sc_ch_raw','sc_ch_semantic_0']]
df_fill_upd_match = pd.merge(df_full_upd, df_sc_ch_match_only, on='sc_ch_raw', how='left')
df_fill_upd_match = df_fill_upd_match.fillna('')
df_fill_upd_match[df_fill_upd_match['sc_ch_semantic_0']!='']
#
########### DO NOT APPLY THE ABOVE BLCOK AS IT MIGHT CONFUSE RATHER THAN CLARIFY ##########



# %% ######################################################################################
#
# Update the cleaned level_1 race/ethnicity back into Master table
############ No need for the 3j tables ###########
#
df_race_clean = pd.read_csv(outPath + 'RaceEthnicityTermsCleanUpd.csv', encoding='utf-8').fillna('')
df_race_clean['RaceEthnic_level1'] = df_race_clean['Race_category'] + '; ' + df_race_clean['Ethnic_category']
df_race_clean['RaceEthnic_level1'][0:5]

# %% 
################# Open full table and replace 'Time' by 'Datetime'
df_full_upd = pd.read_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8').fillna('')
time_records = df_full_upd[df_full_upd['semantic_0']=='Time']
df_full_upd.loc[time_records.index, 'semantic_0'] = 'Datetime'
df_full_upd[df_full_upd['semantic_0']=='Datetime']

# %% create semantic_1 level and update race/ethnicity
df_full_upd['semantic_1'] =''
df_full_upd_match = pd.merge(df_full_upd, df_race_clean, left_on='sc_ch_raw', right_on='SC_content', how='left').fillna('')
df_full_upd_match = df_full_upd_match[(df_full_upd_match['RaceEthnic_level1']!='') & (df_full_upd_match['semantic_0']=='Race/Ethnicity')]
df_full_upd_match
# %% update the Master table
df_full_upd.loc[df_full_upd_match.index, 'semantic_1'] = df_full_upd_match['RaceEthnic_level1']
df_full_upd[df_full_upd['semantic_1']!='']
# %% save the file
df_full_upd.to_csv(outPath + 'MasterTable20210922.csv', encoding='utf-8', index=False)

#
#
#


# %% ######################################################################################
#
# Select only tables with Race/Ethnicity info in it
#
#
df_full_upd = pd.read_csv(outPath + 'TableFullRecordsFlatten3jUpd.csv', encoding='utf-8').fillna('')
df_race_table_list = df_full_upd[df_full_upd['semantic_0_clean']=='Race/Ethnicity'][['Conference','paperIdx','tableIdx']].drop_duplicates()
df_race_only_tables = pd.merge(df_full_upd, df_race_table_list, on=['Conference','paperIdx','tableIdx'],how='inner').fillna('')
df_race_only_tables.to_csv(outPath + 'RaceOnlyTables3j.csv', encoding='utf-8', index=False)
df_race_only_tables


# %% ########################### Pre-fill sample size info ################################
df_race_only_tables = pd.read_csv(outPath + 'RaceOnlyTables3j.csv', encoding='utf-8').fillna('')
df_race_only_tables.head(3)

# %% Extract sample size and % or 95% ci
df_N = df_race_only_tables[df_race_only_tables['sc_ch_raw'].str.contains('n=') | 
                    df_race_only_tables['sc_ch_raw'].str.contains('n =') |
                    df_race_only_tables['sc_ch_raw'].str.contains('N=') | 
                    df_race_only_tables['sc_ch_raw'].str.contains('N =')]
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
    df_race_only_tables.loc[index,'semantic_0_clean'] = 'sample size'
    df_race_only_tables.loc[index,'semantic_1'] = line

df_race_only_tables[df_race_only_tables['semantic_1'].str.contains('n=')]


# %%  save the new table
df_race_only_tables.to_csv(outPath + 'RaceOnlyTables3jPrefill.csv', encoding='utf-8', index=False)

#
#

# %% ###################### Query out unique terms in RaceEthnicityOnlyTables
df_race_only_tables = pd.read_csv(outPath + 'RaceOnlyTablesPrefill.csv', encoding='utf-8').fillna('')
lines = []
for index, row in df_race_only_tables.iterrows():
    raw1 = row['sc_ch_raw'].lower()
    raw2 = re.sub(r'[`\~!@#$^&*_+\[\]{};\'\\:"|<,.<>?‡§†—]', '', raw1).strip()
    raw3 = re.sub(r'[   ]', '', raw2) # remove those invisible thousand separatpr
    if raw3 != '':
        lines.append([row['sc_ch_raw'], raw3,row['tableLink']])

df_race_sc_ch = pd.DataFrame(data=lines, columns=['sc_ch_raw','sc_ch_raw_clean','tableLink'])

# %%
df_race_sc_ch_unique = df_race_sc_ch.groupby('sc_ch_raw_clean')['sc_ch_raw','tableLink'].first().reset_index()
df_race_sc_ch_unique.to_csv(outPath + 'RaceOnlyTablesUnique_SC_CH.csv', encoding='utf-8', index=False)
df_race_sc_ch_unique





# %% #############################################################################
#
#############  For Testing
#
##################################################################################
txt = 'In-home'
txt = re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‡§†—\s]', ' ', txt) # '\s' means space
txt.strip()

# %%
sentence = nlp(txt)
# get all Parts of Speech (pos)
for word in sentence:
    print(word.text,  word.pos_, word.dep_)

