# %%
import os
import os.path
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 30)
dataPath = '.\\TobaccoControl\\tableData\\tables\\'
outPath = '.\\TobaccoControl\\tableData\\'
# import spacy
# from collections import Counter
# import en_core_web_sm

# nlp = en_core_web_sm.load()


# %% 
################################### sub functions ##################################
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

### Function to extract a certain population characteristic from a table
def extractFromTable(fname, matchName, pop_type):

    inputDataFile = dataPath + fname
    # find num of header lines
    header = numHeaderLines(inputDataFile)

    pop_type_lc = pop_type.lower()

    # open the saved table
    df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
        # full_data = main_data.iloc[:, 1:] # if need to drop the first column
    
    pop_data = []
    pop_notes = []
    num_rows = len(df.iloc[:,0])
    num_cols = len(df.iloc[0,:])
    if num_cols < 2: # in the case of table 'TC.2017.338.153143.csv'
        df_value = []
        return df_value
    # Loop through first column for the pop_type and find the value groups
    for i in range(0,num_rows):
        if (pop_type in str(df.iloc[i,0])) or (pop_type_lc in str(df.iloc[i,0])):
            if (df.iloc[i,1] != '') and (df.iloc[i,1] != df.iloc[i,0]): # data in next columns
                for j in range(1, num_cols):
                    hd = df.columns[j]
                    # print(hd, type(hd))
                    if isinstance(hd, tuple):
                        hd = str(hd[0]) + ' ' + str(hd[-1])
                    else:
                        hd = str(hd)
                    pop_data.append(str(df.iloc[i,j]))
                    pop_notes.append(hd)
            else:
                for j in range(i+1, num_rows):
                    if (df.iloc[j,0]==df.iloc[j,1]) or (df.iloc[j,1]=='') or (df.iloc[j,0]==''):
                        break
                    else:
                        details = ''
                        for k in range(1, num_cols):
                            hd = df.columns[k]
                            if isinstance(hd, tuple):
                                hd = str(hd[0]) + ' ' + str(hd[-1])
                            else:
                                hd = str(hd)
                            details += '[' + str(df.iloc[j,k]) + ' (' + hd + ')] '
                        details = details.strip()
                        pop_data.append(str(df.iloc[j,0]))
                        pop_notes.append(details)
            break

    if len(pop_data) > 0:
        # find paper and image link
        df_match = df_AWS[df_AWS['filename']==matchName].iloc[0]
        jName = df_match['Conference']
        pTitle = df_match['Paper Title']
        pDOI = df_match['Paper DOI']
        authors = df_match['Author']
        pLink = df_match['paper_url']
        pCaption = df_match['cap_url']
        imgLink = df_match['url']
        pdata = [[jName, pTitle, pCaption, imgLink, pop_type, '; '.join(pop_data), 
                '; '.join(pop_notes), pDOI, pLink, authors, fname]]
        df_value = pd.DataFrame(data=pdata, columns =['Journal', 'PaperTitle', 'TableCaption', 
                    'LinkToTable', 'Parameter', 'Value', 'Notes/Details', 'PaperDOI',
                    'LinkToPaper','Authors', 'TableFile'])
    else:
        df_value = []
    
    return df_value


# %% 
######## Loop through all files to search population characteristic
dataPath = '.\\TobaccoControl\\tableData\\tables\\' # 'TC.2015.139.145231.csv' 'TC.2015.62.144905.csv'
outPath = '.\\TobaccoControl\\tableData\\'

# find link to paper and table image
df_AWS = pd.read_csv('.\\TobaccoControl\\figtblDataAWS.csv', index_col=0)
df_AWS.head()

# %%
pop_type = 'Age'
pop_df = pd.DataFrame()

# fname = 'TC.2015.139.145231.csv'
# df = extractFromTable(fname, pop_type)

for i, fname in enumerate(os.listdir(dataPath)):
    if i < 0:
        continue
    matchName = fname.replace('.csv', '_1.png')
    if i < 0: 
        break
    if (i%100)==0:
        print(i)
    df = extractFromTable(fname, matchName, pop_type)
    if len(df)>0:
        pop_df = pop_df.append(df)

pop_df = pop_df.reset_index(drop=True) 
pop_df.to_csv(outPath+'Results.'+pop_type+'.csv', encoding='utf-8')
print('Done!')

# %%
pop_df.head(10)


# %% 
### show a specific table
fname = 'TC.2017.338.153143.csv'
inputDataFile = dataPath+fname
header = numHeaderLines(inputDataFile)
df = pd.read_csv(inputDataFile, header=header, index_col=0, encoding="utf-8").fillna('')
df


# %%
###########################################################################
### Calculate the number of mentions for Ethnicity and Gender ###
###########################################################################
#
from collections import Counter
cnot = ['%', '=', '.', '(', '†', '–'] # characters that should not be present 
any_in = lambda a, b: any(i in b for i in a) # function to check inclusion

# open this saved file if not through above steps
outPath = '.\\TobaccoControl\\tableData\\'
pop_df = pd.read_csv(outPath+'Results.Ethnicity.csv', encoding='utf-8')

# standardize ethnic names
nh_alias = ['non-Hispanic', 'Non-Hispanic', 'non-hispanic', 'Non-hispanic', 'Not Hispanic']
nh_white = ['White NH', 'NH white', 'White, NH', 'NH-white', 'white, NH', 'Caucasian NH']
nh_black = ['Black NH', 'NH black', 'Black, NH', 'NH-black', 'black, NH', 'NH Black or African American']
nh_other = ['Other, NH', 'other, NH', 'Other NH', 'NH other race', 'NH others', 'NH other races', 
            'NH other', 'NH Other']
afam = ['African-American Black', 'Black African-American, NH', 'Black African-American',
            'Black African American, NH', 'Black African American, NH', 
            'African-American']
afam_or_black = ['Black or African American', 'Black or African-American']

values = []
for index, row in pop_df.iterrows():
    items = row['Value'].split('; ')
    for item in items:
        if not (item=='' or any_in(cnot,item) or item.split(' ')[0].isnumeric()):
            item_Clr = item.replace('/', ' ').replace('\xa0',' ')
            for nh in nh_alias:
                item_Clr = item_Clr.replace(nh, 'NH')
            for nh_wh in nh_white:
                item_Clr = item_Clr.replace(nh_wh, 'NH White')
            for nh_bl in nh_black:
                item_Clr = item_Clr.replace(nh_bl, 'NH Black')    
            if any_in(nh_other,item_Clr):
                item_Clr = 'NH Other'
            elif any_in(afam,item_Clr):
                item_Clr = 'Black African American'
            elif any_in(afam_or_black, item_Clr):
                item_Clr = 'Black or African American'
            # append to list
            values.append(item_Clr)

# Counter(values)

# %% 
df = pd.DataFrame.from_dict(Counter(values), orient='index').reset_index()
df = df.rename(columns={'index':'ethnicity', 0:'count'})
df = df.sort_values('ethnicity')
df.reset_index(drop=True, inplace=True)
df.to_csv(outPath+'Results.Ethnicity.Cleaned.csv')
df


# %%
#
# open the 'clean3' sheet and make some plots
#
#
xls = pd.ExcelFile(outPath+'Results.Ethnicity.xlsx')
df_clean = pd.read_excel(xls, 'clean4')
df_clean = df_clean[(df_clean['group'] != 'Other') & (df_clean['group'] != 'Non-hispanic') & 
                    (df_clean['group'] != 'Non-white')]
df_clean


# %% 
# plot bar chart
sns.set(style="whitegrid", font_scale=1.5)

df_group = df_clean.groupby(['group'])['count'].sum().reset_index()
plt.figure(figsize=(9, 6))
chart = sns.barplot(x="group", y="count", data=df_group)
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
chart.set(xlabel='General Ethnic Groups', ylabel='Number of Mentions')
plt.show()

# %%
# plot details sub-group ethnic info
sns.set(font_scale=1.2)
g = sns.FacetGrid(df_clean, col="group", col_wrap=2, sharex=False, height=4.5, aspect=1)
g.map(sns.barplot, "ethnicity", "count")
i=0
for axes in g.axes.flat:
    i += 1
    if i==6:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=0, horizontalalignment='center')
    else:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
g.set(xlabel='Ethnic Name', ylabel='Mentions')
g.fig.tight_layout()
plt.show()



# %%
###########################################################################
### Calculate the number of mentions for Age ###
###########################################################################
#
# open this saved file if not through above steps
#
outPath = '.\\TobaccoControl\\tableData\\'
pop_df = pd.read_csv(outPath+'Results.Age.csv', index_col=0, encoding='utf-8')
pop_df

# %%
cnot = ['(%', 'Missing', 'p Value', '(SD)'] # characters that should not be present 
to_replace = ['***', '**', '*', '†', '‡', '(ref)', '(Ref)', '%', '§', '¶', 'Aged', 'Age', 
                'years', '-year olds'] # replace with ''
any_in = lambda a, b: any(i in b for i in a) # function to check inclusion
no_numeric = lambda a: not any(i.isnumeric() for i in a) # check if there is no numeric

values = []
for index, row in pop_df.iterrows():
    items = row['value'].split('; ')
    for item in items:
        item_Clr = item.replace('\u2005', ' ').replace('\u2009', ' ').replace('\xa0',' ')
        if not (item_Clr=='' or any_in(cnot,item_Clr) or no_numeric(item_Clr)):
            if not (item_Clr[0]=='−' or item_Clr[0:2]=='0.' or item_Clr[0:2]=='1.' or 
                    item_Clr[0:2]=='2.' or item_Clr[0:2]=='<0'):
                for to_rp in to_replace:
                    item_Clr = item_Clr.replace(to_rp, '')
                # append to list and organize by table
                values.append(item_Clr.strip() + ';' + '.'.join(row['table'].split('.')[:3]))

Counter(values)

# %% 
### COnvert to dataframe, group by table, and count the unique mentions
df = pd.DataFrame.from_dict(Counter(values), orient='index').reset_index()
df = df.rename(columns={'index':'Age', 0:'count'})
df = df.sort_values('Age')
df.reset_index(drop=True, inplace=True)
df[['Age','paper']] = df['Age'].str.split(';',expand=True)
df.drop('count', axis=1, inplace=True)
df.to_csv(outPath+'Results.Age.Group.csv') # save the age by paper results

df = df.groupby(['Age'])['paper'].count().reset_index() # count the unique paper mentions
df.to_csv(outPath+'Results.Age.Unique.csv')
df

# %%
### read from the 'GroupClean3a' sheet where age ranges are given
#
xls = pd.ExcelFile(outPath+'Results.Age.xlsx')
df_clean = pd.read_excel(xls, 'GroupClean3a')
df_clean

# %%
values = []
for index, row in df_clean.iterrows():
    count, age = int(row['count']), str(row['Age'])
    for i in range(0,count):
        if '<=' in age:
            for j in range(0,int(age.strip('<='))+1):
                values.append(j)
        elif '<' in age:
            for j in range(0,int(age.strip('<'))):
                values.append(j)
        elif '>=' in age:
            for j in range(int(age.strip('>=')),81):
                values.append(j)
        elif '>' in age:
            for j in range(int(age.strip('>'))+1,81):
                values.append(j)
        elif '–' in age:
            st = int(age.split('–')[0])
            ed = int(age.split('–')[1].strip())
            for j in range(st,ed+1):
                values.append(j)
        else:
            values.append(int(age.strip()))

Counter(values)

# %% Convert to dataframe
# df = pd.DataFrame.from_dict(Counter(values), orient='index').reset_index()
df = pd.DataFrame(data=values)
df = df.rename(columns={0:'Age'})
df['type'] = 'range'
# df = df.sort_values('Age').reset_index(drop=True)
df

# %%
### open the 'GroupClean3b' sheet where ages are in the form of mean and SD
#
df_clean2 = pd.read_excel(xls, 'GroupClean3b')
df_clean2

# %%
values2 = []
for index, row in df_clean2.iterrows():
    age_low, age_high = int(row['Age_low']), int(row['Age_high'])
    for i in range(age_low,age_high+1):
        values2.append(i)

Counter(values2)

# %% Convert to dataframe
df2 = pd.DataFrame(data=values2)
df2 = df2.rename(columns={0:'Age'})
df2['type'] = 'm_SD'
df2

# %% combine the above two sets
df_all = df.append(df2)
df_all2 = df_all.copy()
df_all2['type'] = 'combined'
# df_all2
df_all = df_all.append(df_all2)
df_all

# %% create a violin plot
import matplotlib.ticker as ticker
sns.set(font_scale=1.2, style="whitegrid")
fig = plt.gcf()
fig.set_size_inches(8, 6)
ax = sns.violinplot(x=df_all["Age"], y=df_all["type"], cut=0) # inner="stick",
ax.set(xlim=(0, 100), ylabel='Age Format Reported', 
        xlabel='Age\n(0-98 for the \'range\' data; 11-81 for the \'m_SD\' data)')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))           
plt.show()


# %% create a histogram
sns.set(font_scale=1.2, style="whitegrid")
ax1 = sns.FacetGrid(data=df_all, col='type', col_wrap=3, height=4)
ax1.map(sns.distplot, "Age", hist=True, kde=False, bins=range(0,101))
ax1.set(xlim=(0, 100), ylabel='Number of Mentions', xlabel='Age')
ax1.fig.tight_layout()
plt.show()

# ax2 = sns.distplot(df_all["Age"], kde=True, bins=range(0,101))
# ax2.set(xlim=(0, 100), ylabel='Probability Density', xlabel='Age\n(0-98, the most studied is 18-24)')
# plt.show()



#########################################################################################
################## The following are trial by NER approach to extract key terms
######### Does not seem to work well

# %%
# get the tokens
tokens = nlp(''.join(str(df.iloc[:,0].tolist())))
items = [x.text for x in tokens.ents]
Counter(items).most_common(20)


# %% get the DATE NEs
date_list = []
for ent in tokens.ents:
    if ent.label_ == 'DATE':
        date_list.append(ent.text)
        
date_counts = Counter(date_list).most_common(10000)
df_dates = pd.DataFrame(date_counts, columns =['text', 'count'])
df_dates


# %% #################### for testing #######################
dataPath = './TobaccoControl/tableData/tables/' # 'TC.2015.139.145231.csv' 'TC.2015.62.144905.csv'

for i, fname in enumerate(sorted(os.listdir(dataPath))):
    print(i, fname)
    if i==10:
        break

# %%
