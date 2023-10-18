# %% 
# Load libraries
#
from operator import index
import os
import pandas as pd
from collections import Counter

import json
from datetime import datetime
from tqdm import tqdm
import re
from datetime import datetime

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
inputPath = './NER/'
outPath = './SpacyProdigy/'


# %% #################################################################
# Read pre-defined NER tags
#    #################################################################
df_tags = pd.read_csv(inputPath + 'TobaccoNamedEntitiesV3.csv', encoding='utf8')

# global list to store the NER list
kw_list = []    # [0] tag, [1] entity name

for idx, row in df_tags.iterrows():
    # if idx != 25:
    #     continue
    tag = row['tag']
    entities = row['entities'].split(',')
    for entity in entities:
        entClean = entity.strip()
        if entClean != '':
            kw_list.append([tag, entClean])

tag_classes = Counter([x[0] for x in kw_list])
print('Total %d tags in %d classes' % (len(kw_list),len(tag_classes)))
tag_classes


# %%
# Function to label entities in text
#
def find_sub_list(sl,l):    # find a sublist within a list
    # results in (start, end) indices
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def find_time_period(s):
    # label time and time period
    reY = r"\d{1,4}\s\byears?\b|\d{1,4}-\byears?\b|\d{1,4}\sto\s\d{1,4}\s\byears?\b|\d{1,4}\sto\s\d{1,4}|\d{1,4}-\d{1,4}|\d{1,4}–\d{1,4}|\d{4}|"
    reM = r"\d{1,4}\s\bmonths?\b|\d{1,4}-\bmonth?\b|"
    reW = r"\d{1,4}\s\bweeks?\b|\d{1,4}-\bweeks?\b|"
    reD = r"\d{1,4}\s\bdays?\b|\d{1,4}-\bdays?\b|"
    reH = r"\d{1,4}\s\bhours?\b|\d{1,4}-\bhours?\b"
    reAll = reY+reM+reW+reD+reH
    search = re.finditer(reAll, s, flags=re.IGNORECASE)
    all_instances = [[m.start(),m.end()] for m in search]
    return all_instances


def structure_training_data(text, kw_list):
    """
    text is a string with cases
    kw[0] is tag name, e.g., 'M-sts'
    kw[1] is entity name, e.g., '95% CI'
    """
    results = []
    entities = []
    tokens = text.split(' ')    # used to locate start/end word index

    # add a special 'time' kw at the end
    kw_list_ext = kw_list + [['B-tme','#time#']]

    # search for instances of keywords within the text (ignoring letter case)
    for kw in kw_list_ext:
        if kw[1] == '#time#':    # call the speical function on time terms
            all_instances = find_time_period(text)
        else:
            # \b are word boundaries, and s is optional
            search = re.finditer(r'\b'+kw[1]+r's?\b', text, flags=re.IGNORECASE)
            # store the start/end character positions
            all_instances = [[m.start(),m.end()] for m in search] 
        
        # if the callable_iterator found matches, create an 'entities' list
        if len(all_instances)>0:
            prev = ''    # check if instance is the same as prior one (single vs. plural)
            for i in all_instances:
                s, e = i[0], i[1]
                if text[s : e] == prev:   # use string as there may be 'nicotine' vs 'Nicotine'
                    continue
                else:
                    prev = text[s : e]
                tks = text[s : e].split(' ')    # split into words
                results = find_sub_list(tks,tokens)
                for result in results:
                    start, end = result[0], result[1]
                    # print(start, end)
                ### case 1: if a single-word entity is within a phrase entity, ignore it
                ### case 2: if an entity phrase envelops a found entity, remove the found entity
                    already = False
                    for ent in entities:
                        if ent['\"start\"']<=start and ent['\"end\"']>=end:   # case 1
                            already = True          # break immediately
                            break
                        if start<=ent['\"start\"'] and ent['\"end\"']<=end:   # case 2
                            entities.remove(ent)    # remove as many as is enveloped
                    if not already:
                        entities.append({"\"type\"":"\"annotation\"", 
                            "\"text\"":"\"%s\"" % text[s : e],
                            "\"start\"":start, "\"end\"":end, 
                            "\"category\"": "\"%s\"" % kw[0],
                            "\"date\"":"\"%s\"" % datetime.now().strftime("%D %H:%M:%S")}) 
                
    """ {""type"":""annotation"",""text"":""association"",""start"":195,""end"":195,
            ""category"":""T-com"",""date"":""5/23/2022, 13:59:23""}
    """

    # add any found entities into a JSON format within collective_dict
    if len(entities)>0:
        # this is a global list containing all annotated examples
        collective_dict.append(entities)
        return True
    else:
        collective_dict.append('')
        print('No entities identified: ', end='')
        return False


# %%        
# Test the label-entity function
########## Note: must insert a space between , and .
# text = 'Title: Effects of Nicotine and Nicotine Expectancy on Attentional Bias for Emotional Stimuli . Introduction: Nicotine’s effects on mood are thought to enhance its addictive potential . However , the mechanisms underlying the effects of nicotine on affect regulation have not been reliably demonstrated in human laboratory studies . We investigated the effects of nicotine abstinence ( Experiment 1 ) , and nicotine challenge and expectancy ( Experiment 2 ) on attentional bias towards facial emotional stimuli differing in emotional valence . Methods: In Experiment 1 , 46 nicotine-deprived smokers were randomized to either continue to abstain from smoking or to smoke immediately before testing . In Experiment 2 , 96 nicotine-deprived smokers were randomized to smoke a nicotinized or denicotinized cigarette and to be told that the cigarette did or did not contain nicotine . In both experiments participants completed a visual probe task , where positively valenced ( happy ) and negatively valenced ( sad ) facial expressions were presented , together with neutral facial expressions . Results: In Experiment 1 , there was evidence of an interaction between probe location and abstinence on reaction time , indicating that abstinent smokers showed an attentional bias for neutral stimuli . In Experiment 2 , there was evidence of an interaction between probe location , nicotine challenge and expectation on reaction time , indicating that smokers receiving nicotine , but told that they did not receive nicotine , showed an attentional bias for emotional stimuli . Conclusions: Our data suggest that nicotine abstinence appears to disrupt attentional bias towards emotional facial stimuli . These data provide support for nicotine’s modulation of attentional bias as a central mechanism for maintaining affect regulation in cigarette smoking .'        
text = 'Title: Situational and Demographic Factors in the Sudden Growth of Pall Mall , 2002–2014 . Objective Pall Mall gained significant brand share in the cigarette market between 2002 and 2013 . We sought to determine whether demographic shifts occurred among the participants reporting Pall Mall as their usual brand during this time span . Method We examined National Survey of Drug Use and Health ( NSDUH ) data from 2002–2014 . Demographic characteristics included age , education , ethnicity , income , and cigarette use ( cigarettes per day , daily / non-daily smoking , and nicotine dependence ) . We also examined RJ Reynolds investor reports and shareholder documents to determine the impact of tobacco marketing on the growth of Pall Mall . Results Over 2002–2014 , Pall Mall has gained among smokers 26 to 34 years of age . More Pall Mall smokers in 2014 report higher incomes ( over $75000 ) , and also report lower scores on measures of cigarette dependence , compared to 2002 . Pall Mall smokers over time seem to share characteristics of premium cigarette brands smokers . Conclusion The profile of the typical Pall Mall smoker has changed as the brand has gained market share . An association exists between brand positioning and economic forces , which has contributed to an increase in the market share for Pall Mall . Implications It is well known that cigarette marketing drives the sale of tobacco products . The growth in the market share of Pall Mall serves as an excellent example to demonstrate how economic uncertainty paired with brand positioning and advertising worked together to serve as a catalyst for the rapid growth observed for this brand . This paper also looked at various demographic changes that occurred among Pall Mall smokers over a 12 year period and compared them to smokers of all other cigarette brands . The results of this analysis demonstrate the importance of monitoring trends over time among cigarette smokers .'
text = '2002–2014 . 5-20 , 1998 , 1999, supply / demand'
source = "Tobacco Control"

# this dictionary will contain all annotated examples
collective_dict = []
structure_training_data(text, kw_list)
for item in collective_dict:
    print(str(item).replace('\'', '\"'))
    # print(json.dumps(item))


# %% ####################################################
# Read in text dataset and process the data into Rui's format
#
df_text = pd.read_csv(inputPath+'testDataset.csv', encoding='utf-8').fillna('')

# shuffle the dataset
df_text = df_text.sample(frac = 1, random_state=6)

df_train = df_text[:20].reset_index()
df_train['M_doc_id'] = ''
df_train['M_doc_info'] = ''
df_train['M_doc_content'] = ''
df_train['M_display_order'] = ''
for idx, row in df_train.iterrows():
    df_train.loc[idx,'M_doc_id'] = 'd' + str(idx+1)
    df_train.loc[idx,'M_doc_info'] =', '.join([row['Conference'], str(row['Year'])])
    df_train.loc[idx,'M_display_order'] = str(idx+1)
    txt = 'Title: ' + row['Title'] + '. ' + row['Abstract'] + ('Highlights: ' + row['Highlights'] if row['Highlights']!='' else '')
    txt = txt.replace(', ', ' , ').replace('. ', ' . '). \
            replace('? ', ' ? ').replace('(', '( ').replace(")", " )").replace('/', ' / ')
    txt = ' '.join(txt.split())     # to remove extra space (2 or more spaces)
    df_train.loc[idx,'M_doc_content'] = txt
df_train = df_train[['M_doc_id','M_doc_info','M_doc_content','M_display_order']]
df_train.columns = ['M_doc_id','M_doc_info','M_doc_content','M_display_order']

# save the document.csv file
df_train.to_csv(outPath + 'documents.csv', encoding='utf_8_sig', index=False)
df_train.head()


# %% #################################################################
# Label the training data using predefined kw_list
# ####################################################################

collective_dict = []
doc_ids = []
for idx, row in tqdm(df_train.iterrows()):
    text = row['M_doc_content'].replace('\u2005','') # replace the unicode space
    result = structure_training_data(text, kw_list)
    doc_ids.append(row['M_doc_id'])
    if not result:
        print(idx)


# %%
# write out to annotations.csv per Rui's format
lines = []
lines.append('id,doc_id,username,entities,annotation_log,log_dates,is_error_doc,need_discuss,marked_fun,marked_OK\n')
users = ['ML','RL', 'JC', 'CS', 'SM', 'AH', 'QY']

cnt = 0
for user in users:
    i = 0
    for item in collective_dict:
        entities = str(item).replace('\'', '\"')
        line = [str(cnt), doc_ids[i], user, '' if entities=='' else '\"%s\"' % entities]
        line = ','.join(line) + ',,,0,0,0,0\n'
        lines.append(line)
        i += 1
        cnt += 1

with open(outPath + 'annotations.csv', 'w') as f:
    f.writelines(lines)

        
# %%
# gnerate users.csv file
hd = 'M_username,M_assignment_by_doc_id,user_color\n'
users = ['XZ', 'YF', 'RL', 'ML', 'JC', 'CS', 'SM', 'AH', 'QY']
assignments = ';'.join(doc_ids)

lines = [hd]
for u in users:
    line = ','.join([u, assignments, '\n'])
    lines.append(line)
with open(outPath + 'users.csv', 'w') as f:
    f.writelines(lines)



# %%
# TESTING -- label time and time period
s = 'smokers 26 to 34 years, in a 6-year period 3-weeks and 6 weeks from 2005 to 2017. 15 months. a 5-hour and 6 hours, 3-day trip 20 days, 5-20'

reY = r"\d{1,4}\s\byears?\b|\d{1,4}-\byears?\b|\d{1,4}\sto\s\d{1,4}\s\byears?\b|\d{1,4}\sto\s\d{1,4}|\d{1,4}-\d{1,4}|"
reM = r"\d{1,4}\s\bmonths?\b|\d{1,4}-\bmonth?\b|"
reW = r"\d{1,4}\s\bweeks?\b|\d{1,4}-\bweeks?\b|"
reD = r"\d{1,4}\s\bdays?\b|\d{1,4}-\bdays?\b|"
reH = r"\d{1,4}\s\bhours?\b|\d{1,4}-\bhours?\b"
reAll = reY+reM+reW+reD+reH
# search = re.findall(reAll, s)
search = re.finditer(reAll, s)

all_instances = [[m.start(),m.end()] for m in search]
all_instances


# %%
s = 'smokers 26 to 34 years, in a 6-year period 3-weeks and 6 weeks from 2005 to 2017. 15 months. a 5-hour and 6 hours, 3-day trip 20 days, 5-20'
find_time_period(s)


# %%
x = kw_list + ['B-tme','#time#']
x[-5:]
# %%
