# %% 
# Load libraries
#
from operator import index
import os
from xml.dom import INDEX_SIZE_ERR
import pandas as pd
from collections import Counter, deque

import json
from datetime import datetime
from tqdm import tqdm
import re
from datetime import datetime
import spacy

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
inputPath = './NER/'
preAnnoPath = './NER_AnnoSuite/'
outPath = './TextAnnoSuite/'

# load the CPU-trained model 
modelPath = "./NER_AnnoSuite/model_expand_all/model-best" 
nlp_model = spacy.load(modelPath)  # !!! make sure use the right model !!!


# %% 
# Read in text dataset and process the data into Rui's format
#
df_text = pd.read_csv(inputPath+'testDataset.csv', encoding='utf-8').fillna('')

df_train = df_text.reset_index()   ###### the first 50 papers
df_train['M_doc_id'] = ''
df_train['M_doc_info'] = ''
df_train['M_doc_content'] = ''
df_train['M_display_order'] = ''
for idx, row in df_train.iterrows():
    df_train.loc[idx,'M_doc_id'] = 'd' + str(idx+1)
    df_train.loc[idx,'M_doc_info'] =', '.join([row['Conference'], str(row['Year']), row['PaperType'], row['Link']])
    df_train.loc[idx,'M_display_order'] = str(idx+1)
    txt = 'Title: ' + row['Title'] + '. ' + row['Abstract'] + ('Highlights: ' + row['Highlights'] if row['Highlights']!='' else '')
    txt = txt.replace(', ', ' , ').replace('. ', ' . '). \
            replace('? ', ' ? ').replace('(', '( ').replace(")", " )"). \
            replace('/', ' / ').replace('“', '“ ').replace('”', ' ”'). \
            replace('=', ' = ')
    txt = ' '.join(txt.split())     # to remove extra space (2 or more spaces)
    df_train.loc[idx,'M_doc_content'] = txt
df_doc = df_train[['M_doc_id','M_doc_info','M_doc_content','M_display_order']]

# save the document.csv file
df_doc.to_csv(outPath + 'documents.csv', encoding='utf_8_sig', index=False)
df_doc.head()


# %%
### Predict and save results to annotations.csv per Rui's format

collective_dict_anno = []
doc_ids = []

for idx, row in df_doc.iterrows():
    text = row['M_doc_content']
    tokens = text.split(' ')    # used to locate start/end word index
    
    # build a hashmap of {character pos: tken pos} for lookup 
    hash = {}
    pos = 0
    for i in range(len(tokens)):
        hash[pos] = i
        pos += len(tokens[i]) + 1   # 1 is length of space

    doc = nlp_model(text)   # model prediction
    entities = []
    entities_anno = []
    for e in doc.ents:
        st = e.start_char
        while st>1 and text[st-1] != ' ':   # to handle 'non-menthol' case where model
            st -= 1                         # returns 'm' as e.start_char
        start = hash[st]
        ed = start + len(e.text.split(' ')) - 1
        entities.append({"\"type\"":"\"annotation\"", 
            "\"text\"":"\"%s\"" % text[st:e.end_char],
            "\"start\"": start, "\"end\"": ed,     # Rui's foramt: start=end if one word 
            "\"category\"": "\"%s\"" % e.label_,
            "\"date\"":"\"%s\"" % datetime.now().strftime("%D %H:%M:%S")})
        entities_anno.append([st, e.end_char, e.label_]) 
    # collective_dict.append(entities)
    # save the character-level labels for comparison
    collective_dict_anno.append([text, {"entities": entities_anno}])
    # save the ID for use in annotation.csv
    doc_ids.append(row['M_doc_id'])
    if not doc:
        print(idx)

### Save prediciton at character level for comparison to pre-anno labels
with open(outPath + "testSetPredLabels.json", "w") as f:
    json.dump(collective_dict_anno, f)
    print(len(collective_dict_anno))


##################################################################################

# %% 
### Compare pre-anno labels and predicted labels and merge
with open(preAnnoPath + 'testSetAutoLabels.json') as f:
    preAnno = json.load(f)
    print(len(preAnno))

# %%    
### Compare and update
from collections import deque

predicted = collective_dict_anno
collective_dict = []    # to save updated results

for idx, preA in enumerate(preAnno):   
    if idx%20==0: print(idx)
    # if idx != 237: continue
    text = preA[0]
    # print(text)

    tokens = text.split(' ')    # used to locate start/end word index
    # build a hashmap of {character pos: tken pos} for lookup 
    hash = {}
    pos = 0
    for i in range(len(tokens)):
        hash[pos] = i
        pos += len(tokens[i]) + 1   # 1 is length of space

    anno, pred = deque(), deque()
    for an in preA[1]['entities']:
        anno.append(an)
    for pr in predicted[idx][1]['entities']:
        pred.append(pr)

    res = []
    an = anno.popleft() if anno else None
    pr = pred.popleft() if pred else None
    while anno and pred:
        if an[0]<pr[1] and an[1]>pr[0]: # overlap
            ptxt = text[pr[0]:pr[1]]
            txt = text[min(an[0],pr[0]) : max(an[1],pr[1])]
            # print(an[0], text[an[0]:an[1]], pr[0], ptxt)
            start = hash[min(an[0],pr[0])]
            ed = start + len(txt.split(' ')) - 1
            label = an[2]
            # print('%s (%s) %d (%d) \'%s\' (%s)' % (an[2],pr[2],an[0],pr[0],txt,ptxt))
            an = anno.popleft()
            pr = pred.popleft()
        elif an[0] > pr[1]:   # pr in front
            # print('single pr %d \'%s\'' % (pr[0], preA[0][pr[0]:pr[1]]))
            txt = text[pr[0]:pr[1]]
            start = hash[pr[0]] if pr[0] in hash else -1
            ed = start + len(txt.split(' ')) - 1
            label = pr[2]
            pr = pred.popleft()
        elif pr[0] > an[1]: # an in front
            txt = text[an[0]:an[1]]
            start = hash[an[0]] if an[0] in hash else -1
            ed = start + len(txt.split(' ')) - 1
            label = an[2]
            # print('single an %d \'%s\'' % (an[0], txt))
            an = anno.popleft()
        if start > -1:
            res.append({"\"type\"":"\"annotation\"", 
            "\"text\"":"\"%s\"" % txt,
            "\"start\"": start, "\"end\"": ed,     # Rui's foramt: start=end if one word 
            "\"category\"": "\"%s\"" % label,
            "\"date\"":"\"%s\"" % datetime.now().strftime("%D %H:%M:%S")})

    while anno:
        an = anno.popleft()
        txt = text[an[0]:an[1]]
        start = hash[an[0]]
        ed = start + len(txt.split(' ')) - 1
        label = an[2]
        res.append({"\"type\"":"\"annotation\"", 
            "\"text\"":"\"%s\"" % txt,
            "\"start\"": start, "\"end\"": ed,     # Rui's foramt: start=end if one word 
            "\"category\"": "\"%s\"" % label,
            "\"date\"":"\"%s\"" % datetime.now().strftime("%D %H:%M:%S")})
    while pred:
        pr = pred.popleft()
        txt = text[pr[0]:pr[1]]
        start = hash[pr[0]]
        ed = start + len(txt.split(' ')) - 1
        label = pr[2]
        res.append({"\"type\"":"\"annotation\"", 
            "\"text\"":"\"%s\"" % txt,
            "\"start\"": start, "\"end\"": ed,     # Rui's foramt: start=end if one word 
            "\"category\"": "\"%s\"" % label,
            "\"date\"":"\"%s\"" % datetime.now().strftime("%D %H:%M:%S")})

    collective_dict.append(res)


# %% check
# for an in preAnno[0][1]['entities']:
#     print(an[0], an[1], preAnno[0][0][an[0]:an[1]])
for pr in predicted[0][1]['entities']:    
    print(pr[0], pr[1], predicted[0][0][pr[0]:pr[1]])
    
##################################################################################



# %%
### Save results to annotations.csv per Rui's format
lines = []
lines.append('id,doc_id,username,entities,annotation_log,log_dates,is_error_doc,need_discuss,marked_fun,marked_OK\n')
users = ['XZ', 'YF', 'RL', 'ML', 'JC', 'CS', 'SM', 'AH']

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
### gnerate users.csv file
hd = 'M_username,M_assignment_by_doc_id,user_color\n'
users = ['XZ', 'YF', 'RL', 'ML', 'JC', 'CS', 'SM', 'AH']
assignments = ';'.join(doc_ids)

lines = [hd]
for u in users:
    line = ','.join([u, assignments, '\n'])
    lines.append(line)
with open(outPath + 'users.csv', 'w') as f:
    f.writelines(lines)



# %%
# TESTING -- label time and time period
s = 'Title: Situational and Demographic Factors in the Sudden Growth of Pall Mall , 2002–2014 . Objective Pall Mall gained significant brand share in the cigarette market between 2002 and 2013 .'
print(s.find('between'))
tokens = s.split(' ')
print(tokens.index('between'))

hash = {}
pos = 0
for i in range(len(tokens)):
    hash[pos] = i
    pos += len(tokens[i]) + 1   # 1 is length of space
hash    

# %%
from collections import deque
a = deque()
for an in anno:
    a.append(an)
a.popleft()


# %%
a
# %%
