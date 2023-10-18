# %% 
# Load libraries
#
from operator import index
import os
import pandas as pd
from collections import Counter

import spacy
from spacy import displacy
from spacy.tokens import DocBin
import json
from datetime import datetime
from tqdm import tqdm
import re

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
outPath = './NER/'


# %% #################################################################
# Read pre-defined NER tags
#    #################################################################
df_tags = pd.read_csv(outPath + 'TobaccoNamedEntitiesV3.csv', encoding='utf8')

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
def structure_training_data(text, kw_list):
    """
    text is a string with cases
    kw[0] is tag name, e.g., 'M-sts'
    kw[1] is entity name, e.g., '95% CI'
    """
    results = []
    entities = []

    # search for instances of keywords within the text (ignoring letter case)
    for kw in kw_list:
        # \b are word boundaries, and s is optional
        search = re.finditer(r'\b'+kw[1]+r's?\b', text, flags=re.IGNORECASE)
        
        # store the start/end character positions
        all_instances = [[m.start(),m.end()] for m in search] 
        
        # if the callable_iterator found matches, create an 'entities' list
        if len(all_instances)>0:
            for i in all_instances:
                start = i[0]
                end = i[1]
            ### case 1: if a single-word entity is within a phrase entity, ignore it
            ### case 2: if an entity phrase envelops a found entity, remove the found entity
                already = False
                for ent in entities:
                    if ent[0]<=start and ent[1]>=end:   # case 1
                        already = True          # break immediately
                        break
                    if start<=ent[0] and ent[1]<=end:   # case 2
                        entities.remove(ent)    # remove as many as is enveloped
                if not already:
                    entities.append((start, end, kw[0])) 
                
    # add any found entities into a JSON format within collective_dict
    if len(entities)>0:
        results = [text, {"entities": entities}]
        # this is a global dictionary containing all annotated examples
        collective_dict['TRAINING_DATA'].append(results)
        return True
    else:
        print('No entities identified: ', end='')
        return False


# %%        
# Test the label-entity function

text = ('nicotine and nicotine replacement therapy Family Smoking '
        'Prevention and Tobacco Control Act is enacted.'
        ' smokeless tobacco are used more in China than in the United '
        'States. smoking cesation is the goal to pursue when the')

# this dictionary will contain all annotated examples
collective_dict = {'TRAINING_DATA': []}

structure_training_data(text, kw_list)
print(collective_dict['TRAINING_DATA'])


# %% ####################################################
# Read in text dataset and split into train, dev, and test
#
df_text = pd.read_csv('./LGBTQ/lgbtq_articles_clean.csv', encoding='utf8')

# shuffle the dataset
df_text = df_text.sample(frac = 1, random_state=6)
# split into train, val, and test
# train_num = int(len(df_text) * 0.8)
# dev_num = int(len(df_text) * 0.1)

# df_train = df_text[:train_num].reset_index()
# df_dev = df_text[train_num:train_num+dev_num].reset_index()
# df_test = df_text[train_num+dev_num:].reset_index()
# df_test.to_csv(outPath + 'testDataset.csv', index=False)

# print(train_num, len(df_train), dev_num, len(df_dev), len(df_test))

# %% 
# evaluate a specific abstract -- skip this when generating SPACY file
evalulate = True
if evalulate:
    collective_dict = {'TRAINING_DATA': []}
    idx = 166
    # structure_training_data(df_test.loc[idx,'abstract'], kw_list)
    text ='Tianjin, China Electronic cigarettes are used more.'
    structure_training_data(text, kw_list)
    # print(df_train.loc[idx,'abstract'])
    print(collective_dict['TRAINING_DATA'])


# %% #################################################################
# Label the training data using predefined kw_list
# ####################################################################
dataset = 'dev'   # set the train or dev or all here
df_data = df_text
# if dataset == 'train':
#     df_data = df_text # df_train
# elif dataset == 'dev':
#     df_data = df_dev
# else:       # use all data for training
#     df_data = None

collective_dict = {'TRAINING_DATA': []}
for idx, row in tqdm(df_data.iterrows()):
    text = row['Abstract'].replace('\u2005','') # replace the unicode space
    result = structure_training_data(text, kw_list)
    if not result:
        print(idx)
    
print(dataset, len(collective_dict['TRAINING_DATA']))


# ####################################################################
# Convert training data to Spacy Doc object (for v3)
# ####################################################################

# create a blank model
nlp = spacy.blank('en')

def create_training(TRAIN_DATA):
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA):
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annot["entities"]:
            # alignment_mode = 'strict' / 'contract' / 'expand'
            # https://spacy.io/api/doc
            span = doc.char_span(start, end, label=label, 
                                alignment_mode="expand") 

            # skip if the character indices do not map to a valid span
            if span is None:
                print("Skipping entity.", text[start-1:end+1], label)
            else:
                ents.append(span)
                # handle erroneous entity annotations by removing them
                try:
                    doc.ents = ents
                except:
                    # print("BAD SPAN:", span, "\n")
                    ents.pop()
        doc.ents = ents

        # pack Doc objects into DocBin
        db.add(doc)
    return db

# 
# create training data and exprot to spacy format
# %%
TRAIN_DATA = collective_dict['TRAINING_DATA']
TRAIN_DATA_DOC = create_training(TRAIN_DATA)

TRAIN_DATA_DOC.to_disk(outPath + "TEST_DATA_ALL_LGBTQ_V3.spacy")



# %%
print(TRAIN_DATA)

# %%
