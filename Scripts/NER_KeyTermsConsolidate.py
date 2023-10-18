# %% 
# Load libraries
#
from operator import index
import os
import pandas as pd
from collections import Counter

import spacy
from spacy import displacy
import re
import inflect  # inflect.engine() uses plural() to convert singular to plural

#### install pattern library using "conda install -c conda-forge/label/cf202003 pattern"
from pattern.en import comparative, superlative 

spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_lg')

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
inputPath = './NER/'
outPath = './NER/'


# %% #################################################################
# Read pre-defined NER tags
#    #################################################################
df_tags = pd.read_csv(inputPath + 'TobaccoNamedEntitiesV3.csv', encoding='utf8')

# global list to store the NER list
kw_lists = []    # [0] tag, [1] entity name
kw_description = []

for idx, row in df_tags.iterrows():
    # if idx != 0:
    #     continue
    kw_list = []
    tag = row['tag']
    entities = row['entities'].split(',')
    for entity in entities:
        entClean = entity.strip()
        if entClean != '':
            kw_list.append([entClean, tag])
    kw_lists.append(kw_list)
    kw_description.append(','.join([row['type'],row['subtype'],row['tagName'],tag]))

tag_classes = [(x[0][1], len(x)) for x in kw_lists]
print('Total %d tags in %d classes' % (sum([x[1] for x in tag_classes]),len(tag_classes)))
kw_description



# %%
# compare similary of two terms
term0 = nlp('lowest income')
term1 = nlp('low income')
print(term0, 'vs.', term1)
print("Similarity:", term0.similarity(term1))

# %%
# convert a singular noun to plural
word = 'knife'   # must be a singular
p = inflect.engine()
print(p.plural(word))
print("The singular of ", word, " is ", p.plural_noun(word))

# %%
# convert an adjective to comparative or superlative form
print(comparative('lower'), superlative('bad'))




# %%
# loop through raw terms to remove plural and "-"
lemma_tags_plural = ["NNS", "NNPS"]    # plural forms
lemma_tags_relative = ["JJR", "JJS", "RBR", "RBS"]    # plural forms
lemma_exceptions = ['best']
british_English = ['flavor']

lines = []

df_tags_new = df_tags
df_tags_new['entities_clean'] = ''

for idx in range(len(kw_lists)):
    if idx < 0:
        continue
    terms = sorted(kw_lists[idx])
    terms_describ = kw_description[i]        
    terms_new = []                      

    for term in terms:
        term_p = nlp(term[0])
        term_sep = []
        for token in term_p:
            lemma = token.text
            if token.tag_ in lemma_tags_plural:    # change to singular
                lemma = token.lemma_    
            if (token.tag_ in lemma_tags_relative and 
                    token.text not in lemma_exceptions):
                lemma = token.lemma_    # change to singular
            term_sep.append(lemma)

        # connect two words separated by '-'
        term_new = []
        i = 0
        while i < len(term_sep):
            tm = term_sep[i]
            if tm=='-':
            # this block is to remove "-"
                i += 1
                continue
            if tm[0]=='\'':     # case like "bachelor 's degree"
                term_new.pop()
                term_new.append(term_sep[i-1]+tm)
                i += 1
            else:
                term_new.append(tm)
                i += 1

        terms_new.append(' '.join(term_new))
    
    print(term[1], len(terms_new), end=' ')
    # remove duplicate ones
    terms_new = [x for x in Counter(terms_new)]
    print(len(terms_new))

    df_tags_new.loc[idx, 'entities_clean'] = ','.join(terms_new)
    df_tags_new.loc[idx, 'entities'] = ','.join(sorted([x.strip() for x in df_tags_new.loc[idx, 'entities'].split(',')]))
    
df_tags_new.to_csv(outPath + 'TobaccoNamedEntitiesV3clean.csv', index=False)




# %%
