# %% 
# Load libraries
#
import os
import os.path
import pandas as pd
import re
from collections import Counter

import spacy
import scispacy # SciSpacy's 4 NER models are too domain-specific
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

nlp = spacy.load('en_core_sci_lg') # en_core_web_sm, en_core_web_lg, en_core_sci_lg
# nlp.add_pipe("abbreviation_detector") # comment out to avoid screen output
# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pd.set_option('display.max_colwidth', 50)
# pd.set_option('display.max_rows', 20)
outPath = './SeedSentences/'


# %% 
# Read all paperData and select relevant columns
#
df_paper1 = pd.read_csv('./TobaccoControl/paperData.csv', encoding='utf8')
df_paper2 = pd.read_csv('./NLP_results_3j/paperDataOthers.csv', encoding='utf8')
df_paper = df_paper1.append(df_paper2).fillna('')
num_highlights = len(df_paper[df_paper['Highlights']!=''])
df_paper = df_paper[['Conference', 'Year', 'Title', 'PaperType', 'Abstract', 'Link', 'AuthorNames', 'AuthorAffiliation']]
df_paper = df_paper[df_paper['Abstract']!=''].reset_index(drop=True)
print('Num of papers with abstract:', len(df_paper))
print('Num of papers with highlights:', num_highlights)
df_paper.to_csv(outPath+'paperAbstractsAll.csv',index=False, encoding='utf8')
print(len(df_paper))


# %% 
# Get all abstracts containing 'conclusion'
#
indices = []; i=0
for index, row in df_paper.iterrows():
    abstract = row['Abstract']
    if 'conclusion' in abstract.lower():
        i += 1
        indices.append(index)
        # print(index, row['Conference'], row['PaperType'])       
df_conclusion = df_paper.loc[indices]
df_conclusion.to_csv(outPath + 'AbstractsContainConclusion.csv', index=False, encoding='utf8')
print('Num of abstracts with \'conclusion\':', i) 
df_conclusion.head()

#################### Above this line run only the first time ##########


# %% 
# Function to get the 'conclusion' part of abstract
#
def GetTextAfterWord(txt, term='conclusion'):
    words = txt.split(' ')
    pos = -1
    for word in words:
        if term.lower() in word.lower():
            pos = words.index(word)
            break
    if pos >= 0:
        txttail = ' '.join(words[pos+1:])
        return txttail
    else:
        return ''

# %%
# Function to extract UMLS entity and sematic types
#
def getEntitySemantic(sent):
    doc = nlp(sent)
    results = []
    for ent in doc.ents: 
        # not every entity identified has a UMLS match; below shows matched ones
        for umls_ent in ent._.kb_ents:
            concept_ID, score = umls_ent
            # get entity type code
            ent_types = linker.kb.cui_to_entity[umls_ent[0]].types
            # get reference to the KB's semantic tree
            semantic_tree = linker.kb.semantic_type_tree
            # get the sematic name of a type
            stycodes = [(stycode, semantic_tree.get_canonical_name(stycode)) 
                    for stycode in ent_types]
            # results.append('%s (%0.1f, %s)' % (ent.text, score, stycodes[0][1]))
            results.append('%s (%s)' % (ent.text, stycodes[0][1]))
            break # show only the first (highest score) entity & semantic
    
    if len(results)==0:
        return ''
    else:
        return '; '.join(results)


# %% 
# Read in the abstracts for annotation
#
df_conclusion = pd.read_csv(outPath+'AbstractsContainConclusion.csv', encoding='utf8')
print('Num of \'conclusion\' abstracts:', len(df_conclusion))


# %%
# Extract sentences from the abstracts, and NERs from sentences
#
sentences = []
for index, row in df_conclusion.iterrows():
    if index % 20 == 0:
        print(index)
    abstract = row['Abstract']
    # get the part of abstract after 'conclusion'
    txttail = GetTextAfterWord(abstract)
    txtnlp = nlp(txttail)
    # split into sentences
    sents = list(txtnlp.sents)
    # get NERs for each sentence
    for sent in sents:
        ners = getEntitySemantic(str(sent))
        sentences.append([row['Conference'], row['Year'], row['Title'], str(sent), ners, row['Link']])

# save the results
df_sentences = pd.DataFrame(data=sentences, columns=['Conference','Year','Title','Sentence','NamedEntity','Link'])
df_sentences.to_csv(outPath + 'ConclusionSentences.csv', encoding='utf8')
print(len(df_sentences))
df_sentences.head()


# %%
# random select a fraction of abstracts
#
df_sentences_part = df_sentences.sample(frac=0.1, random_state=6).reset_index(drop=True)
df_sentences_part.to_csv(outPath + 'ConclusionSentencesPart.csv', encoding='utf8')
print('Num of randomly selected abstracts:', len(df_sentences_part))


# %% #########################################
# testing
sent = 'Nicotine Metabolite Ratio is Associated With smoking'
getEntitySemantic(sent)


# %%
