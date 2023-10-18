# %%
import os
import os.path
import pandas as pd
from collections import Counter
from pandas.core.indexes.base import Index
import seaborn as sns # restart kernal if error
import matplotlib.pyplot as plt
import spacy
import en_core_web_lg
### Note: install gensim==3.8.3 instead of the latest 
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
from transformers import pipeline

nlp = en_core_web_lg.load()
pd.set_option('display.max_colwidth', 100)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
### Functions to handle sentences and summarization
def f(seq): # Order preserving unique sentences - sometimes duplicate sentences appear in summaries
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]
 
def summary(x, perc): # x input document, perc: percentage of the original document to keep
    if len(split_sentences(x)) > 2:
        test_summary = summarize(x, word_count = 100, split=True) #  ratio = perc, 
        test_summary = ' '.join(map(str, f(test_summary)))
    else:
        print(len(split_sentences(x)))
        test_summary = x
    return test_summary


# %% 
######## Loop through all files to search population characteristic
paperData = './NLP_results_3j/paperDataOthers.csv'
outPath = './NLP_results_3j/'

df_paper = pd.read_csv(paperData, encoding='utf-8').fillna('')
del df_paper['Index']
df_paper
# print(len(df_paper['Abstract']))
# df_paper['Abstract'][1]


# %% 
# Load BERT model trained on CNN/Daily Mail News Dataset for abstractive summarizatoin
# For long summary, the BERT abstractive result is not as good as Gensim extractive result
# But BERT's shorter summary appears better
summarizer = pipeline("summarization")
### below is t5 model (https://arxiv.org/abs/1910.10683)
### summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")


# %%
### Extract key terms using NLP spacy model and summarize
#
non_loc_ch = ['≥', '=', 'pH', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lines = []

for index, row in df_paper.iterrows():
    # if index!=112 and index!=422: # single table; two tables
    if index < 0:
        continue
    print('index ', index)
    txt = row['Title'] + ' ' + row['Abstract']
    tokens = nlp(txt)
    loc_list = []
    org_list = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE': # countries, cities, states; 'LOC': non-GPE locations, mountain ranges, bodies of water
            loc_list.append(ent.text)
        elif ent.label_ == 'ORG':
            org_list.append(ent.text)
    loc_counts = Counter(loc_list).most_common(20)
    org_counts = Counter(org_list).most_common(20)
    locs = [(x[0]) for x in loc_counts]
    locs_txt = []
    # remove non-text terms and connect a string
    for loc in locs:
        if [ele for ele in non_loc_ch if(ele in loc)]:
            continue
        else:
            if loc=='US':
                loc = 'USA'
            locs_txt.append(loc)
    locs_txt = Counter(locs_txt).most_common(20)
    locs_txt = ','.join([(x[0]) for x in locs_txt])
    if locs_txt == '':
        locs_txt = 'USA (assumed)'
    # print(locs_txt)

    # get summary of abstract and highlights
    if row['Abstract'] != '':
        txt = row['Abstract']
        # print(txt, '\n')
        for tm in ['Objectives', 'Objective', 'Methods', 'Conclusions', 'Conclusion']:
            txt = txt.replace(tm, '')
        myabstract = summary(txt, 0.2)
        # print('Gensim summary:\n', myabstract, '\n')
        # For long summary, the BERT abstractive result is not as good as Gensim extractive result
        # But BERT's shorter summary appears better
        bertTopic = summarizer(txt, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
        bertTopic = '.'.join(bertTopic.split(' .')[:3])
        if bertTopic[-1] != '.':
            bertTopic += '.'
        # print('Bert 3-lines:\n', bertTopic, '\n')
    else:
        bertTopic = ''
        myabstract = ''
    if row['Highlights'] != '':
        txt = row['Highlights']
        # print('Original highlights:\n', txt, '\n')
        myhighlight = summary(txt, 0.2)
        # print('Gensim highlight:\n', myhighlight.strip(), '\n')
    else:
        myhighlight = ''

    lines.append([row['Conference'], row['Title'], row['Year'],
                locs_txt, row['Link'], row['AuthorKeywords'],
                row['Abstract'], row['Highlights'], bertTopic, myabstract, 
                myhighlight])

df_paper_loc = pd.DataFrame(lines, columns=['Conference', 'paperTitle', 'Year', 
                                'Location', 'paperLink', 'authorKeywords', 'original_abstract', 
                                'original_highlights', '3LineTopic', '100_word_abstract', 
                                '100_word_highlight'])
df_paper_loc.to_csv(outPath + 'TobaccoPapersKeySummaryOthers.csv', encoding='utf-8')

# %%
df_paper_loc.tail()



# %%
### Select papers for team review
#
df_paper_loc = pd.read_csv(outPath + 'TobaccoPapersKeySummary.csv', index_col=0, encoding='utf-8').fillna('')
df_paper_loc

# randomly select papers
# df_select = df_paper_loc[(df_paper_loc['Ethnicity']!='') | (df_paper_loc['Age']!='')]
# df_select.sample(12).to_csv(outPath + 'RandomPapersReview.csv')

# %% select specified ones
selected = [173,112,462,399,645,794,256,857,325,237,826,422]
df_select = pd.DataFrame()
for idx in selected:
    df_s = df_paper_loc.iloc[idx]
    df_select = df_select.append(df_s, ignore_index=True)
df_select = df_select.reindex(df_paper_loc.columns, axis=1)
df_select.index = selected # set index to a list
df_select.to_csv(outPath + 'TobaccoPapersKeySummaryAssign.csv', encoding='utf-8')
df_select


# %%
### Select papers for invidial review
#
outPath = './TobaccoControl/NLP_results/'
df_select = pd.read_csv(outPath + 'TobaccoPapersKeySummaryAssign.csv', index_col=0, encoding='utf-8').fillna('')
df_select.head()

# %%
# find link to paper and table image
df_AWS = pd.read_csv('./TobaccoControl/figtblDataAWS.csv', index_col=0)
df_AWS.head()

# %%
lines = []
for idx, row in df_select.iterrows():
    print(idx)
    tblCaps = row['TableCaps'].split('\n')
    tblLinks = row['TableLinks'].split('\n')
    tblimgLinks = df_AWS[(df_AWS['paper_url'] == row['PaperLink']) & (df_AWS['vis_type'] == 16)]
    for i in range(0,len(tblCaps)):
        imgLink = tblimgLinks.iloc[i]['url']
        lines.append([idx, row['PaperTitle'], row['PaperLink'], tblCaps[i], imgLink, 
                        tblLinks[i], '', ''])
    lines.append([idx, row['PaperTitle'], row['PaperLink'], row['original_highlights'], 
                    row['original_abstract'], '', '', ''])

df_ind = pd.DataFrame(lines, columns=['index', 'PaperTitle', 'PaperLink', 'TableCaption/Highlight',
                        'ImageLink/Abstract', 'TableLink', 'KnowledgeMap', 'SummaryOrInsights'])
df_ind.to_csv(outPath + 'TobaccoPapersIndividualReview.csv', encoding='utf-8', index=False)
df_ind.head()
    


# %% #################### for testing #######################
int(3+0.6)
# %%
