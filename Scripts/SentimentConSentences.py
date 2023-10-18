# %% 
# Load libraries
#
import os
import pandas as pd
from tqdm import tqdm as tqdm
from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
outPath = './SeedSentences/'

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")


# %%
# Load in the dataset
#
is_full_set = False
if is_full_set:
    df_sentences = pd.read_csv(outPath + 'ConclusionSentences.csv', index_col=0, encoding='utf8')
else:
    df_sentences = pd.read_csv(outPath + 'ConclusionSentencesPart.csv', index_col=0, encoding='utf8')
print(len(df_sentences))
df_sentences.head()


# %%
# ##################### Testing #####################
#
df_sentences['Sentiment'] = ''
df_sentences['Score'] = 0
for index, row in df_sentences.iterrows():
    if index<0:
        break
    if index % 50 == 0:
        print(index)
    sentence = row['Sentence']
    result = sentiment_analysis(sentence)
    senti = result[0]['label']
    score = result[0]['score']
    if score<0.8:
        df_sentences.loc[index,'Sentiment'] = 'neutral'
    else:
        df_sentences.loc[index,'Sentiment'] = senti.lower()
    df_sentences.loc[index,'Score'] = '%0.2f' % score

df_sentences.to_csv(outPath+'ConclusionSentencesPartUpd.csv', index=False, encoding='utf8')
df_sentences.head()


# %%
