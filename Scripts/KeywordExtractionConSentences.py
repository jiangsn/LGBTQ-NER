# %% 
# Load libraries
#
import os
import pandas as pd
import spacy
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nlp = spacy.load('en_core_sci_lg')
outPath = './SeedSentences/'


# %%
# Load in the dataset
#
is_full_set = True
if is_full_set:
    df_sentences = pd.read_csv(outPath + 'ConclusionSentences.csv', index_col=0, encoding='utf8')
else:
    df_sentences = pd.read_csv(outPath + 'ConclusionSentencesPart.csv', index_col=0, encoding='utf8')
print(len(df_sentences))
df_sentences.head()


# %%
################## Extract key words from data
#
tokens_to_skip = ['finding','result','study','research','researcher',
                  'smoke','method','evaluate','requirement',
                  'change','literature','day','time','state',
                  'reduce','information','compare',
                  'estimate','increase','decrease','analysis',
                  'implement','investigate','year','month','assess',
                  'smoking','smoker','level','content']

tokens_to_replace = {'us':'USA', 'united-states':'USA', 
                     'associate-with':'association',
                     'cigarettes':'cigarette',
                     'cigarette-smoking':'cigarette-smoking',
                     'cessation':'smoking-cessation',
                     'smoke-cessation':'smoking-cessation',
                     'young-adults':'young-adult',
                     'electronic-cigarette':'e-cigarette',
                     'electronic-cigarettes':'e-cigarette',
                     'adolescents':'adolescent',
                     'youth':'youth',
                     'adult':'adult','adults':'adult',
                     'secondhand':'second-hand',
                     'secondhand-smoke':'second-hand',
                     'nicotine':'nicotine',
                     'or':'odds-ratio',
                     'ci':'confidence-interval'}

extracted_words = []

for idx, row in df_sentences.iterrows():
    if idx<0:
        break
    if idx % 50 == 0:
        print(idx)
    text = row['Sentence']
    doc = nlp(text)
    # doc: token.text, token.pos_, token.tag_, token.dep_, token.lemma_
    
    for ent in doc.ents:
        tokens = ent.lemma_.split(' ')
        if len(tokens)>1:
            phrase = '-'.join(tokens) 
        else:
            phrase = tokens[0]
        if not phrase.lower() in tokens_to_skip:
            if phrase.lower() in tokens_to_replace:
                phrase = tokens_to_replace[phrase.lower()]
            extracted_words.append(phrase)

# %%
num_to_plot = 100

token_counts = Counter(extracted_words).most_common(num_to_plot)
print(token_counts)

extracted_tokens = [token[0] for token in token_counts 
                  for i in range(token[1])]
random.shuffle(extracted_tokens)
extracted_text = ' '.join(extracted_tokens)


# %%
################ Plot WordCloud map
#
wordcloud = WordCloud(width = 1000, height = 800,
                background_color ='black',
                stopwords = None, max_words=500,
                min_font_size = 16,
                random_state = 2).generate(extracted_text)

# plot the WordCloud image                      
plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
if is_full_set:
    plt.savefig(outPath+'abstract_conclusion_top_%d_words.png' % num_to_plot)
else:
    plt.savefig(outPath+'partial_con_sentences_top_%d_words.png' % num_to_plot)

plt.show()


# %%
# ##################### Testing #####################
#
extracted_text = ' '.join(extracted_tokens)
print(extracted_text)

# %%
