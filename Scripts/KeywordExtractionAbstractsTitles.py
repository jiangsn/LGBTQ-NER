# %% 
# Load libraries
#
import os
import pandas as pd
import spacy
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nlp = spacy.load('en_core_sci_lg')
outPath = './SeedSentences/'


# %%
# Load in the Abstract dataset
#
df_all = pd.read_csv(outPath + 'paperAbstractsAll.csv', encoding='utf8')
is_abstract = True

if is_abstract:
    df_source = df_all['Abstract'].reset_index()
else:    
    df_source = df_all['Title'].reset_index()

df_source.columns = ['index','text']
print(len(df_source))
df_source.head()


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

for idx, row in df_source.iterrows():
    if idx<0:
        break
    if idx % 100 == 0:
        print(idx)
    text = row['text']
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
# select top n keywords
#
num_to_plot = 50

token_counts = Counter(extracted_words).most_common(num_to_plot)
print(token_counts)

extracted_tokens = [token[0] for token in token_counts 
                  for i in range(token[1])]
random.shuffle(extracted_tokens)
extracted_text = ' '.join(extracted_tokens) # repeated text for Word Cloud

# save data for bar chart
token_n = [[tk[0],tk[1]] for tk in token_counts]
df_tokens = pd.DataFrame(data=token_n, columns=['Keyword','Count'])



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
if is_abstract:
    plt.savefig(outPath+'abstracts_top_%d_words.png' % num_to_plot)
else:
    plt.savefig(outPath+'titles_top_%d_words.png' % num_to_plot)

plt.show()


# %%
################ Plot bar chart
#
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.3)
fig, ax = plt.subplots(figsize=(7, 12))
g = sns.barplot(x="Count", y="Keyword", data=df_tokens)
g.set_ylabel('Top Words')
if is_abstract:
    plt.title('Top %d Words from Paper Abstracts' % num_to_plot)
    plt.tight_layout(pad = 0.5)
    plt.savefig(outPath+'abstracts_top_%d_words_bar_plot.png' % num_to_plot, dpi=150)
else:
    plt.title('Top %d Words from Paper Titles' % num_to_plot)
    plt.tight_layout(pad = 0.5)
    plt.savefig(outPath+'titles_top_%d_words_bar_plot.png' % num_to_plot, dpi=150)




# %%
# Prepare texts for Seed NER -- combine title, abstract, and highlights
#
df_all = pd.read_csv(outPath + 'paperAbstractsAll.csv', encoding='utf8')

lines = []
for index, row in df_all.iterrows():
    abs = row['Abstract']
    tlt = row['Title']
    lines.append(tlt + ' ' + abs + '\n')

with open(outPath+'titles_abstracts_all.csv', 'w') as f:
    f.writelines(lines)

# %%
import nltk
nltk.download('punkt')


# %%
# ##################### Testing #####################
#
tk = 'Cigarettes Smoking'
doc = nlp(tk)
for ent in doc.ents:
    print(ent.lemma_)
# %%
