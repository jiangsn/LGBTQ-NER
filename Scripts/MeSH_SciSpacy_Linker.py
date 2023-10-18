# %%
### load scispacy and linker
import spacy
from spacy import displacy
import scispacy
from scispacy.linking import EntityLinker
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# models from https://allenai.github.io/scispacy/ 
nlp = spacy.load("en_core_sci_lg")
# nlp = spacy.load("en_core_web_lg")

kb_name = 'umls'    # 'mesh', 'umls'

config = {
    "resolve_abbreviations": True,  
    "linker_name": kb_name,
    "max_entities_per_mention": 2
}

nlp.add_pipe("scispacy_linker", config=config)
linker = nlp.get_pipe("scispacy_linker")


# %%
### function to extract from linked knowledge base

def kb_extractor(text):
    doc = nlp(text)
    for e in doc.ents:
        res = {}
        # level0: [ent1, ent2, ...], level1 (cui, score)
        if e._.kb_ents:
            res['text'] = e.text.lower()
            cui, score = e._.kb_ents[0]
            if cui in cui_set:
                continue
            cui_set.add(cui)
            res['cui'], res['score'] = cui, '%0.2f' % score
            # [1] is name, [2] is alias
            res['name'] = linker.kb.cui_to_entity[cui][1].lower()
            alias = linker.kb.cui_to_entity[cui][2]
            alias = [al.lower() for al in alias if (',' not in al) and ('(' not in al)]
            alias = list(Counter(alias).keys())
            res['aliases'] = ', '.join(sorted(alias))
        else:
            res['text'] = e.text.lower()
            cui, score = res['text'], -1.0
            if cui in cui_set:
                continue
            cui_set.add(cui)
            res['cui'], res['score'] = cui, score
            res['name'] = e.lemma_
            res['aliases'] = ''
        # print('\"%s\" %s (%0.2f)' % (e, cui, score))
        # print('Name:', res['name'])
        # print('Aliases:', res['aliases'])
        # print(res, '\n')            
        results.append(res)


# %%
################ get knowledge base entities ################

### test on one line or abstract
results = []
cui_set = set()
# text = "Title: Investigating the Effects of Exposure to Waterpipe Smoke on Pregnancy Outcomes Using an Animal Model . Introduction: In recent years , waterpipe tobacco smoking has been increasing in popularity all over the world . In this study , we explored effects of waterpipe smoking on pregnancy outcomes in rats . Methods: Animals were exposed to waterpipe tobacco smoking using a whole body exposure system 2 hours per day during pregnancy . A control group was exposed to fresh air only . Results: The results showed significant association between exposure to waterpipe smoke during pregnancy and low birth weight ( P < .01 ) and neonatal death ( P < .01 ) . In addition , the rate of growth of offspring of the waterpipe group was significantly lower than that of control group as measured by body weight gain during the first 3 months of life ( P < .001 ) . No effect was found for waterpipe smoking on mean number of progeny and male to female ratio among offspring . Conclusion: Waterpipe smoking is associated with adverse effects on pregnancy outcomes . Implications: In this study , we investigated for the first time the effect of waterpipe smoking on pregnancy outcomes using animal model . The results clearly showed that waterpipe smoking is associated with adverse effects on pregnancy outcomes that include low birth weigh , neonatal survival , and growth retardation ."
text = "e-cigarette is associated with smoking cessation"
text = 'Objective There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. This paper reviews the scientific literature on air quality, SHS exposure, health effects and economic outcomes related to SHS and smoking restrictions in casinos, as well as on smoking prevalence among casino patrons and problem gamblers. Data sources Peer reviewed studies published from January 1998 to March 2011. Data synthesis Evidence from air quality, biomarker and survey studies indicates that smoking in casinos is a significant public health problem. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS, as documented by elevated levels of SHS constituents in the air of casinos and by elevated levels of tobacco-specific biomarkers in non-smokers’ blood, urine and saliva. Partial smoking restrictions in casinos do not effectively protect non-smokers from SHS. Findings suggest that the smoking prevalence of casino patrons is comparable with that of the general public, although this prevalence may be higher among problem gamblers. Few studies have examined the economic impact of smoke-free policies in casinos, and the results of these studies are mixed. Conclusions Employees and patrons are exposed to SHS in casinos, posing a significant, preventable risk to their health. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. ","There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS. Partial smoking restrictions in casinos do not effectively protect nonsmokers from SHS. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. ",http://dx.doi.org/10.1136/tobaccocontrol-2013-051368,"There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS. Partial smoking restrictions in casinos do not effectively protect nonsmokers from SHS. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. '
kb_extractor(text)
print(len(results))
df = pd.DataFrame.from_dict(results, dtype=object)
df.iloc[0]['aliases']


# %%
### extract from all 300 test set
results = []
cui_set = set()

df_test = pd.read_csv('./TextAnnoSuite/documents.csv', encoding='utf_8_sig')
for idx, row in df_test.iterrows():
    if idx % 10 == 0: print(idx)
    text = row['M_doc_content']
    kb_extractor(text)
    
print(len(results))
df_KB = pd.DataFrame.from_dict(results, dtype=object)
if kb_name=='mesh':
    df_KB.to_csv('./MeSH/testset_by_mesh.csv', encoding='utf_8_sig', index=False)
else:
    df_KB.to_csv('./MeSH/testset_by_umls.csv', encoding='utf_8_sig', index=False)
df_KB.head()


# %%
############ extract from all papers ############
results = []
cui_set = set()

df_text = pd.read_csv('./NER/paperAbstractsAll.csv', encoding='utf8').fillna('')
for idx, row in df_text.iterrows():
    if idx % 10 == 0: print(idx)
    text = 'Title: ' + row['Title'] + '. ' + row['Abstract'] + ('Highlights: ' + row['Highlights'] if row['Highlights']!='' else '')
    kb_extractor(text)
    # print(len(results), len(cui_set))

print(len(results))
df_KB = pd.DataFrame.from_dict(results, dtype=object)
if kb_name=='mesh':
    df_KB.to_csv('./MeSH/testset_by_mesh.csv', encoding='utf8', index=False)
else:
    df_KB.to_csv('./MeSH/testset_by_umls.csv', encoding='utf8', index=False)
df_KB.head()


# %% 
### calculate some stats
n = len(df_KB)
m = len(df_KB[df_KB['score'].astype('float') > 0])
print('total # of terms %d; # of %s terms %d' % (n,kb_name,m))





# %%
################ display the NERs ###############
text = "Title: Investigating the Effects of Exposure to Waterpipe Smoke on Pregnancy Outcomes Using an Animal Model . Introduction: In recent years , waterpipe tobacco smoking has been increasing in popularity all over the world . In this study , we explored effects of waterpipe smoking on pregnancy outcomes in rats . Methods: Animals were exposed to waterpipe tobacco smoking using a whole body exposure system 2 hours per day during pregnancy . A control group was exposed to fresh air only . Results: The results showed significant association between exposure to waterpipe smoke during pregnancy and low birth weight ( P < .01 ) and neonatal death ( P < .01 ) . In addition , the rate of growth of offspring of the waterpipe group was significantly lower than that of control group as measured by body weight gain during the first 3 months of life ( P < .001 ) . No effect was found for waterpipe smoking on mean number of progeny and male to female ratio among offspring . Conclusion: Waterpipe smoking is associated with adverse effects on pregnancy outcomes . Implications: In this study , we investigated for the first time the effect of waterpipe smoking on pregnancy outcomes using animal model . The results clearly showed that waterpipe smoking is associated with adverse effects on pregnancy outcomes that include low birth weigh , neonatal survival , and growth retardation ."
# text = "e-cigarette is associated with smoking cessation"
text = 'Objective There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. This paper reviews the scientific literature on air quality, SHS exposure, health effects and economic outcomes related to SHS and smoking restrictions in casinos, as well as on smoking prevalence among casino patrons and problem gamblers. Data sources Peer reviewed studies published from January 1998 to March 2011. Data synthesis Evidence from air quality, biomarker and survey studies indicates that smoking in casinos is a significant public health problem. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS, as documented by elevated levels of SHS constituents in the air of casinos and by elevated levels of tobacco-specific biomarkers in non-smokers’ blood, urine and saliva. Partial smoking restrictions in casinos do not effectively protect non-smokers from SHS. Findings suggest that the smoking prevalence of casino patrons is comparable with that of the general public, although this prevalence may be higher among problem gamblers. Few studies have examined the economic impact of smoke-free policies in casinos, and the results of these studies are mixed. Conclusions Employees and patrons are exposed to SHS in casinos, posing a significant, preventable risk to their health. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. ","There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS. Partial smoking restrictions in casinos do not effectively protect nonsmokers from SHS. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. ",http://dx.doi.org/10.1136/tobaccocontrol-2013-051368,"There is no safe level of secondhand smoke (SHS) exposure. Most US casinos continue to allow smoking, thus exposing workers and patrons to the hazards of SHS. Workers and patrons in casinos that allow smoking are exposed to high levels of SHS. Partial smoking restrictions in casinos do not effectively protect nonsmokers from SHS. Policies completely prohibiting smoking in casinos would be expected to greatly reduce or eliminate SHS exposure in casinos, thereby protecting the health of casino workers and patrons. '
doc = nlp(text)
# en_core_sci_lg do not name entities - use specific _ner_ models:
# https://stackoverflow.com/questions/59265404/scispacy-for-biomedical-named-entitiy-recognitionner
displacy.render(doc, style="ent")
for e in doc.ents:
    print(e, e.label_)



# %% 
################# testing
alias = ['data, flow', 'data flow', 'water flow', 'flow, water', 'Data Flow']
alias = [al.lower() for al in alias if (',' not in al)]
alias = list(Counter(alias).keys())
alias
# %%
df_KB.head()
# %%
