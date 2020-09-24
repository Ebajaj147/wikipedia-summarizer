import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
#%matplotlib inline


# import wikipedia sentences
candidate_sentences = pd.read_csv("sentences.csv")
candidate_sentences.shape


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]



entity_pairs = []

for i in tqdm(candidate_sentences["sentence"]):
  entity_pairs.append(get_entities(i))




def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object
  matcher = Matcher(nlp.vocab)

  #define the pattern
  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

  matcher.add("matching_1", None, pattern)

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]]

  return(span.text)




relations = [get_relation(i) for i in tqdm(candidate_sentences['sentence'])]



print("\n\n")

print(pd.Series(relations).value_counts()[:])

print("\n\n")

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})




print(kg_df[:20])

# create a directed-graph from a dataframe
G = nx.from_pandas_edgelist(kg_df, "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())

e = nx.get_edge_attributes(G, 'edge')


# print(edge_labels.keys())

def getList(dict):
    return dict.keys()


# Driver program

kez = getList(e)
ke = list(kez)

for i in range(0, len(ke)):
    ke[i] = ke[i][:2]

fdict = dict(zip(ke, list(e.values())))
#print("\n\n")

#print(fdict)
#print("\n\n")


#print(e)

print("\n\n")

plt.figure(figsize=(22,22))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels = fdict, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None, rotate=True)
plt.savefig('KG.png')
plt.savefig('KG.pdf')

plt.show()




##for particular relation:
G = nx.from_pandas_edgelist(kg_df[kg_df['edge'] == "is"], "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

e = nx.get_edge_attributes(G, 'edge')


# Driver program

kez = getList(e)
ke = list(kez)

for i in range(0, len(ke)):
    ke[i] = ke[i][:2]

fdict = dict(zip(ke, list(e.values())))
# print(fdict)

plt.figure(figsize=(22, 22))
pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
# edge_labels = nx.get_edge_attributes(G,'edge')
# edge_labels = {('ideally  we', 'march'): 'start', ('studio', 'sequel'): 'start', ('however  cracks', 'operation'): 'start'}
# print(type(edge_labels))
nx.draw_networkx_edge_labels(G, pos, edge_labels=fdict, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None, rotate=True)
#plt.show()