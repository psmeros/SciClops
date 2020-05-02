import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import networkx as nx

############################### CONSTANTS ###############################

sciclops_dir = str(Path.home()) + '/data/sciclops/' 
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))
health = set(map(str.lower, open(sciclops_dir + 'small_files/hn_vocabulary/health.txt').read().splitlines()))


nlp = spacy.load('en_core_web_lg')
for word in STOP_WORDS:
    for w in (word, word[0].capitalize(), word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True

NUM_CLUSTERS = 100 
LAMBDA = 0.6
############################### ######### ###############################

################################ HELPERS ################################

def sts(text_1, text_2):
	#semantic = text_1.similarity(text_2)
	
	#text_1 = set(text_1.text.split()).intersection(hn_vocabulary).update(set([e.text for e in nlp(text_1).ents]))
	#text_2 = set(text_2.text.split()).intersection(hn_vocabulary).update(set([e.text for e in nlp(text_2).ents]))
	jaccard = len(text_1.intersection(text_2)) / (len(text_1.union(text_2)) or 1)
	return jaccard
	#return np.mean([semantic,jaccard])


def clean_text(text):
	return set([token.text.lower() for token in nlp(text) if not (token.is_punct | token.is_space | token.is_stop)])

############################### ######### ###############################

claimsKG = pd.read_csv(sciclops_dir+'small_files/claimKG/claims.csv')
claims_clusters = pd.read_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
papers_clusters = pd.read_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t').merge(pd.read_csv(sciclops_dir + 'cache/paper_details_v1.tsv.bz2', sep='\t')[['url', 'full_text']], on='url')


claims_clusters['cluster'] = claims_clusters[[str(i) for i in range(NUM_CLUSTERS)]].idxmax(axis=1)
claims_clusters = claims_clusters[['url', 'claim', 'popularity', 'cluster']]

claims_clusters['domain'] = claims_clusters.url.apply(lambda u: re.sub(r'^(http(s)?://)?(www\.)?', r'', urlsplit(u).netloc))
claims_clusters = claims_clusters.merge(pd.read_csv(sciclops_dir + 'small_files/news_outlets/acsh.tsv', sep='\t'), left_on='domain', right_on='outlet', how='left').drop(['domain', 'outlet'], axis=1).fillna(0.0)

claims_clusters['rate'] = 1 - ((claims_clusters['rate'] - min(claims_clusters['rate'])) / (max(claims_clusters['rate']) - min(claims_clusters['rate'])))
claims_clusters['popularity'] = claims_clusters['popularity']/max(claims_clusters['popularity'])

claims_clusters['weight'] = LAMBDA * claims_clusters['popularity'] + (1-LAMBDA) * claims_clusters['rate']

for i in range(NUM_CLUSTERS):
	claims = claims_clusters[claims_clusters['cluster'] == str(i)]
	G = nx.Graph()
	claims.apply(lambda c: (lambda s, w: [G.add_edge(e1, e2, weight=w) for e1 in s for e2 in s if e1 in health and e2 not in health])(set(c.claim.split()).intersection(hn_vocabulary), c.weight), axis=1)
	if not nx.is_empty(G):
		(lambda d: print(max(d, key=d.get)))(nx.edge_betweenness_centrality(G, weight='weight'))


h = 'cancer'
s = 'marijuana'

[print(c) for c in claims_clusters[claims_clusters.claim.str.contains(h) & claims_clusters.claim.str.contains(s)]['claim']]
[print(p) for p in papers_clusters[papers_clusters.full_text.str.contains(h) & papers_clusters.full_text.str.contains(s)]['title']]
[print(k) for k in claimsKG[claimsKG.claimText.str.contains(h) & claimsKG.claimText.str.contains(s)]['claimText']]


#Query for https://data.gesis.org/claimskg/sparql

# PREFIX schema:<http://schema.org/>
# PREFIX nif:<http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
# PREFIX :<http://data.gesis.org/claimskg/organization/>

# SELECT ?claimText ?claimKeywords ?rating
# (GROUP_CONCAT(DISTINCT ?claimEntity;separator=",") as ?claimEntities) 

# WHERE {
# ?claim a schema:CreativeWork.
# ?claim schema:text ?claimText.
# ?claim schema:keywords ?claimKeywords.
# ?claim schema:mentions/nif:isString ?claimEntity.

# ?claimReview schema:itemReviewed ?claim.
# ?claimReview schema:reviewRating ?reviewRating.
# ?reviewRating schema:author :claimskg.
# ?reviewRating schema:alternateName ?rating.

# FILTER( ?rating = "FALSE"@en || ?rating = "TRUE"@en) 

# } GROUP BY ?claimText ?claimKeywords ?rating

# offset 10000 limit 10000
