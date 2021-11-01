import os
import random
import re
from pathlib import Path
from urllib.parse import urlsplit

import networkx as nx
import numpy as np
import pandas as pd
import requests
import spacy

############################### CONSTANTS ###############################
sciclops_dir = str(Path.home()) + '/data/sciclops/' 
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))
health = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/health.txt').read().splitlines()))

random.seed(42)

NUM_CLUSTERS = 10
LAMBDA = 0.3
############################### ######### ###############################

################################ HELPERS ################################

def claimbuster():
    def fact_check(claim):
        # Define the endpoint (url), payload (sentence to be scored), api-key (api-key is sent as an extra header)
        api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
        request_headers = {"x-api-key": os.getenv('CLAIMBUSTER_KEY')}
        payload = {"input_text": claim}
        # Send the POST request to the API and store the api response
        api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)
        # Print out the JSON payload the API sent back
        return api_response.json()
    
    df = pd.read_csv(str(Path.home()) + '/data/sciclops/etc/evaluation/raw_claims.csv')
    df['fact_check'] = df['Scientific Claim'].apply(fact_check)
    df['fact_check'] =  np.digitize(df['fact_check'].apply(lambda fc: fc['results'][0]['score']),[.2, .4, .6, .8, 1]) - 2
    df.to_csv(str(Path.home()) + '/data/sciclops/etc/evaluation/claimbuster.csv', index=False)

def ClaimsKG_query():
#Query for https://data.gesis.org/claimskg/sparql

	query = '''
	PREFIX schema:<http://schema.org/>
	PREFIX nif:<http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
	PREFIX :<http://data.gesis.org/claimskg/organization/>

	SELECT ?claimText ?claimKeywords ?rating
	(GROUP_CONCAT(DISTINCT ?claimEntity;separator=",") as ?claimEntities) 

	WHERE {
	?claim a schema:CreativeWork.
	?claim schema:text ?claimText.
	?claim schema:keywords ?claimKeywords.
	?claim schema:mentions/nif:isString ?claimEntity.

	?claimReview schema:itemReviewed ?claim.
	?claimReview schema:reviewRating ?reviewRating.
	?reviewRating schema:author :claimskg.
	?reviewRating schema:alternateName ?rating.

	FILTER( ?rating = "FALSE"@en || ?rating = "TRUE"@en) 

	} GROUP BY ?claimText ?claimKeywords ?rating

	offset 10000 limit 10000
	'''
	return query

############################### ######### ###############################

def enhance_context(max_related = 3):
	nlp = spacy.load('en_core_web_lg')

	claimsKG = pd.read_csv(sciclops_dir+'etc/claimKG/claims.csv')
	claims_clusters = pd.read_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
	papers_clusters = pd.read_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t').merge(pd.read_csv(sciclops_dir + 'cache/paper_details_v1.tsv.bz2', sep='\t')[['url', 'full_text']], on='url')


	claims_clusters['cluster'] = claims_clusters[[str(i) for i in range(NUM_CLUSTERS)]].idxmax(axis=1)
	claims_clusters = claims_clusters[['url', 'claim', 'popularity', 'cluster']]

	claims_clusters['domain'] = claims_clusters.url.apply(lambda u: re.sub(r'^(http(s)?://)?(www\.)?', r'', urlsplit(u).netloc))
	claims_clusters = claims_clusters.merge(pd.read_csv(sciclops_dir + 'etc/news_outlets/acsh.tsv', sep='\t'), left_on='domain', right_on='outlet', how='left').drop(['domain', 'outlet'], axis=1).fillna(0.0)

	claims_clusters['rate'] = 1 - ((claims_clusters['rate'] - min(claims_clusters['rate'])) / (max(claims_clusters['rate']) - min(claims_clusters['rate'])))
	claims_clusters['popularity'] = claims_clusters['popularity']/max(claims_clusters['popularity'])

	claims_clusters['weight'] = LAMBDA * claims_clusters['popularity'] + (1-LAMBDA) * claims_clusters['rate']

	pairs = []
	max_pairs_per_cluster = 5
	for i in range(NUM_CLUSTERS):
		claims = claims_clusters[claims_clusters['cluster'] == str(i)]
		G = nx.Graph()
		claims.apply(lambda c: (lambda s, w: [G.add_edge(e1, e2, weight=G.get_edge_data(e1, e2, default={'weight': 0})['weight'] + w) for e1 in s for e2 in s if e1 in health and e2 not in health])(set(c.claim.split()).intersection(hn_vocabulary), c.weight), axis=1)
		if not nx.is_empty(G):
			pairs += (lambda d: list(dict(sorted(d.items(), key=lambda x:x[1], reverse = True)[:max_pairs_per_cluster]).keys()))(nx.edge_betweenness_centrality(G, weight='weight'))

	claims_enhanced_context = []
	for p in pairs:
		claims = claims_clusters[claims_clusters.claim.str.contains(p[0]) & claims_clusters.claim.str.contains(p[1])][['claim', 'url']].values.tolist()
		papers = papers_clusters[papers_clusters.full_text.str.contains(p[0]) & papers_clusters.full_text.str.contains(p[1])][['title', 'url']].values.tolist()
		if not papers:
			continue
		kg = claimsKG[claimsKG.claimText.str.contains(p[0]) | claimsKG.claimText.str.contains(p[1])][['claimText', 'rating']].values.tolist()
		
		for c in claims:
			claims_enhanced_context += [[p[0]+'-'+p[1], c[0], c[1], claims, papers, kg]]
			claims_enhanced_context += [[p[1]+'-'+p[0], c[0], c[1], claims, papers, kg]]

	claims_enhanced_context = pd.DataFrame(claims_enhanced_context)

	claims_enhanced_context = claims_enhanced_context.drop_duplicates(subset=1)

	def find_most_similar(claim, related, pos):
		related = eval(related)
		if len(related) <= pos:
			return ('', '')
		d = {(r[0],r[1]):nlp(claim).similarity(nlp(r[0])) for r in related}
		return sorted(d.items(), key=lambda x:x[1], reverse = True)[pos][0]

	claims_enhanced_context['topic'] = claims_enhanced_context['0']
	claims_enhanced_context['main_claim'] = claims_enhanced_context['1']
	claims_enhanced_context['main_claim_URL'] = claims_enhanced_context['2']

	for i in range(max_related):
		claims_enhanced_context['related_claim_'+str(i+1)], claims_enhanced_context['related_claim_'+str(i+1)+'_URL'] = zip(*claims_enhanced_context.apply(lambda x: find_most_similar(x['1'], x['3'], i), axis=1))

	for i in range(max_related):
		claims_enhanced_context['related_paper_'+str(i+1)], claims_enhanced_context['related_paper_'+str(i+1)+'_URL'] = zip(*claims_enhanced_context.apply(lambda x: find_most_similar(x['1'], x['4'], i), axis=1))

	for i in range(max_related):
		claims_enhanced_context['related_factcheck_'+str(i+1)], claims_enhanced_context['related_factcheck_'+str(i+1)+'_LABEL'] = zip(*claims_enhanced_context.apply(lambda x: find_most_similar(x['1'], x['5'], i), axis=1))


	claims_enhanced_context = claims_enhanced_context.drop([str(i) for i in range(6)], axis=1)

	claims_enhanced_context.to_csv(sciclops_dir + 'evaluation/claims_enhanced_context.csv', index=False)


if __name__ == "__main__":
	enhance_context()