import re
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import spacy
from sklearn.metrics import mean_squared_error

############################### CONSTANTS ###############################

sciclops_dir = str(Path.home()) + '/data/sciclops/' 
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))
health = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/health.txt').read().splitlines()))

NUM_CLUSTERS = 10
LAMBDA = 0.3
############################### ######### ###############################


def worktime():
	print(pd.read_csv(sciclops_dir+'etc/evaluation/enhanced_context.csv').WorkTimeInSeconds.median())
	print(pd.read_csv(sciclops_dir+'etc/evaluation/original_context.csv').WorkTimeInSeconds.median())
	print(pd.read_csv(sciclops_dir+'etc/evaluation/no_context.csv').WorkTimeInSeconds.median())

worktime()

def RMSE():
	df_enhanced = pd.read_csv(sciclops_dir+'etc/evaluation/enhanced_context.csv')
	df_enhanced['claim'] = df_enhanced['Input.main_claim']
	df_enhanced ['validity'] = 0 * df_enhanced['Answer.ValidityNA.ValidityNA'] + (-2) * df_enhanced['Answer.Validity-2.Validity-2'] + (-1) * df_enhanced['Answer.Validity-1.Validity-1'] + 0 * df_enhanced['Answer.Validity0.Validity0'] + 1 * df_enhanced['Answer.Validity+1.Validity+1'] + 2 * df_enhanced['Answer.Validity+2.Validity+2']
	df_enhanced = df_enhanced[['claim', 'validity']]
	df_enhanced = df_enhanced.groupby('claim').mean().reset_index()

	df_original = pd.read_csv(sciclops_dir+'etc/evaluation/original_context.csv')
	df_original['claim'] = df_original['Input.main_claim']
	df_original ['validity'] = 0 * df_original['Answer.ValidityNA.ValidityNA'] + (-2) * df_original['Answer.Validity-2.Validity-2'] + (-1) * df_original['Answer.Validity-1.Validity-1'] + 0 * df_original['Answer.Validity0.Validity0'] + 1 * df_original['Answer.Validity+1.Validity+1'] + 2 * df_original['Answer.Validity+2.Validity+2']
	df_original = df_original[['claim', 'validity']]
	df_original = df_original.groupby('claim').mean().reset_index()

	df_no = pd.read_csv(sciclops_dir+'etc/evaluation/no_context.csv')
	df_no['claim'] = df_no['Input.main_claim']
	df_no ['validity'] = 0 * df_no['Answer.ValidityNA.ValidityNA'] + (-2) * df_no['Answer.Validity-2.Validity-2'] + (-1) * df_no['Answer.Validity-1.Validity-1'] + 0 * df_no['Answer.Validity0.Validity0'] + 1 * df_no['Answer.Validity+1.Validity+1'] + 2 * df_no['Answer.Validity+2.Validity+2']
	df_no = df_no[['claim', 'validity']]
	df_no = df_no.groupby('claim').mean().reset_index()

	df_sylvia = pd.read_csv(sciclops_dir+'etc/evaluation/sylvia.csv')
	df_sylvia['claim'] = df_sylvia['Scientific Claim']
	df_sylvia['validity'] = df_sylvia['Validity [-2,+2]']
	df_sylvia = df_sylvia[['claim', 'validity']]

	df_dimitra = pd.read_csv(sciclops_dir+'etc/evaluation/dimitra.csv')
	df_dimitra['claim'] = df_dimitra['Scientific Claim']
	df_dimitra['validity'] = df_dimitra['Validity [-2,+2]']
	df_dimitra = df_dimitra[['claim', 'validity']]

	df_experts = df_dimitra.merge(df_sylvia, on='claim')
	df_experts.validity_x = df_experts.validity_x.fillna(df_experts.validity_y)
	print(mean_squared_error(df_experts.validity_x, df_experts.validity_y, squared=False))
	df_experts['validity'] = df_experts[['validity_x', 'validity_y']].mean(axis=1)
	df_experts = df_experts[['claim', 'validity']]

	df = df_experts.merge(df_no, on='claim')
	print(mean_squared_error(df.validity_x, df.validity_y, squared=False))

	df = df_experts.merge(df_original, on='claim')
	print(mean_squared_error(df.validity_x, df.validity_y, squared=False))

	df = df_experts.merge(df_enhanced, on='claim')
	print(mean_squared_error(df.validity_x, df.validity_y, squared=False))

def KDEs():
	df_enhanced = pd.read_csv(sciclops_dir+'etc/evaluation/enhanced_context.csv')
	df_enhanced['claim'] = df_enhanced['Input.main_claim']
	df_enhanced['confidence'] = 1 * df_enhanced['Answer.Confidence1.Confidence1'] + 2 * df_enhanced['Answer.Confidence2.Confidence2'] + 3 * df_enhanced['Answer.Confidence3.Confidence3'] + 4 * df_enhanced['Answer.Confidence4.Confidence4'] + 5 * df_enhanced['Answer.Confidence5.Confidence5']
	df_enhanced['effort'] = 0 * df_enhanced['Answer.Effort0.Effort0'] + 1 * df_enhanced['Answer.Effort1.Effort1'] + 2 * df_enhanced['Answer.Effort2.Effort2'] + 3 * df_enhanced['Answer.Effort3.Effort3'] + 4 * df_enhanced['Answer.Effort4.Effort4'] + 5 * df_enhanced['Answer.Effort5.Effort5']
	df_enhanced ['validity'] = 0 * df_enhanced['Answer.ValidityNA.ValidityNA'] + (-2) * df_enhanced['Answer.Validity-2.Validity-2'] + (-1) * df_enhanced['Answer.Validity-1.Validity-1'] + 0 * df_enhanced['Answer.Validity0.Validity0'] + 1 * df_enhanced['Answer.Validity+1.Validity+1'] + 2 * df_enhanced['Answer.Validity+2.Validity+2']
	df_enhanced = df_enhanced[['claim', 'confidence', 'effort', 'validity']]

	df_original = pd.read_csv(sciclops_dir+'etc/evaluation/original_context.csv')
	df_original['claim'] = df_original['Input.main_claim']
	df_original['confidence'] = 1 * df_original['Answer.Confidence1.Confidence1'] + 2 * df_original['Answer.Confidence2.Confidence2']+ 3 * df_original['Answer.Confidence3.Confidence3']
	df_original['effort'] = 0 * df_original['Answer.Effort0.Effort0'] + 1 * df_original['Answer.Effort1.Effort1'] + 2 * df_original['Answer.Effort2.Effort2'] + 3 * df_original['Answer.Effort3.Effort3'] 
	df_original ['validity'] = 0 * df_original['Answer.ValidityNA.ValidityNA'] + (-2) * df_original['Answer.Validity-2.Validity-2'] + (-1) * df_original['Answer.Validity-1.Validity-1'] + 0 * df_original['Answer.Validity0.Validity0'] + 1 * df_original['Answer.Validity+1.Validity+1'] + 2 * df_original['Answer.Validity+2.Validity+2']
	df_original = df_original[['claim', 'confidence', 'effort', 'validity']]

	df_no = pd.read_csv(sciclops_dir+'etc/evaluation/no_context.csv')
	df_no['claim'] = df_no['Input.main_claim']
	df_no['confidence'] = 1 * df_no['Answer.Confidence1.Confidence1'] + 2 * df_no['Answer.Confidence2.Confidence2']+ 3 * df_no['Answer.Confidence3.Confidence3']
	df_no['effort'] = 0 * df_no['Answer.Effort0.Effort0'] + 1 * df_no['Answer.Effort1.Effort1'] + 2 * df_no['Answer.Effort2.Effort2'] + 3 * df_no['Answer.Effort3.Effort3'] 
	df_no ['validity'] = 0 * df_no['Answer.ValidityNA.ValidityNA'] + (-2) * df_no['Answer.Validity-2.Validity-2'] + (-1) * df_no['Answer.Validity-1.Validity-1'] + 0 * df_no['Answer.Validity0.Validity0'] + 1 * df_no['Answer.Validity+1.Validity+1'] + 2 * df_no['Answer.Validity+2.Validity+2']
	df_no = df_no[['claim', 'confidence', 'effort', 'validity']]

	sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
	sns.set_palette('colorblind')
	fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(25,10))
	for ind, ax in zip(['confidence', 'effort'], [ax0, ax1]):
		for df, l, c in zip([df_no, df_original, df_enhanced], ['Without', 'With Original', 'With Enhanced'], ['#2DA8D8FF', '#2A2B2DFF', '#D9514EFF']):
			#df = df.sort_values(by='confidence')[:int(.3*len(df))]
			ax = sns.kdeplot(df.groupby('claim').mean()[ind], label=l+' Context', color=c, shade= True, ax=ax)
			ax.set(ylim=(0, .9))	
			ax.set_xlabel(ind.capitalize(), fontsize='xx-large')
			ax.get_legend().remove()
			ax.set_xticks([0,2,4])
			ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize='x-large')

	ax0.set_ylabel('Density', fontsize='xx-large')	
	ax0.tick_params(axis='y', which='major', labelsize='x-large')
	ax1.get_yaxis().set_visible(False)

	lines, labels = ax1.get_legend_handles_labels()    
	legend = fig.legend(lines, labels, loc = 'upper right', ncol=1, bbox_to_anchor=(.83, .89), frameon=False, fontsize='xx-large')
	
	for handle in legend.legendHandles:
		handle.set_linewidth('7.0')	

	sns.despine(left=True, bottom=True)
	plt.show()
	fig.savefig(sciclops_dir+'etc/evaluation/KDEs.pdf', bbox_inches='tight')

def prepare_claims():
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
	claims_enhanced_context.to_csv(sciclops_dir + 'etc/evaluation/claims_enhanced_context_v1.csv', index=False)

def microtask_preparation():
	max_related = 3
	nlp = spacy.load('en_core_web_lg')

	df = pd.read_csv(sciclops_dir + 'etc/evaluation/claims_enhanced_context_v1.csv')

	def find_most_similar(claim, related, pos):
		related = eval(related)
		if len(related) <= pos:
			return ('', '')
		d = {(r[0],r[1]):nlp(claim).similarity(nlp(r[0])) for r in related}
		return sorted(d.items(), key=lambda x:x[1], reverse = True)[pos][0]

	df['topic'] = df['0']
	df['main_claim'] = df['1']
	df['main_claim_URL'] = df['2']

	for i in range(max_related):
		df['related_claim_'+str(i+1)], df['related_claim_'+str(i+1)+'_URL'] = zip(*df.apply(lambda x: find_most_similar(x['1'], x['3'], i), axis=1))

	for i in range(max_related):
		df['related_paper_'+str(i+1)], df['related_paper_'+str(i+1)+'_URL'] = zip(*df.apply(lambda x: find_most_similar(x['1'], x['4'], i), axis=1))

	for i in range(max_related):
		df['related_factcheck_'+str(i+1)], df['related_factcheck_'+str(i+1)+'_LABEL'] = zip(*df.apply(lambda x: find_most_similar(x['1'], x['5'], i), axis=1))


	df = df.drop([str(i) for i in range(6)], axis=1)

	df.to_csv(sciclops_dir + 'etc/evaluation/claims_enhanced_context_v2.csv', index=False)


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
