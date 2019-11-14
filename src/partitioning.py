import string
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pke
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from nltk.corpus import stopwords
from pandarallel import pandarallel

from matrix_preparation import clean_claim, read_graph

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
############################### ######### ###############################

################################ HELPERS ################################
def keyphrase_extraction(text, n=10):	
	extractor = pke.unsupervised.MultipartiteRank()
	extractor.load_document(input=text)

	pos = {'NOUN', 'PROPN', 'ADJ'}
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += stopwords.words('english')
	extractor.candidate_selection(pos=pos, stoplist=stoplist)
	extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')

	keyphrases = extractor.get_n_best(n)
	return keyphrases

def extract_keyphrases():
	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	articles['keyphrases'] = articles.apply(lambda a: keyphrase_extraction(str(a.title) + ' ' + str(a.full_text)), axis=1)

	ght = [l.strip() for e in list(pd.read_csv(sciclops_dir + 'small_files/global_health_threats/ght.tsv', sep='\t').Keywords) for l in e.split('|')]
	articles['partition'] = articles.keyphrases.apply(lambda l: set([kph for sl in [k.split() for (k, _) in l] for kph in sl if kph in ght]))
	articles.to_csv(scilens_dir + 'article_details_v4.tsv.bz2', sep='\t', index=False)

############################### ######### ###############################

def keyphrase_relations(keyphrase, blacklist):
	pandarallel.initialize()
	
	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	claims = articles[['url', 'quotes']].drop_duplicates(subset='url')

	G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
	G.remove_nodes_from(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())

	claims.quotes = claims.quotes.parallel_apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
	claims = claims.explode('quotes').rename(columns={'quotes': 'claim'})
	claims = claims[~claims['claim'].isna()]

	claims = claims[claims['claim'].str.contains(keyphrase)]

	claims['clean_claim'] = claims['claim'].parallel_apply(clean_claim)
	claims = claims[claims['clean_claim'].str.len() != 0]

	tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
	claims['popularity'] = claims.url.parallel_apply(lambda u: sum([tweets.loc[t]['popularity'] for t in G.predecessors(u) if t in tweets.index]))

	claims['popularity'] = claims['popularity'] * 1/claims['clean_claim'].apply(len)

	claims = claims[['clean_claim', 'popularity']]
	claims = claims.explode('clean_claim')
	claims = claims[~claims['clean_claim'].isin(blacklist)]

	claims = claims.groupby(['clean_claim']).sum().reset_index().sort_values('popularity', ascending=False)[:15]

	claims = claims.rename(columns={'clean_claim':'health and nutrition terms'})
	claims = claims.set_index('health and nutrition terms')

	sns.set(context='paper', style='white', color_codes=True)
	plt.figure(figsize=(2,5))
	#colorticks=[1, 50, 100]
	#ax = sns.heatmap(claims, norm=LogNorm(claims['popularity'].min(), claims['popularity'].max()), cbar_kws={'ticks':colorticks, 'format':ScalarFormatter()}, vmin = 1, vmax=250, cmap='copper')
	ax = sns.heatmap(claims, cmap='copper')
	# ax.tick_params(axis='x', labelbottom=False)
	# ax.yaxis.label.set_visible(False)
	plt.savefig(sciclops_dir+'figures/'+keyphrase+'.png', bbox_inches='tight', transparent=True)
	plt.show()
	

if __name__ == "__main__":
	keyphrase_relations('meat', ['meat', 'beef', 'pork', 'chicken', 'TMAO', 'lamb'])
