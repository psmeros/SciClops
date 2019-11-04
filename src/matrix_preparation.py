from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from pandarallel import pandarallel
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load('en_core_web_lg')
hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()

NUM_CLUSTERS = 10
CLAIM_THRESHOLD = 10
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())

#Remove stopwords/Lemmatize
def clean_claim(text):
	if (not text.endswith('.')) or ('\n' in text):
		text = []
	else:
		text = [str(w.lemma_) for w in nlp(text) if not (w.is_stop or len(w) == 1)]

		#remove small claims
		if len(text) < CLAIM_THRESHOLD:
			text = []
		else:
			text = [w for w in hn_vocabulary if w in text]
	return text

#Remove stopwords/Lemmatize
def clean_paper(text):
	text = [str(w.lemma_) for w in nlp(str(text)) if not (w.is_stop or len(w) == 1)]
	return text


def transform_to_vec(papers, claims, reduction_alg, reduction_dim=None, passage='prelude'):
	if passage == 'title':
		papers['passage'] = papers.title
	elif passage == 'prelude':
		papers['passage'] = papers.title + ' ' + papers.full_text.apply(lambda w: w.split('\n')[0])
	elif passage == 'full_text':
		papers['passage'] = papers.title + ' ' + papers.full_text
	
	
	print('transforming...')
	if reduction_alg == 'GSDMM':
		mgp = MovieGroupProcess(K=NUM_CLUSTERS, alpha=0.01, beta=0.01, n_iters=50)
		claims['cluster'] = mgp.fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))
		claims_vec = np.zeros((len(claims), NUM_CLUSTERS))
		claims_vec[np.arange(len(claims)), claims.cluster.to_numpy()] = 1

		mgp = MovieGroupProcess(K=NUM_CLUSTERS, alpha=0.01, beta=0.01, n_iters=50)
		papers['cluster'] = mgp.fit(papers['passage'], len(set([e for l in papers['passage'].tolist() for e in l])))
		papers_vec = np.zeros((len(papers), NUM_CLUSTERS))
		papers_vec[np.arange(len(papers)), papers.cluster.to_numpy()] = 1

	elif reduction_alg == 'LDA':
		#TODO
		pass

	elif reduction_alg =='PCA':
		papers_vec = papers['passage'].parallel_apply(lambda x: nlp(' '.join(clean_paper(x))).vector).apply(pd.Series).values
		claims_vec = claims['clean_claim'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values
		pca = TruncatedSVD(reduction_dim).fit(papers_vec).fit(claims_vec)
		papers_vec = pca.transform(papers_vec)
		claims_vec = pca.transform(claims_vec)

	elif reduction_alg =='T-SNE':
		papers_vec = papers['passage'].parallel_apply(lambda x: nlp(' '.join(clean_paper(x))).vector).apply(pd.Series).values
		claims_vec = claims['clean_claim'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values
		tsne = TSNE(reduction_dim).fit(papers_vec).fit(claims_vec)
		papers_vec = tsne.transform(papers_vec)
		claims_vec = tsne.transform(claims_vec)

	papers_vec = pd.DataFrame(papers_vec, index=papers.index)
	claims_vec = pd.DataFrame(claims_vec, index=claims.index)

	return papers_vec, claims_vec

def matrix_preparation(reduction_alg, reduction_dim=None, use_cache=True):
	if use_cache:
		cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim'])
		claims_vec = pd.read_csv(sciclops_dir + 'cache/claims_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
		papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t', index_col='url')

	else:
		pandarallel.initialize()
		
		articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
		claims = articles[['url', 'quotes']].drop_duplicates(subset='url')

		G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
		G.remove_nodes_from(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())
		
		papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url')
		refs = set(papers['url'].unique())
		claims['refs'] = claims.url.parallel_apply(lambda u: set(G.successors(u)).intersection(refs))
		refs = set([e for l in claims['refs'].to_list() for e in l])
		papers = papers[papers['url'].isin(refs)]

		tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
		claims['popularity'] = claims.url.parallel_apply(lambda u: sum([tweets.loc[t]['popularity'] for t in G.predecessors(u) if t in tweets.index]))

		claims.quotes = claims.quotes.parallel_apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
		claims = claims.explode('quotes').rename(columns={'quotes': 'claim'})
		claims = claims[~claims['claim'].isna()]

		#cleaning
		print('cleaning...')
		claims['clean_claim'] = claims['claim'].parallel_apply(clean_claim)
		claims = claims[claims['clean_claim'].str.len() != 0]

		claims = claims.set_index(['url', 'claim', 'popularity'])
		papers = papers.set_index('url')

		mlb = MultiLabelBinarizer()
		cooc = pd.DataFrame(mlb.fit_transform(claims.refs), columns=mlb.classes_, index=claims.index)

		papers_vec, claims_vec = transform_to_vec(papers, claims, reduction_alg, reduction_dim)
		
		#caching    
		cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')
		claims_vec.to_csv(sciclops_dir + 'cache/claims_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t')
		papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t')
		
	return cooc, papers_vec, claims_vec


############################### ######### ###############################


if __name__ == "__main__":
	matrix_preparation('PCA', 2, use_cache=True)
