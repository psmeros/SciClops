import re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import torch.nn as nn
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.nn.functional import gumbel_softmax, softmax
from torch import optim

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load('en_core_web_lg')
hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()

NUM_CLUSTERS = 10
embeddings_dim = 50
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


def transform_papers(papers, passage):
	print('transforming papers...')
	
	if passage == 'title':
		papers_text = papers.title
	elif passage == 'prelude':
		papers_text = papers.title + ' ' + papers.full_text.apply(lambda w: w.split('\n')[0])
	elif passage == 'full_text':
		papers_text = papers.title + ' ' + papers.full_text
	
	papers_vec = papers_text.parallel_apply(lambda x: nlp(' '.join(clean_paper(x))).vector).apply(pd.Series).values
	papers_vec = TruncatedSVD(NUM_CLUSTERS).fit_transform(papers_vec)
	#papers_vec = GaussianMixture(NUM_CLUSTERS).fit(papers_vec).predict_proba(papers_vec)
	papers_vec = pd.DataFrame(papers_vec, index=papers.index)

	return papers_vec

def transform_claims(claims, clustering):
	print('transforming claims...')

	if clustering == 'GSDMM':
		mgp = MovieGroupProcess(K=NUM_CLUSTERS, alpha=0.01, beta=0.01, n_iters=50)
		claims['cluster'] = mgp.fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))
		claims_vec = np.zeros((len(claims), NUM_CLUSTERS))
		claims_vec[np.arange(len(claims)), claims.cluster.to_numpy()] = 1
	elif clustering == 'PCA-GMM':
		claims_vec = claims['clean_claim'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values
		claims_vec = TruncatedSVD(2).fit_transform(claims_vec)	
		claims_vec = GaussianMixture(NUM_CLUSTERS).fit(claims_vec).predict_proba(claims_vec)
	elif clustering == 'TSNE-GMM':
		claims_vec = claims['clean_claim'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values
		claims_vec = TSNE().fit_transform(claims_vec)	
		claims_vec = GaussianMixture(NUM_CLUSTERS).fit(claims_vec).predict_proba(claims_vec)

	claims_vec = pd.DataFrame(claims_vec, index=claims.index)
	return claims_vec


def data_preprocessing(clustering, passage='prelude', use_cache=True):
	'''
	:param representation: bow, embeddings, bow_embeddings
	:param passage: title, prelude, full_text (filter for paper text)
	'''
	if use_cache:
		cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim'])
		claims_vec = pd.read_csv(sciclops_dir + 'cache/claims_vec.tsv.bz2', sep='\t', index_col=['url', 'claim'])
		papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+'_'+passage+'.tsv.bz2', sep='\t', index_col='url')

	else:
		pandarallel.initialize()
		
		articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
		claims = articles[['url', 'quotes']].drop_duplicates(subset='url')

		claims.quotes = claims.quotes.parallel_apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
		claims = claims.explode('quotes').rename(columns={'quotes': 'claim'})
		claims = claims[~claims['claim'].isna()]

		#cleaning
		print('cleaning...')
		claims['clean_claim'] = claims['claim'].parallel_apply(clean_claim)
		claims = claims[claims['clean_claim'].str.len() != 0]

		G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
		G.remove_nodes_from(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())
		claims['refs'] = claims.url.parallel_apply(lambda u: set(G[u]))
		refs = [e for l in claims['refs'].to_list() for e in l]

		papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url')
		papers = papers[papers['url'].isin(refs)]

		claims = claims.set_index(['url', 'claim'])
		papers = papers.set_index('url')

		mlb = MultiLabelBinarizer()
		cooc = pd.DataFrame(mlb.fit_transform(claims.refs), columns=mlb.classes_, index=claims.index)

		papers_vec = transform_papers(papers, passage)
		claims_vec = transform_claims(claims, clustering)

		#caching    
		cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')
		claims_vec.to_csv(sciclops_dir + 'cache/claims_vec.tsv.bz2', sep='\t')
		papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_'+'_'+passage+'.tsv.bz2', sep='\t')
		
	return cooc, claims_vec, papers_vec


############################### ######### ###############################

cooc, claims_vec, papers_vec = data_preprocessing('PCA-GMM', use_cache=True)
papers_vec_index = papers_vec.index
cooc = torch.Tensor(cooc.values.astype(float))
claims_vec = torch.Tensor(claims_vec.values.astype(float))
papers_vec = torch.Tensor(papers_vec.values.astype(float))


# Hyper Parameters
num_epochs = 5000
learning_rate = 1.e-3
weight_decay = 0.0
hidden = 50
gamma = 1.e-5
batch_size = 2048

class ClusterNet(nn.Module):
	def __init__(self, shape):
		super(ClusterNet, self).__init__()
		
		self.P_prime = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(shape[0], shape[1])), requires_grad=True)
		
		self.papersNet = nn.Sequential(
			nn.Linear(NUM_CLUSTERS, hidden),
			nn.BatchNorm1d(hidden),
			nn.ReLU(),
			nn.Linear(hidden, NUM_CLUSTERS),
			nn.BatchNorm1d(NUM_CLUSTERS),
			nn.Softmax(dim=1)
			# nn.Linear(NUM_CLUSTERS, NUM_CLUSTERS),
			# nn.BatchNorm1d(NUM_CLUSTERS),
			# nn.ReLU()
		)
		
	def forward(self, P):
		return self.papersNet(P)	

	def loss(self, P, L, C):

		C_prime = L @ P

		#return nn.MSELoss()(C_prime, C) + gamma * torch.norm(C_prime, p='fro')
		return torch.norm(C - C_prime, p='fro') + gamma * torch.norm(P, p='fro')
		
#Model training
model = ClusterNet(papers_vec.shape)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

for epoch in range(num_epochs):
	p = np.random.permutation(len(papers_vec))

	mean_loss = []
	for i in range(0, len(p), batch_size):
		P = model.P_prime[p[i:i+batch_size]]
		L = cooc[:, p[i:i+batch_size]]
		C = claims_vec
		#P = Variable(P, requires_grad=True)
		L = Variable(L, requires_grad=False)   
		C = Variable(C, requires_grad=False)

		P = model(P)
		loss = model.loss(P, L, C)
		mean_loss.append(loss.data.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if epoch%5 == 0:
		print(sum(mean_loss)/len(mean_loss))

papers_vec = pd.DataFrame(model(papers_vec).detach().numpy(), index=papers_vec_index)
papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_learnt'+'.tsv.bz2', sep='\t')