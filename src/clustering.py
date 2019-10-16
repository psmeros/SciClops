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
from torch.autograd import Variable
from torch.nn.functional import gumbel_softmax, softmax
from torch.optim import SGD

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load('en_core_web_lg')
hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()

num_clusters = 20
embeddings_dim = 300
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())

#Remove stopwords/Lemmatize
def nlp_clean(text):
	return ' '.join([str(w.lemma_) for w in nlp(text) if not (w.is_stop or len(w) == 1)])


def keywords_relation(clean_claims, partition):
	
	clean_claims = clean_claims.apply(lambda c: [w for w in hn_vocabulary if w in c])
	clean_claims = clean_claims[clean_claims.apply(lambda c: len(c) > 0 and c!= [partition])]
	
	clean_claims[clean_claims.apply(lambda c: len(c)==1)] = clean_claims[clean_claims.apply(lambda c: len(c)==1)].apply(lambda c: c + [partition])
	clean_claims = clean_claims.apply(lambda c: [[c[k1], c[k2]] for k1 in range(len(c)) for k2 in range(k1+1,len(c))])
	clean_claims = pd.DataFrame(clean_claims).rename(columns={'clean_claim': 'pairs'})
	clean_claims['weight'] = 1/clean_claims['pairs'].apply(len) 
	clean_claims = clean_claims.explode('pairs')
	clean_claims['k1'], clean_claims['k2'] = zip(*clean_claims['pairs'].apply(lambda p: (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])))
	clean_claims = clean_claims.drop('pairs', axis=1)
	clean_claims = clean_claims.groupby(['k1', 'k2']).sum().reset_index().sort_values('weight', ascending=False)[:15]
	clean_claims['k1'], clean_claims['k2'] = zip(*clean_claims.apply(lambda c: (c['k1'],c['k2']) if c['k1'] == partition else (c['k2'],c['k1']),axis=1))
	clean_claims = clean_claims[['k2', 'weight']].set_index('k2')
	
	sns.set(context='paper', style='white', color_codes=True)
	plt.figure(figsize=(2,5))
	colorticks=[100, 200, 300, 400, 600, 1000]
	ax = sns.heatmap(clean_claims, norm=LogNorm(clean_claims.min(), clean_claims.max()), cbar_kws={'ticks':colorticks, 'format':ScalarFormatter()}, vmin = 100, vmax=1000, cmap='copper')
	ax.tick_params(axis='x', labelbottom=False)
	ax.yaxis.label.set_visible(False)
	plt.savefig(sciclops_dir+'figures/'+partition+'.png', bbox_inches='tight', transparent=True)
	plt.show()
	

def data_preprocessing(representation, partition='cancer', passage='prelude', use_cache=True):
	'''
	:param representation: bow, embeddings, bow_embeddings
	:param partition: article topic e.g., cancer
	:param passage: title, prelude, full_text (filter for paper text)
	'''
	if use_cache:
		cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col='url')
		claim_vec = pd.read_csv(sciclops_dir + 'cache/claim_vec.tsv.bz2', sep='\t', index_col='url')
		papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t', index_col='url')

	else:
		pandarallel.initialize()
		
		articles = pd.read_csv(sciclops_dir + 'cache/'+partition+'_articles.tsv.bz2', sep='\t')
		claims = articles[['url', 'quotes']].drop_duplicates(subset='url')

		claims.quotes = claims.quotes.parallel_apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
		claims = claims.explode('quotes').rename(columns={'quotes': 'claim'})
		claims = claims[~claims['claim'].isna()]

		#nlp cleaning
		claims['clean_claim'] = claims['claim'].parallel_apply(lambda c: list(set([w for w in nlp_clean(c).split()])))
		#remove small claims
		claims = claims[claims['clean_claim'].parallel_apply(lambda c: len(c) >= 5)]

		#plot relation
		#keywords_relation(claims['clean_claim'], partition)

		#GSDMM model
		mgp = MovieGroupProcess(K=num_clusters, alpha=0.01, beta=0.01, n_iters=50)
		claims['cluster'] = mgp.fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))

		papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url')
		G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
		claims['refs'] = claims.url.parallel_apply(lambda u: set(G[u]))
		claims = claims.set_index('url')
		papers = papers.set_index('url')

		#cleaning
		print('cleaning...')
		blacklist_refs  = set(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())
		claims['refs'] = claims.refs.parallel_apply(lambda r: (r - blacklist_refs).intersection(set(papers.index.to_list())))
		mlb = MultiLabelBinarizer()
		cooc = pd.DataFrame(mlb.fit_transform(claims.refs), columns=mlb.classes_, index=claims.index)
		papers = papers[papers.index.isin(list(cooc.columns))]
		papers.title = papers.title.astype(str)
		papers.full_text = papers.full_text.astype(str)
		
		print('vectorizing...')
		if passage == 'title':
			papers_text = papers.title
		elif passage == 'prelude':
			papers_text = papers.title + ' ' + papers.full_text.apply(lambda w: w.split('\n')[0])
		elif passage == 'full_text':
			papers_text = papers.title + ' ' + papers.full_text

		if representation == 'embeddings':
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
		elif representation == 'bow':
			vectorizer = TfidfVectorizer(vocabulary=hn_vocabulary).fit(papers_text)
			papers_vec = vectorizer.transform(papers_text)
			PCA = TruncatedSVD(embeddings_dim).fit(claim_vec).fit(papers_vec)
			papers_vec = pd.DataFrame(PCA.transform(papers_vec), index=papers.index)
		elif representation == 'bow_embeddings':
			papers_text = papers_text.apply(lambda t: ' '.join([w for w in hn_vocabulary if w in t]))
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)

		#clusters to one-hot
		claim_vec = np.zeros((len(claims), num_clusters))
		claim_vec[np.arange(len(claims)), claims.cluster.to_numpy()] = 1
		claim_vec = pd.DataFrame(claim_vec, index=claims.index)

		#caching    
		cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')
		claim_vec.to_csv(sciclops_dir + 'cache/claim_vec.tsv.bz2', sep='\t')
		papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t')
	
	cooc = torch.Tensor(cooc.values.astype(float))
	claim_vec = torch.Tensor(claim_vec.values.astype(float))
	papers_vec = torch.Tensor(papers_vec.values.astype(float))
	
	return cooc, claim_vec, papers_vec


############################### ######### ###############################

cooc, claim_vec, papers_vec = data_preprocessing('embeddings')


# Hyper Parameters
num_epochs = 5000
learning_rate = 1.e-3
weight_decay = 0.0
hidden = 150

class ClusterNet(nn.Module):
	def __init__(self, L, C):
		super(ClusterNet, self).__init__()

		self.L = L
		self.C = C
		
		#self.L_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(self.L.shape[0], self.L.shape[1])), requires_grad=True)
		self.papersNet = nn.Sequential(
			nn.Linear(embeddings_dim, hidden),
			nn.BatchNorm1d(hidden),
			nn.ReLU(),
			nn.Linear(hidden, num_clusters),
			nn.BatchNorm1d(num_clusters),
			nn.Softmax(dim=1)
		)
		
	def forward(self, P):
		return self.papersNet(P)	

	def loss(self, P):

		C_prime = self.L @ P

		#loss = nn.MSELoss()
		#print(torch.max(C, 1))		
		#P_diff = torch.mean((1.0 - torch.max(P, 1)[0] - torch.min(P, 1)[0])**2)

		return torch.norm(self.C - C_prime, p='fro')

		#return loss(C_prime, )  + P_diff

		#cluster_spread_loss = torch.sum(torch.sum(torch.tril(D, diagonal=-1), dim=0) + torch.sum(torch.triu(D, diagonal=1), dim=0))
		#print('cluster loss',cluster_spread_loss)
		#diag = torch.diagonal(D)
		#balance_loss = diag.max() - diag.min()
		#balance_loss = torch.sum(torch.abs((torch.diagonal(D, 0, dim1=-2, dim2=-1) - torch.sum(torch.diagonal(D, 0, dim1=-2, dim2=-1), dim=0).unsqueeze(dim=0).repeat(1, self.num_clusters)/self.num_clusters))**2)
		#print('balance loss',balance_loss)
		#print(cluster_spread_loss, balance_loss)
		#return linear_comb * cluster_spread_loss + (1-linear_comb) * balance_loss
		

#Model training
model = ClusterNet(cooc, claim_vec)
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

for epoch in range(num_epochs):
	papers_vec = Variable(papers_vec)   
	P = model(papers_vec)
	loss = model.loss(P)
	if epoch%10 == 0:
		print(loss.data.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

#claims = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')
#[u for u in pd.DataFrame(A[:,5].data.tolist()).nlargest(5, 0).join(claims)['url']]

