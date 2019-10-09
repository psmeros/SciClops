import re
from math import sqrt
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from pandarallel import pandarallel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.autograd import Variable
from torch.nn.functional import gumbel_softmax, softmax
from torch.optim import SGD

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load("en") #TODO: change to en_core_web_lg 
hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
		return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())


def data_preprocessing(representation, partition='cancer', passage='prelude', use_cache=True):
	'''
	:param representation: bow, embeddings, bow_embeddings
	:param partition: article topic e.g., cancer
	:param passage: title, prelude, full_text (filter for paper text)
	'''
	if use_cache:
		cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col='url')
		claim_vec = pd.read_csv(sciclops_dir + 'cache/claim_vec_'+representation+'.tsv.bz2', sep='\t', index_col='url')
		papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t', index_col='url')

	else:
		pandarallel.initialize()
		representation_dim = 200
		

		articles = pd.read_csv(sciclops_dir + 'cache/'+partition+'_articles.tsv.bz2', sep='\t')
		claims = articles[['url', 'quotes']]

		claims.quotes = claims.quotes.apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
		claims = claims.explode('quotes').rename(columns={'quotes': 'claim'})
		
		papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t')
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
		claims.claim = claims.claim.astype(str)
		papers.title = papers.title.astype(str)
		papers.full_text = papers.full_text.astype(str)
		
		print('vectorizing...')
		if passage == 'title':
			papers_text = papers.title
		elif passage == 'prelude':
			papers_text = papers.title + ' ' + papers.full_text.apply(lambda w: w.split('\n')[0])
		elif passage == 'full_text':
			papers_text = papers.title + ' ' + papers.full_text

		claim_text = claims.claim

		if representation == 'embeddings':
			claim_vec = claim_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
		elif representation == 'bow':
			vectorizer = TfidfVectorizer(vocabulary=hn_vocabulary).fit(claim_text).fit(papers_text)
			claim_vec = vectorizer.transform(claim_text)
			papers_vec = vectorizer.transform(papers_text)
			PCA = TruncatedSVD(representation_dim).fit(claim_vec).fit(papers_vec)
			claim_vec = pd.DataFrame(PCA.transform(claim_vec), index=articles.index)
			papers_vec = pd.DataFrame(PCA.transform(papers_vec), index=papers.index)
		elif representation == 'bow_embeddings':
			claim_text = claim_text.apply(lambda t: ' '.join([w for w in hn_vocabulary if w in t]))
			papers_text = papers_text.apply(lambda t: ' '.join([w for w in hn_vocabulary if w in t]))
			claim_vec = claim_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)

		#caching    
		cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')
		claim_vec.to_csv(sciclops_dir + 'cache/claim_vec_'+representation+'.tsv.bz2', sep='\t')
		papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t')
	
	cooc = torch.Tensor(cooc.values.astype(float))
	claim_vec = torch.Tensor(claim_vec.values.astype(float))
	papers_vec = torch.Tensor(papers_vec.values.astype(float))
	
	return cooc, claim_vec, papers_vec


############################### ######### ###############################

cooc, claim_vec, papers_vec = data_preprocessing('bow_embeddings')

# Hyper Parameters
num_epochs = 5000
learning_rate = 1.e-4
weight_decay = 0.0

num_clusters = 2
linear_comb = .85

class ClusterNet(nn.Module):
	def __init__(self, num_clusters, L, embeddings_dim):
		super(ClusterNet, self).__init__()

		self.num_clusters = num_clusters
		self.L = L
		
		#self.L_prime = nn.Parameter(init.xavier_normal_(torch.Tensor(self.L.shape[0], self.L.shape[1])), requires_grad=True)

		self.claimsNet = nn.Sequential(
			nn.Linear(embeddings_dim, num_clusters),
			#nn.BatchNorm1d(num_clusters),
			nn.Softmax(dim=1)
        )
		self.papersNet = nn.Sequential(
			nn.Linear(embeddings_dim, num_clusters),
			#nn.BatchNorm1d(num_clusters),
			nn.Softmax(dim=1)
		)
		
	def forward(self, claims, papers):
		C = self.claimsNet(claims)
		P = self.papersNet(papers)
		#print(C)
		return C, P
	

	def loss(self, C, P, epoch):

		C_prime = self.L @ P

		#print(C_prime)
		loss = nn.MSELoss()
		# if epoch%2 == 0:
		# 	std = torch.sum((torch.std(C, dim=1)))
		# 	std = std if not torch.isnan(std) else 0
		# 	return loss(C, C_prime.data) + std
		# else:
		# 	std = torch.sum((torch.std(C_prime, dim=1)))
		# 	std = std if not torch.isnan(std) else 0
		# 	return loss(C_prime, C.data) + std

		#print(torch.max(C, 1))
		C_diff = C.shape[0] - torch.sum(torch.max(C, 1)[0] - torch.min(C, 1)[0])
		#C_prime_diff = C_prime.shape[0] -  torch.sum(torch.max(C_prime, 1)[0] - torch.min(C_prime, 1)[0])
		#std_C = std_C if not torch.isnan(std_C) else 0
		#std_C_prime = std_C_prime if not torch.isnan(std_C_prime) else 0
		return loss(C_prime, C) + C_diff #+ C_prime_diff


		#cluster_spread_loss = torch.sum(torch.sum(torch.tril(D, diagonal=-1), dim=0) + torch.sum(torch.triu(D, diagonal=1), dim=0))
		#print('cluster loss',cluster_spread_loss)
		#diag = torch.diagonal(D)
		#balance_loss = diag.max() - diag.min()
		#balance_loss = torch.sum(torch.abs((torch.diagonal(D, 0, dim1=-2, dim2=-1) - torch.sum(torch.diagonal(D, 0, dim1=-2, dim2=-1), dim=0).unsqueeze(dim=0).repeat(1, self.num_clusters)/self.num_clusters))**2)
		#print('balance loss',balance_loss)
		#print(cluster_spread_loss, balance_loss)
		#return linear_comb * cluster_spread_loss + (1-linear_comb) * balance_loss
		

#Model training
model = ClusterNet(num_clusters, cooc, claim_vec.shape[1])
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

for epoch in range(num_epochs):
	claim_vec = Variable(claim_vec)
	papers_vec = Variable(papers_vec)   
	C, P = model(claim_vec, papers_vec)
	loss = model.loss(C, P, epoch)
	if epoch%100 == 0:
		print(loss.data.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()




#claims = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')
#[u for u in pd.DataFrame(A[:,5].data.tolist()).nlargest(5, 0).join(claims)['url']]
