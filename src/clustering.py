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
from torch.nn import init
from torch.nn.functional import gumbel_softmax
from torch.optim import SGD

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load("en_core_sci_md")

############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
		return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())


def data_preprocessing(representation, passage, use_cache=True):
	'''
	:param representation: bow, embeddings, bow_embeddings
	:param passage: title, prelude, full_text
	'''
	if use_cache:
		cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col='url')
		articles_vec = pd.read_csv(sciclops_dir + 'cache/articles_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t', index_col='url')
		papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t', index_col='url')

	else:
		pandarallel.initialize()
		representation_dim = 200
		
		articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
		papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t')
		G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
		articles['refs'] = articles.url.parallel_apply(lambda u: set(G[u]))
		articles = articles.set_index('url')
		papers = papers.set_index('url')

		#cleaning
		print('cleaning...')
		blacklist_refs  = set(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())
		articles['refs'] = articles.refs.parallel_apply(lambda r: (r - blacklist_refs).intersection(set(papers.index.to_list())))
		mlb = MultiLabelBinarizer()
		cooc = pd.DataFrame(mlb.fit_transform(articles.refs), columns=mlb.classes_, index=articles.index)
		papers = papers[papers.index.isin(list(cooc.columns))]
		articles.title = articles.title.astype(str)
		articles.full_text = articles.full_text.astype(str)
		papers.title = papers.title.astype(str)
		papers.full_text = papers.full_text.astype(str)
		
		print('vectorizing...')
		if passage == 'title':
			articles_text = articles.title
			papers_text = papers.title
		elif passage == 'prelude':
			articles_text = articles.title + ' ' + articles.full_text.apply(lambda w: w.split('\n')[0])
			papers_text = papers.title + ' ' + papers.full_text.apply(lambda w: w.split('\n')[0])
		elif passage == 'full_text':
			articles_text = articles.title + ' ' + articles.full_text
			papers_text = papers.title + ' ' + papers.full_text
		
		if representation == 'embeddings':
			articles_vec = articles_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
		elif representation == 'bow':
			hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
			vectorizer = TfidfVectorizer(vocabulary=hn_vocabulary).fit(articles_text).fit(papers_text)
			articles_vec = vectorizer.transform(articles_text)
			papers_vec = vectorizer.transform(papers_text)
			PCA = TruncatedSVD(representation_dim).fit(articles_vec).fit(papers_vec)
			articles_vec = pd.DataFrame(PCA.transform(articles_vec), index=articles.index)
			papers_vec = pd.DataFrame(PCA.transform(papers_vec), index=papers.index)
		elif representation == 'bow_embeddings':
			hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
			#articles_text = articles_text.apply(lambda t: ' '.join([(w + ' ')*t.count(w) for w in hn_vocabulary]))
			#papers_text = papers_text.apply(lambda t: ' '.join([(w + ' ')*t.count(w) for w in hn_vocabulary]))
			articles_text = articles_text.apply(lambda t: ' '.join([w for w in hn_vocabulary if w in t]))
			papers_text = papers_text.apply(lambda t: ' '.join([w for w in hn_vocabulary if w in t]))
			articles_vec = articles_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)
			papers_vec = papers_text.parallel_apply(lambda x: nlp(x).vector).apply(pd.Series)

		#caching    
		cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')
		articles_vec.to_csv(sciclops_dir + 'cache/articles_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t')
		papers_vec.to_csv(sciclops_dir + 'cache/papers_vec_'+representation+'_'+passage+'.tsv.bz2', sep='\t')
	
	cooc = torch.Tensor(cooc.values.astype(float))
	articles_vec = torch.Tensor(articles_vec.values.astype(float))
	papers_vec = torch.Tensor(papers_vec.values.astype(float))
	
	return cooc, articles_vec, papers_vec


############################### ######### ###############################

cooc, articles_vec, papers_vec = data_preprocessing('bow_embeddings', 'prelude')

# Hyper Parameters
num_epochs = 100
learning_rate = 1.e-6
weight_decay = 0.0

num_clusters = 10
linear_comb = .85

class ClusterNet(nn.Module):
	def __init__(self, num_clusters, cooc, embeddings_dim):
		super(ClusterNet, self).__init__()

		self.num_clusters = num_clusters
		self.cooc = cooc

		self.articlesNet = nn.Sequential(
			nn.Linear(embeddings_dim, num_clusters),
			nn.BatchNorm1d(num_clusters),
			nn.Softmax(dim=1)
        )
		self.papersNet = nn.Sequential(
			nn.Linear(embeddings_dim, num_clusters),
			nn.BatchNorm1d(num_clusters),
			nn.Softmax(dim=1)
		)
		
	def forward(self, articles, papers):
		A = self.articlesNet(articles)
		P = self.papersNet(papers)
		return A, P
	

	def loss(self, A, P):
		D = A.t() @ self.cooc @ P

		#print(D)
		cluster_spread_loss = torch.sum(torch.sum(torch.tril(D, diagonal=-1), dim=0) + torch.sum(torch.triu(D, diagonal=1), dim=0))
		#print('cluster loss',cluster_spread_loss)
		#diag = torch.diagonal(D)
		#balance_loss = diag.max() - diag.min()

		balance_loss = torch.sum(torch.abs((torch.diagonal(D, 0, dim1=-2, dim2=-1) - torch.sum(torch.diagonal(D, 0, dim1=-2, dim2=-1), dim=0).unsqueeze(dim=0).repeat(1, self.num_clusters)/self.num_clusters))**2)


		#print('balance loss',balance_loss)
		#print(cluster_spread_loss, balance_loss)
		return linear_comb * cluster_spread_loss + (1-linear_comb) * balance_loss
		

#Model training
model = ClusterNet(num_clusters, cooc, articles_vec.shape[1])
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

for epoch in range(num_epochs):    
	optimizer.zero_grad()
	A, P = model(articles_vec, papers_vec)
	loss = model.loss(A, P)
	print(loss.data.item())
	loss.backward()
	optimizer.step()




#articles = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')
#[u for u in pd.DataFrame(A[:,5].data.tolist()).nlargest(5, 0).join(articles)['url']]
