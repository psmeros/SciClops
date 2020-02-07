from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from torch import optim

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

np.random.seed(42)
NUM_CLUSTERS = 10
############################### ######### ###############################

################################ HELPERS ################################

def load_matrices(representation, dimension=None):
	cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	claims = pd.read_csv(sciclops_dir + 'cache/claims_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	papers = pd.read_csv(sciclops_dir + 'cache/papers_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col='url')
	return cooc, papers, claims


# Hyper Parameters
num_epochs = 300
learning_rate = 1.e-3
hidden = 50
batch_size = 512
gamma = 1.e-3

class ClusterNet(nn.Module):
	def __init__(self, shape):
		super(ClusterNet, self).__init__()
		
		self.P_prime = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(shape[0], NUM_CLUSTERS)), requires_grad=True)
		
		self.papersNet = nn.Sequential(
			nn.Linear(shape[1], hidden),
			nn.BatchNorm1d(hidden),
			nn.ReLU(),
			nn.Linear(hidden, NUM_CLUSTERS),
			nn.BatchNorm1d(NUM_CLUSTERS),
			nn.Softmax(dim=1),
			# nn.Linear(NUM_CLUSTERS, hidden),
			# nn.Linear(hidden, NUM_CLUSTERS),
			# # #nn.BatchNorm1d(NUM_CLUSTERS),
			# nn.Softmax(dim=1),
		)
		
	def forward(self, P):
		return self.papersNet(P)	

	def loss(self, P, L, C):
		C_prime = L @ P
		return torch.norm(C_prime - C, p='fro') - gamma * (torch.norm(P, p='fro') + torch.norm(C, p='fro'))
############################### ######### ###############################

def eval_clusters(cooc, papers, claims, top_k):
	top_papers = np.unique((-papers).argsort(axis=0)[:top_k].flatten())
	
	P = papers[top_papers]
	L = cooc[:, top_papers]
	mask = (~np.all(L == 0, axis=1))
	L = L[mask]
	C = claims[mask]
	
	labels_inherited = np.multiply(L, np.argmax(P, axis=1)).max(axis=1)
	labels_expected = np.argmax(C, axis=1)

	v1 = v_measure_score(labels_expected, labels_inherited)

	top_claims = np.unique((-claims).argsort(axis=0)[:top_k].flatten()) 
	
	C = claims[top_claims]
	L = cooc[top_claims]
	mask = (~np.all(L == 0, axis=0))
	L = L[:, mask]
	P = papers[mask]
	
	labels_inherited = np.multiply(L.T, np.argmax(C, axis=1)).max(axis=1)
	labels_expected = np.argmax(P, axis=1)

	v2 = v_measure_score(labels_expected, labels_inherited)

	print('claims:', v1)
	print('papers:', v2)
	print('average:', (v1+v2)/2)

def disjoint_clustering(method, top_k=5, dimension=None):

	if method == 'GMM':
		cooc, papers, claims = load_matrices(representation='embeddings', dimension=dimension)

		cooc = cooc.values
		papers = papers.values
		claims = claims.values
		
		claims = GaussianMixture(NUM_CLUSTERS, covariance_type='spherical', tol=0.5, random_state=42).fit(claims).predict_proba(claims)
		papers = GaussianMixture(NUM_CLUSTERS, covariance_type='spherical', tol=0.5, random_state=42).fit(papers).predict_proba(papers)
		
	elif method == 'KMeans':
		cooc, papers, claims = load_matrices(representation='embeddings', dimension=dimension)

		cooc = cooc.values
		papers = papers.values
		claims = claims.values
		
		p_cluster = KMeans(NUM_CLUSTERS, random_state=42).fit(papers).predict(papers)
		c_cluster = KMeans(NUM_CLUSTERS, random_state=42).fit(claims).predict(claims)
		
		claims = np.zeros((len(claims), NUM_CLUSTERS))
		claims[np.arange(len(claims)), c_cluster] = 1
		papers = np.zeros((len(papers), NUM_CLUSTERS))
		papers[np.arange(len(papers)), p_cluster] = 1

	elif method == 'LDA':
		cooc, papers, claims = load_matrices(representation='textual')
		cooc = cooc.values		
		papers = papers['clean_passage']
		claims = claims['clean_claim']
		
		CV = CountVectorizer().fit(papers).fit(claims)
		papers = CV.transform(papers)
		claims = CV.transform(claims)

		papers = LatentDirichletAllocation(n_components=NUM_CLUSTERS, n_jobs=-1).fit(papers).transform(papers)
		claims = LatentDirichletAllocation(n_components=NUM_CLUSTERS, n_jobs=-1).fit(claims).transform(claims)

	elif method == 'GSDMM':
		cooc, papers, claims = load_matrices(representation='textual')
		cooc = cooc.values

		p_cluster = MovieGroupProcess(K=NUM_CLUSTERS).fit(papers['clean_passage'], len(set([e for l in papers['clean_passage'].tolist() for e in l])))
		c_cluster = MovieGroupProcess(K=NUM_CLUSTERS).fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))
		
		papers = np.zeros((len(papers), NUM_CLUSTERS))
		papers[np.arange(len(papers)), np.array(p_cluster)] = 1
		claims = np.zeros((len(claims), NUM_CLUSTERS))
		claims[np.arange(len(claims)), np.array(c_cluster)] = 1


	eval_clusters(cooc, papers, claims, top_k)

def align_clustering(prior, learn_transform, top_k):

	cooc, papers, claims = load_matrices(representation='embeddings')
	claims_index = claims.index
	papers_index = papers.index
	claims = claims.values
	papers = papers.values
	cooc = cooc.values

	claims = GaussianMixture(NUM_CLUSTERS, weights_init=prior, covariance_type='spherical', tol=0.5, random_state=42).fit(claims).predict_proba(claims)

	cooc_unique, index = np.unique(cooc, axis=0, return_index=True)
	claims_unique = claims[index]

	cooc_unique = torch.Tensor(cooc_unique.astype(float))
	papers = torch.Tensor(papers.astype(float))
	claims_unique = torch.Tensor(claims_unique.astype(float))
	
	#Model training
	model = ClusterNet(papers.shape)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

	for epoch in range(num_epochs):
		rand = np.random.permutation(len(papers))

		mean_loss = []
		for i in range(0, len(rand), batch_size):
			optimizer.zero_grad()
			
			L = cooc_unique[:, rand[i:i+batch_size]]
			C = claims_unique

			if learn_transform:
				P = papers[rand[i:i+batch_size]]
				P = model(P)
			else:
				P = model.P_prime[rand[i:i+batch_size]]
			
			loss = model.loss(P, L, C)
			mean_loss.append(loss.detach().numpy())
			loss.backward()
			optimizer.step()

		if epoch%1 == 0:
			print(sum(mean_loss)/len(mean_loss))

	if learn_transform:
		papers = model(papers).detach().numpy()
	else:
		papers = model.P_prime.detach().numpy()
	
	eval_clusters(cooc, papers, claims, top_k)

	papers = pd.DataFrame(papers, index=papers_index)
	claims = pd.DataFrame(claims, index=claims_index)

	return papers, claims


def popularity_clustering(learn_transform, iterations=1, top_k=5):
	
	prior = [1/NUM_CLUSTERS for _ in range(NUM_CLUSTERS)]
	
	for _ in range(iterations):
		papers, claims = align_clustering(prior, learn_transform, top_k)
		
		popularity = claims.reset_index('popularity')['popularity']
		prior = [sum(claims[i]*popularity) for i in range(NUM_CLUSTERS)]
		prior = [p/sum(prior) for p in prior]

	papers.to_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')
	claims.to_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')



if __name__ == "__main__":
	#disjoint_clustering(method='LDA')
	#disjoint_clustering(method='GSDMM')
	#disjoint_clustering(method='GMM')
	#disjoint_clustering(method='GMM', dimension=10)
	#disjoint_clustering(method='KMeans', dimension=10)
	#disjoint_clustering(method='KMeans')
	popularity_clustering(learn_transform=False, iterations=2)