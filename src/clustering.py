from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from torch import optim

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

NUM_CLUSTERS = 10
############################### ######### ###############################

################################ HELPERS ################################

def load_matrices(representation, dimension=None):
	cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	claims_vec = pd.read_csv(sciclops_dir + 'cache/claims_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col='url')
	return cooc, papers_vec, claims_vec

def transform_to_clusters(claims_vec, prior):
	gmm = GaussianMixture(NUM_CLUSTERS,weights_init=prior).fit(claims_vec)
	claims_vec = gmm.predict_proba(claims_vec)
	return claims_vec

# Hyper Parameters
num_epochs = 100
learning_rate = 1.e-3
hidden = 50
batch_size = 512
gamma = 1.e-3

class ClusterNet(nn.Module):
	def __init__(self, shape):
		super(ClusterNet, self).__init__()
		
		self.P_prime = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(shape[0], shape[1])), requires_grad=True)
		
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
		return torch.norm(C_prime - C, p='fro') - gamma * torch.norm(P, p='fro')
############################### ######### ###############################

def align_clusters(cooc, papers_vec, claims_vec):

	cooc, index = np.unique(cooc, axis=0, return_index=True)
	claims_vec = claims_vec[index]

	cooc = torch.Tensor(cooc.astype(float))
	papers_vec = torch.Tensor(papers_vec.astype(float))
	claims_vec = torch.Tensor(claims_vec.astype(float))
	
	#Model training
	model = ClusterNet(papers_vec.shape)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

	for epoch in range(num_epochs):
		p = np.random.permutation(len(papers_vec))

		mean_loss = []
		for i in range(0, len(p), batch_size):
			#P = papers_vec[p[i:i+batch_size]]
			P = model.P_prime[p[i:i+batch_size]]
			L = cooc[:, p[i:i+batch_size]]
			C = claims_vec

			optimizer.zero_grad()
			#P = model(P)
			loss = model.loss(P, L, C)
			mean_loss.append(loss.detach().numpy())
			loss.backward()
			optimizer.step()

		if epoch%1 == 0:
			print(sum(mean_loss)/len(mean_loss))

	papers_vec = model(papers_vec)
	print('Reconstruction Error', torch.norm(cooc @ model.P_prime -  claims_vec, p='fro'))
	papers_vec = papers_vec.detach().numpy()
	return papers_vec


def joint_clustering(method, dimension=None):

	if method == 'GMM':
		cooc, papers, claims = load_matrices(representation='embeddings', dimension=dimension)

		cooc = cooc.values
		papers = papers.values
		claims = claims.values
		
		gmm = GaussianMixture(NUM_CLUSTERS).fit(papers).fit(claims)
		claims = gmm.predict_proba(claims)
		papers = gmm.predict_proba(papers)
		
	elif method == 'LDA':
		cooc, papers, claims = load_matrices(representation='textual')
		cooc = cooc.values		
		papers = papers['clean_passage']
		claims = claims['clean_claim']
		
		CV = CountVectorizer().fit(papers).fit(claims)
	
		papers = CV.transform(papers)
		claims = CV.transform(claims)

		LDA = LatentDirichletAllocation(n_components=NUM_CLUSTERS, n_jobs=-1).fit(papers).fit(claims)

		papers = LDA.transform(papers)
		claims = LDA.transform(claims)

	elif method == 'GSDMM':
		cooc, papers, claims = load_matrices(representation='textual')

		mgp = MovieGroupProcess(K=NUM_CLUSTERS)
		claims['cluster'] = mgp.fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))
		papers['cluster'] = mgp.fit(papers['clean_passage'], len(set([e for l in papers['clean_passage'].tolist() for e in l])))

		claims_vec = np.zeros((len(claims), NUM_CLUSTERS))
		claims_vec[np.arange(len(claims)), claims.cluster.to_numpy()] = 1
		papers_vec = np.zeros((len(papers), NUM_CLUSTERS))
		papers_vec[np.arange(len(papers)), papers.cluster.to_numpy()] = 1

		claims = claims_vec
		papers = papers_vec
		cooc = cooc.values


	cooc = torch.Tensor(cooc.astype(float))
	papers = torch.Tensor(papers.astype(float))
	claims = torch.Tensor(claims.astype(float))

	print('Reconstruction Error', torch.norm(cooc @ papers - claims, p='fro'))
	print('Non-Uniformity Reguralizer', torch.norm(papers, p='fro') + torch.norm(claims, p='fro'))


def two_step_clustering():
	
	cooc, papers_vec, claims_vec = load_matrices(representation='embeddings', dimension=10)

	prior = [1/NUM_CLUSTERS for _ in range(NUM_CLUSTERS)]
	
	for _ in range(1):
		claims_clust = transform_to_clusters(claims_vec.values, prior)
		papers_clust = align_clusters(cooc.values, papers_vec.values, claims_clust)

		papers_clust = pd.DataFrame(papers_clust, index=papers_vec.index)
		claims_clust = pd.DataFrame(claims_clust, index=claims_vec.index)
		papers_clust.to_csv(sciclops_dir + 'cache/papers_vec_clusters.tsv.bz2', sep='\t')
		claims_clust.to_csv(sciclops_dir + 'cache/claims_vec_clusters.tsv.bz2', sep='\t')

		popularity = claims_clust.reset_index('popularity')['popularity']
		prior = [sum(claims_clust[i]*popularity) for i in range(NUM_CLUSTERS)]
		prior = [p/sum(prior) for p in prior]


if __name__ == "__main__":
	#joint_clustering(method='LDA')
	#joint_clustering(method='GSDMM')
	joint_clustering(method='GMM', dimension=10)
	#joint_clustering(method='GMM', dimension=100)
