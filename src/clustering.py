from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from torch import optim

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

NUM_CLUSTERS = 10
############################### ######### ###############################

################################ HELPERS ################################

def load_matrices(reduction_alg, reduction_dim):
	cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	claims_vec = pd.read_csv(sciclops_dir + 'cache/claims_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	papers_vec = pd.read_csv(sciclops_dir + 'cache/papers_vec_'+reduction_alg+'_'+str(reduction_dim)+'.tsv.bz2', sep='\t', index_col='url')
	return cooc, papers_vec, claims_vec

def transform_to_clusters(papers_vec, claims_vec, prior):
	gmm = GaussianMixture(NUM_CLUSTERS,weights_init=prior).fit(papers_vec).fit(claims_vec)
	papers_vec = gmm.predict_proba(papers_vec)
	claims_vec = gmm.predict_proba(claims_vec)
	
	return papers_vec, claims_vec

# Hyper Parameters
num_epochs = 10
learning_rate = 1.e-3
hidden = 50
batch_size = 512
gamma = 1.e-5

class ClusterNet(nn.Module):
	def __init__(self):
		super(ClusterNet, self).__init__()
		
		self.papersNet = nn.Sequential(
			nn.Linear(NUM_CLUSTERS, hidden),
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
	model = ClusterNet()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

	for epoch in range(num_epochs):
		p = np.random.permutation(len(papers_vec))

		mean_loss = []
		for i in range(0, len(p), batch_size):
			P = papers_vec[p[i:i+batch_size]]
			L = cooc[:, p[i:i+batch_size]]
			C = claims_vec

			optimizer.zero_grad()
			P = model(P)
			loss = model.loss(P, L, C)
			mean_loss.append(loss.detach().numpy())
			loss.backward()
			optimizer.step()

		if epoch%1 == 0:
			print(sum(mean_loss)/len(mean_loss))

	print('Reconstruction Error', torch.norm(cooc @ model(papers_vec) -  claims_vec, p='fro'))
	papers_vec = model(papers_vec).detach().numpy()
	return papers_vec




if __name__ == "__main__":

	cooc, papers_vec, claims_vec = load_matrices('PCA', 2)

	prior = [1/NUM_CLUSTERS for _ in range(NUM_CLUSTERS)]
	
	for _ in range(2):
		papers_clust, claims_clust = transform_to_clusters(papers_vec.values, claims_vec.values, prior)
		papers_clust = align_clusters(cooc.values, papers_clust, claims_clust)

		papers_clust = pd.DataFrame(papers_clust, index=papers_vec.index)
		claims_clust = pd.DataFrame(claims_clust, index=claims_vec.index)
		papers_clust.to_csv(sciclops_dir + 'cache/papers_vec_clusters.tsv.bz2', sep='\t')
		claims_clust.to_csv(sciclops_dir + 'cache/claims_vec_clusters.tsv.bz2', sep='\t')

		popularity = claims_clust.reset_index('popularity')['popularity']
		prior = [sum(claims_clust[i]*popularity) for i in range(NUM_CLUSTERS)]
		prior = [p/sum(prior) for p in prior]
