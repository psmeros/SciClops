from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn

from pandarallel import pandarallel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch import optim

from matrix_preparation import matrix_preparation

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

nlp = spacy.load('en_core_web_lg')
hn_vocabulary = open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()

NUM_CLUSTERS = 10
CLAIM_THRESHOLD = 10
############################### ######### ###############################

################################ HELPERS ################################

def transform_to_clusters(papers_vec, claims_vec, prior):
	gmm = GaussianMixture(NUM_CLUSTERS,weights_init=prior).fit(papers_vec).fit(claims_vec)
	papers_vec = gmm.predict_proba(papers_vec)
	claims_vec = gmm.predict_proba(claims_vec)
	
	return papers_vec, claims_vec

# Hyper Parameters
num_epochs = 50
learning_rate = 1.e-3
weight_decay = 0.0
hidden = 50
gamma = 1.e-0
batch_size = 256#2048

class ClusterNet(nn.Module):
	def __init__(self):
		super(ClusterNet, self).__init__()
		
		self.papersNet = nn.Sequential(
			nn.Linear(NUM_CLUSTERS, NUM_CLUSTERS),
			nn.BatchNorm1d(NUM_CLUSTERS),
			nn.Softmax(dim=1),
		)
		
	def forward(self, P):
		return self.papersNet(P)	

	def loss(self, P, L, C):
		C_prime = L @ P
		return torch.norm(C_prime - C, p='fro')
############################### ######### ###############################

def align_clusters(cooc, papers_vec, claims_vec):

	cooc = torch.Tensor(cooc.astype(float))
	papers_vec = torch.Tensor(papers_vec.astype(float))
	claims_vec = torch.Tensor(claims_vec.astype(float))

	# cooc, index = np.unique(cooc, axis=0, return_index=True)
	# claims_vec = claims_vec[index]

	# claims_vec = torch.Tensor([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]])
	# cooc = torch.Tensor([[1, 0, 0, 0], [0, 0, 1, 0]])
	# papers_vec = torch.Tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
	# NUM_CLUSTERS = 10
			
	#Model training
	model = ClusterNet()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

	for epoch in range(num_epochs):
		p = np.random.permutation(len(papers_vec))

		mean_loss = []
		for i in range(0, len(p), batch_size):
			#P = model.P_prime[p[i:i+batch_size]]
			P = papers_vec[p[i:i+batch_size]]
			L = cooc[:, p[i:i+batch_size]]
			C = claims_vec
			P = Variable(P, requires_grad=True)
			L = Variable(L, requires_grad=False)   
			C = Variable(C, requires_grad=False)

			P = model(P)
			loss = model.loss(P, L, C)
			mean_loss.append(loss.data.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch%1 == 0:
			print(sum(mean_loss)/len(mean_loss))

	papers_vec = model(papers_vec).detach().numpy()
	return papers_vec


if __name__ == "__main__":
	cooc, papers_vec, claims_vec = matrix_preparation('PCA', 2, use_cache=True)

	prior = [1/NUM_CLUSTERS for _ in range(NUM_CLUSTERS)]
	
	for _ in range(2):
		papers_clust, claims_clust = transform_to_clusters(papers_vec.values, claims_vec.values, prior)
		papers_clust = align_clusters(cooc.values, papers_clust, claims_clust)

		pd.DataFrame(papers_clust, index=papers_vec.index).to_csv(sciclops_dir + 'cache/papers_vec_clusters.tsv.bz2', sep='\t')
		pd.DataFrame(claims_clust, index=claims_vec.index).to_csv(sciclops_dir + 'cache/claims_vec_clusters.tsv.bz2', sep='\t')