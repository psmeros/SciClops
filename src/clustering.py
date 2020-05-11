from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from spacy.lang.en.stop_words import STOP_WORDS
from torch import optim

from gsdmm import MovieGroupProcess

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))

nlp = spacy.load('en_core_web_lg')
for word in STOP_WORDS:
    for w in (word, word[0].capitalize(), word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True

np.random.seed(42)
torch.manual_seed(42)

# Hyper Parameters
num_epochs = 50
learning_rate = 1.e-3
hidden = 50
batch_size = 128
beta = 1.e-3
gamma = 1.e-3
############################### ######### ###############################

################################ HELPERS ################################

def load_matrices(representation, dimension=None):
	cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	claims = pd.read_csv(sciclops_dir + 'cache/claims_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	papers = pd.read_csv(sciclops_dir + 'cache/papers_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'title', 'popularity'])
	return cooc, papers, claims

def standalone_clustering(method):
	dimension = 10 if method.startswith('PCA') else None

	if method.endswith('GMM'):
		cooc, papers, claims = load_matrices(representation='embeddings', dimension=dimension)

		cooc = cooc.values
		papers_index = papers.index
		claims_index = claims.index
		papers = papers.values
		claims = claims.values
		
		model = GaussianMixture(NUM_CLUSTERS, covariance_type='spherical', tol=0.5, random_state=42).fit(np.concatenate([claims, papers]))
		claims_clusters = model.predict_proba(claims)
		papers_clusters = model.predict_proba(papers)
		
	elif method.endswith('KMeans'):
		cooc, papers, claims = load_matrices(representation='embeddings', dimension=dimension)

		cooc = cooc.values
		papers_index = papers.index
		claims_index = claims.index		
		papers = papers.values
		claims = claims.values
		
		model = KMeans(NUM_CLUSTERS, random_state=42).fit(np.concatenate([claims, papers]))
		c_cluster = model.predict(claims)
		p_cluster = model.predict(papers)

		claims_clusters = np.zeros((len(claims), NUM_CLUSTERS))
		claims_clusters[np.arange(len(claims)), c_cluster] = 1
		papers_clusters = np.zeros((len(papers), NUM_CLUSTERS))
		papers_clusters[np.arange(len(papers)), p_cluster] = 1

	elif method == 'LDA':
		cooc, papers, claims = load_matrices(representation='textual')
		cooc = cooc.values
		papers_index = papers.index
		claims_index = claims.index
		papers = papers['clean_passage']
		claims = claims['clean_claim']
		
		CV = CountVectorizer().fit(pd.concat([claims, papers]))

		model = LatentDirichletAllocation(n_components=NUM_CLUSTERS, n_jobs=-1).fit(CV.transform(pd.concat([claims, papers])))
		papers_clusters = model.transform(CV.transform(papers))
		claims_clusters = model.transform(CV.transform(claims))

	elif method == 'GSDMM':
		cooc, papers, claims = load_matrices(representation='textual')
		cooc = cooc.values
		papers_index = papers.index
		claims_index = claims.index
		
		c_cluster = MovieGroupProcess(K=NUM_CLUSTERS, n_iters=5).fit(claims['clean_claim'], len(set([e for l in claims['clean_claim'].tolist() for e in l])))
		p_cluster = MovieGroupProcess(K=NUM_CLUSTERS, n_iters=5).fit(papers['clean_passage'], len(set([e for l in papers['clean_passage'].tolist() for e in l])))
		
		claims_clusters = np.zeros((len(claims), NUM_CLUSTERS))
		claims_clusters[np.arange(len(claims)), c_cluster] = 1
		papers_clusters = np.zeros((len(papers), NUM_CLUSTERS))
		papers_clusters[np.arange(len(papers)), p_cluster] = 1

	papers_clusters = pd.DataFrame(papers_clusters, index=papers_index)
	claims_clusters = pd.DataFrame(claims_clusters, index=claims_index)

	return papers, claims, papers_clusters, claims_clusters, cooc

class ClusterNet(nn.Module):
	def __init__(self, clustering_type, init_clustering_method):
		super(ClusterNet, self).__init__()
		
		self.clustering_type = clustering_type

		if 'compute_C' in self.clustering_type:
			self.papers, _, papers_clusters, claims_clusters, self.cooc = standalone_clustering(method=init_clustering_method)
			
			self.papers_index = papers_clusters.index
			self.claims_index = claims_clusters.index
			self.claims_clusters = claims_clusters.values
			self.cooc_unique, index = np.unique(self.cooc, axis=0, return_index=True)
			self.claims_unique = self.claims_clusters[index]

			self.cooc_unique = torch.Tensor(self.cooc_unique.astype(float))
			self.papers = torch.Tensor(self.papers.astype(float))
			self.claims_unique = torch.Tensor(self.claims_unique.astype(float))

			if 'transform_P' in self.clustering_type:
				self.papersNet = nn.Sequential(
					nn.Linear(self.papers.shape[1], hidden),
					nn.BatchNorm1d(hidden),
					nn.ReLU(),
					nn.Linear(hidden, NUM_CLUSTERS),
					nn.Softmax(dim=1)
				)
			elif 'align_P' in self.clustering_type:
				self.papers_clusters = nn.Parameter(nn.init.eye_(torch.Tensor(self.papers.shape[0], NUM_CLUSTERS)), requires_grad=True)

		elif 'compute_P' in self.clustering_type:
			_, self.claims, papers_clusters, claims_clusters, self.cooc = standalone_clustering(method=init_clustering_method)

			self.papers_index = papers_clusters.index
			self.claims_index = claims_clusters.index
			self.papers_clusters = papers_clusters.values
			self.cooc_unique, index = np.unique(self.cooc, axis=1, return_index=True)
			self.papers_unique = self.papers_clusters[index]

			self.cooc_unique = torch.Tensor(self.cooc_unique.astype(float))
			self.claims = torch.Tensor(self.claims.astype(float))
			self.papers_unique = torch.Tensor(self.papers_unique.astype(float))

			if 'transform_C' in self.clustering_type:
				self.claimsNet = nn.Sequential(
					nn.Linear(self.claims.shape[1], hidden),
					nn.BatchNorm1d(hidden),
					nn.ReLU(),
					nn.Linear(hidden, NUM_CLUSTERS),
					nn.Softmax(dim=1)
				)
			elif 'align_C' in self.clustering_type:
				self.claims_clusters = nn.Parameter(nn.init.eye_(torch.Tensor(self.claims.shape[0], NUM_CLUSTERS)), requires_grad=True)

		elif self.clustering_type in ['coordinate-transform', 'coordinate-align', 'compute-align']:
			self.papers, self.claims, papers_clusters, claims_clusters, self.cooc = standalone_clustering(method=init_clustering_method)
			
			self.papers_index = papers_clusters.index
			self.claims_index = claims_clusters.index

			self.cooc_unique_C, self.index_C = np.unique(self.cooc, axis=0, return_index=True)
			self.cooc_unique_P, self.index_P = np.unique(self.cooc, axis=1, return_index=True)

			self.cooc_unique_C = torch.Tensor(self.cooc_unique_C.astype(float))
			self.papers = torch.Tensor(self.papers.astype(float))
			self.cooc_unique_P = torch.Tensor(self.cooc_unique_P.astype(float))
			self.claims = torch.Tensor(self.claims.astype(float))

			if 'coordinate-transform' in self.clustering_type:
				self.claimsNet = nn.Sequential(
					nn.Linear(self.claims.shape[1], hidden),
					nn.ReLU(),
					nn.Linear(hidden, NUM_CLUSTERS),
					nn.Softmax(dim=1)
				)
				self.papersNet = nn.Sequential(
					nn.Linear(self.papers.shape[1], hidden),
					nn.ReLU(),
					nn.Linear(hidden, NUM_CLUSTERS),
					nn.Softmax(dim=1)
				)
				
			elif 'coordinate-align' in self.clustering_type:
				self.claims_clusters = nn.Parameter(nn.init.eye_(torch.Tensor(self.claims.shape[0], NUM_CLUSTERS)), requires_grad=True)
				self.papers_clusters = nn.Parameter(nn.init.eye_(torch.Tensor(self.papers.shape[0], NUM_CLUSTERS)), requires_grad=True)

			elif 'compute-align' in self.clustering_type:
				self.papers_clusters_orig = torch.Tensor(papers_clusters.values.astype(float))
				self.claims_clusters_orig = torch.Tensor(claims_clusters.values.astype(float))

				self.papers_clusters = nn.Parameter(torch.Tensor(papers_clusters.values.astype(float)), requires_grad=True)
				self.claims_clusters = nn.Parameter(torch.Tensor(claims_clusters.values.astype(float)), requires_grad=True)


	def compute_permutation(self, epoch):
		if 'transform_P' in self.clustering_type or 'align_P' in self.clustering_type: 
			self.permutation = np.random.permutation(len(self.papers))	
		elif 'transform_C' in self.clustering_type or 'align_C' in self.clustering_type:
			self.permutation = np.random.permutation(len(self.claims))
		else:
			self.permutation = np.random.permutation(len(self.papers)) if epoch%2==0 else np.random.permutation(len(self.claims))
			self.epoch = epoch

		return self.permutation

	def forward(self, batch):
		if 'compute_C' in self.clustering_type:
			L = self.cooc_unique[:, self.permutation[batch:batch+batch_size]]
			C = self.claims_unique

			if 'transform_P' in self.clustering_type:
				P = self.papersNet(self.papers[self.permutation[batch:batch+batch_size]])
			elif 'align_P' in self.clustering_type:
				P = self.papers_clusters[self.permutation[batch:batch+batch_size]]

		elif 'compute_P' in self.clustering_type:
			L = self.cooc_unique[self.permutation[batch:batch+batch_size]]
			P = self.papers_unique

			if 'transform_C' in self.clustering_type:
				C = self.claimsNet(self.claims[self.permutation[batch:batch+batch_size]])
			elif 'align_C' in self.clustering_type:
				C = self.claims_clusters[self.permutation[batch:batch+batch_size]]
		
		elif self.clustering_type in ['coordinate-transform', 'coordinate-align', 'compute-align']:
		
			if self.epoch%2==0:
				L = self.cooc_unique_C[:, self.permutation[batch:batch+batch_size]]

				if  'compute-align' in self.clustering_type:
					C = torch.Tensor(self.claims_clusters[self.index_C].detach().numpy().astype(float))
					P = self.papers_clusters[self.permutation[batch:batch+batch_size]]
					self.P_orig = self.papers_clusters_orig[self.permutation[batch:batch+batch_size]]
				elif 'coordinate-align' in self.clustering_type:
					C = torch.Tensor(self.claims_clusters[self.index_C].detach().numpy().astype(float))
					P = self.papers_clusters[self.permutation[batch:batch+batch_size]]
				elif 'coordinate-transform' in self.clustering_type:
					C = torch.Tensor(self.claimsNet(self.claims[self.index_C]).detach().numpy().astype(float))
					P = self.papersNet(self.papers[self.permutation[batch:batch+batch_size]])
			else:
				L = self.cooc_unique_P[self.permutation[batch:batch+batch_size]]
				
				if'compute-align' in self.clustering_type:
					P = torch.Tensor(self.papers_clusters[self.index_P].detach().numpy().astype(float))
					C = self.claims_clusters[self.permutation[batch:batch+batch_size]]
					self.C_orig = self.claims_clusters_orig[self.permutation[batch:batch+batch_size]]
				elif 'coordinate-align' in self.clustering_type:
					P = torch.Tensor(self.papers_clusters[self.index_P].detach().numpy().astype(float))
					C = self.claims_clusters[self.permutation[batch:batch+batch_size]]
				elif 'coordinate-transform' in self.clustering_type:
					P = torch.Tensor(self.papersNet(self.papers[self.index_P]).detach().numpy().astype(float))
					C = self.claimsNet(self.claims[self.permutation[batch:batch+batch_size]])

		return P, L, C

	def final_clusters(self):
		if 'compute_C' in self.clustering_type:
			claims_clusters, cooc = self.claims_clusters, self.cooc
			if 'transform_P' in self.clustering_type:
				papers_clusters = self.papersNet(self.papers).detach().numpy()
			elif 'align_P' in self.clustering_type:
				papers_clusters = self.papers_clusters.detach().numpy()

		elif 'compute_P' in self.clustering_type:
			papers_clusters, cooc = self.papers_clusters, self.cooc
			if 'transform_C' in self.clustering_type:
				claims_clusters = self.claimsNet(self.claims).detach().numpy()
			elif 'align_C' in self.clustering_type:
				claims_clusters = self.claims_clusters.detach().numpy()

		elif self.clustering_type in ['coordinate-transform', 'coordinate-align', 'compute-align']:
			cooc = self.cooc
			if 'coordinate-align' in self.clustering_type or 'compute-align' in self.clustering_type:
				papers_clusters = self.papers_clusters.detach().numpy()
				claims_clusters = self.claims_clusters.detach().numpy()
			elif 'coordinate-transform' in self.clustering_type:
				papers_clusters = self.papersNet(self.papers).detach().numpy()
				claims_clusters = self.claimsNet(self.claims).detach().numpy()

		papers_clusters = pd.DataFrame(papers_clusters, index=self.papers_index)
		claims_clusters = pd.DataFrame(claims_clusters, index=self.claims_index)

		return papers_clusters, claims_clusters, cooc

	def loss(self, P, L, C):
		if 'compute-align' in self.clustering_type:
			if self.epoch%2==0:
				return gamma * torch.norm((L @ P) - C, p='fro') + (1 - gamma) * torch.norm(self.P_orig - P, p='fro')
			else:
				return gamma * torch.norm((L @ P) - C, p='fro') + (1 - gamma) * torch.norm(self.C_orig - C, p='fro')
		else:
			return torch.norm((L @ P) - C, p='fro') - beta * (torch.norm(P, p='fro') + torch.norm(C, p='fro'))
		

############################### ######### ###############################

def compute_clusterings(clustering_type, init_clustering_method=None):

	if clustering_type in ['LDA', 'GSDMM', 'GMM', 'PCA-GMM', 'KMeans', 'PCA-KMeans']:
		_, _, papers_clusters, claims_clusters, cooc = standalone_clustering(clustering_type)
		return papers_clusters, claims_clusters, cooc
	elif clustering_type.startswith('compute-align'):
		global gamma 
		gamma = float(clustering_type.split('-')[2])
		clustering_type = 'compute-align'
	
	model = ClusterNet(clustering_type, init_clustering_method)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

	for epoch in range(num_epochs):
		permutation = model.compute_permutation(epoch)

		mean_loss = []
		for batch in range(0, len(permutation), batch_size):
			optimizer.zero_grad()
			P, L, C = model.forward(batch)
			loss = model.loss(P, L, C)
			mean_loss.append(loss.detach().numpy())
			loss.backward()
			optimizer.step()

	papers_clusters, claims_clusters, cooc = model.final_clusters()
	return papers_clusters, claims_clusters, cooc


def eval_clusters(papers_clusters, claims_clusters, cooc):
	#papers_clusters, claims_clusters, cooc = compute_clusterings('LDA', 'PCA-GMM')
	#threshold for faster computation; it has to be 100% for full comparison
	EVAL_THRESHOLD = 1.0
	
	papers_index = papers_clusters.index
	claims_index = claims_clusters.index
	papers_clusters = papers_clusters.values
	claims_clusters = claims_clusters.values

	# P@k
	k = 3

	top_papers = np.unique((-papers_clusters).argsort(axis=0)[:int(EVAL_THRESHOLD*len(papers_clusters))].flatten())

	P = papers_clusters[top_papers]
	L = cooc[:, top_papers]
	mask = (~np.all(L == 0, axis=1))
	L = L[mask]
	C = claims_clusters[mask]

	P_at_k = np.apply_along_axis(lambda x : {i:x[i] for i in np.argsort(x)[-k:]}, 1, P)

	labels_inherited = []
	for Li in L:
		z = Counter()
		for d in P_at_k[np.nonzero(Li)[0]]:
			z.update(Counter(d))
		labels_inherited += [sorted(z, key=z.get, reverse=True)[:k]]

	labels_inherited = np.array(labels_inherited)
	labels_expected = np.argsort(C, axis=1)[:, -k:]

	p1 = [len(np.setdiff1d(li, le, assume_unique=True))<k for li, le in zip(labels_inherited, labels_expected)]
	p1 = sum(p1)/len(p1)


	top_claims = np.unique((-claims_clusters).argsort(axis=0)[:int(EVAL_THRESHOLD*len(claims_clusters))].flatten())

	C = claims_clusters[top_claims]
	L = cooc[top_claims]
	mask = (~np.all(L == 0, axis=0))
	L = L[:, mask]
	P = papers_clusters[mask]

	C_at_k = np.apply_along_axis(lambda x : {i:x[i] for i in np.argsort(x)[-k:]}, 1, C)

	labels_inherited = []
	for Li in L.T:
		z = Counter()
		for d in C_at_k[np.nonzero(Li)[0]]:
			z.update(Counter(d))
		labels_inherited += [sorted(z, key=z.get, reverse=True)[:k]]

	labels_inherited = np.array(labels_inherited)
	labels_expected = np.argsort(P, axis=1)[:, -k:]

	p2 = [len(np.setdiff1d(li, le, assume_unique=True))<k for li, le in zip(labels_inherited, labels_expected)]
	p2 = sum(p2)/len(p2)

	p = np.mean([p1, p2])


	#Average Silhouette Width
	papers_clusters = pd.DataFrame(papers_clusters[top_papers], index=papers_index[top_papers])
	papers_clusters_repr = papers_clusters.reset_index(['url', 'popularity'], drop=True).idxmax().reset_index().rename(columns={'index':'cluster', 0:'title'})
	papers_clusters = papers_clusters.idxmax(axis=1)
	papers_clusters = papers_clusters.reset_index().drop(['url', 'popularity'], axis=1).rename(columns={0:'cluster'})

	claims_clusters = pd.DataFrame(claims_clusters[top_claims], index=claims_index[top_claims])
	claims_clusters_repr = claims_clusters.reset_index(['url', 'popularity'], drop=True).idxmax().reset_index().rename(columns={'index':'cluster', 0:'claim'})
	claims_clusters = claims_clusters.idxmax(axis=1)
	claims_clusters = claims_clusters.reset_index().drop(['url', 'popularity'], axis=1).rename(columns={0:'cluster'})

	def clean_text(text):
		return nlp(' '.join([token.text.lower() for token in nlp(text) if not (token.is_punct | token.is_space | token.is_stop)]))

	papers_clusters['title_clean'] = papers_clusters['title'].apply(clean_text)
	papers_clusters_repr['title_clean'] = papers_clusters_repr['title'].apply(clean_text)
	claims_clusters['claim_clean'] = claims_clusters['claim'].apply(clean_text)
	claims_clusters_repr['claim_clean'] = claims_clusters_repr['claim'].apply(clean_text)

	def sts(text_1, text_2):
		semantic = text_1.similarity(text_2)
		
		text_1 = set(text_1.text.split()).intersection(hn_vocabulary)
		text_2 = set(text_2.text.split()).intersection(hn_vocabulary)
		jaccard = len(text_1.intersection(text_2)) / (len(text_1.union(text_2)) or 1)

		return np.mean([semantic,jaccard])

	papers = papers_clusters.merge(claims_clusters_repr)
	mean_pc = papers.apply(lambda p: sts(p['claim_clean'], p['title_clean']), axis=1).mean()

	claims = claims_clusters.merge(papers_clusters_repr)
	mean_cp = claims.apply(lambda p: sts(p['claim_clean'], p['title_clean']), axis=1).mean()

	papers = papers_clusters.merge(papers_clusters_repr, on='cluster')
	mean_pp = papers.apply(lambda p: sts(p['title_clean_x'], p['title_clean_y']), axis=1).mean()

	claims = claims_clusters.merge(claims_clusters_repr, on='cluster')
	mean_cc = claims.apply(lambda p: sts(p['claim_clean_x'], p['claim_clean_y']), axis=1).mean()

	asw = np.mean([mean_pc, mean_cp, mean_pp, mean_cc])
		
	return p, asw
	

if __name__ == "__main__":
	compare = False
	if compare:
		clustering_types = ['LDA', 'GSDMM', 'GMM', 'PCA-GMM', 'KMeans', 'PCA-KMeans', 'compute_C_transform_P', 'compute_C_align_P', 'compute_P_transform_C', 'compute_P_align_C', 'coordinate-align', 'coordinate-transform', 'compute-align-0.1', 'compute-align-0.5', 'compute-align-0.9']
		results = []
		for NUM_CLUSTERS in [10, 20, 50, 100]:
			for clustering_type in clustering_types:
				papers_clusters, claims_clusters, cooc = compute_clusterings(clustering_type, 'GMM')
				p, asw = eval_clusters(papers_clusters, claims_clusters, cooc)
				results += [[NUM_CLUSTERS, clustering_type, p, asw]]
			print(results)
		
		pd.DataFrame(results, columns=['clusters', 'method', 'P@3', 'ASW']).to_csv(sciclops_dir + 'cache/clustering_results.tsv', sep='\t', index=None)

	else:
		NUM_CLUSTERS = 100
		papers_clusters, claims_clusters, _ = compute_clusterings('compute-align-0.5', 'GMM')
		papers_clusters.to_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')
		claims_clusters.to_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
