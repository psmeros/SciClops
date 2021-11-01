from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from gsdmm import MovieGroupProcess
from pandarallel import pandarallel
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.lang.en.stop_words import STOP_WORDS
from torch import optim

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))

CLAIM_THRESHOLD = 10
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
	text = [w for w in hn_vocabulary if w in text]
	return text

def matrix_preparation(representations, pca_dimensions=None):
	pandarallel.initialize(verbose=0)
	
	claims = pd.read_csv(sciclops_dir+'cache/claims_raw.tsv.bz2', sep='\t')

	G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
	G.remove_nodes_from(open(sciclops_dir + 'small_files/blacklist/sources.txt').read().splitlines())
	
	papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url')

	print('cleaning papers...')
	#papers['clean_passage'] = papers.title + ' ' + papers.full_text.parallel_apply(lambda w: w.split('\n')[0])
	#papers['clean_passage'] = clean_paper(papers['clean_passage'])
	papers['clean_passage'] = papers.title.parallel_apply(clean_paper)
	papers = papers[papers['clean_passage'].str.len() != 0]
	papers['popularity'] = papers.url.parallel_apply(lambda u: G.in_degree(u))
	refs = set(papers['url'].unique())

	print('cleaning claims...')	
	claims['refs'] = claims.url.parallel_apply(lambda u: set(G.successors(u)).intersection(refs))
	claims = claims[claims['refs'].str.len() != 0]

	tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
	claims['popularity'] = claims.url.parallel_apply(lambda u: sum([tweets.loc[t]['popularity'] for t in G.predecessors(u) if t in tweets.index]))

	claims.claim = claims.claim.apply(eval)
	claims = claims.explode('claim')

	claims['clean_claim'] = claims['claim'].parallel_apply(clean_claim)
	claims = claims[claims['clean_claim'].str.len() != 0]
	refs = set([e for l in claims['refs'].to_list() for e in l])
	papers = papers[papers['url'].isin(refs)]

	papers = papers.set_index(['url', 'title', 'popularity'])
	claims = claims.set_index(['url', 'claim', 'popularity'])
	papers_index = papers.index
	claims_index = claims.index

	mlb = MultiLabelBinarizer()
	cooc = pd.DataFrame(mlb.fit_transform(claims.refs), columns=mlb.classes_, index=claims.index)
	cooc.to_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t')

	for representation in representations:
		print('transforming...')
		if representation =='textual':
			papers_vec = papers['clean_passage'].parallel_apply(lambda x: ' '.join(x))
			claims_vec = claims['clean_claim'].parallel_apply(lambda x: ' '.join(x))

		elif representation =='embeddings':
			papers_vec = papers['clean_passage'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values
			claims_vec = claims['clean_claim'].parallel_apply(lambda x: nlp(' '.join(x)).vector).apply(pd.Series).values

		print('caching...')
		if representation == 'embeddings' and pca_dimensions != None:
			for dimension in pca_dimensions:
				pca = TruncatedSVD(dimension).fit(claims_vec).fit(papers_vec)
				pd.DataFrame(pca.transform(papers_vec), index=papers_index).to_csv(sciclops_dir + 'cache/papers_'+representation+'_'+str(dimension)+'.tsv.bz2', sep='\t')
				pd.DataFrame(pca.transform(claims_vec), index=claims_index).to_csv(sciclops_dir + 'cache/claims_'+representation+'_'+str(dimension)+'.tsv.bz2', sep='\t')	

		pd.DataFrame(papers_vec, index=papers_index).to_csv(sciclops_dir + 'cache/papers_'+representation+'.tsv.bz2', sep='\t')
		pd.DataFrame(claims_vec, index=claims_index).to_csv(sciclops_dir + 'cache/claims_'+representation+'.tsv.bz2', sep='\t')	

def load_matrices(representation, dimension=None):
	matrix_preparation(representations=['textual','embeddings'], pca_dimensions=[10])
	cooc = pd.read_csv(sciclops_dir + 'cache/cooc.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	claims = pd.read_csv(sciclops_dir + 'cache/claims_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'claim', 'popularity'])
	papers = pd.read_csv(sciclops_dir + 'cache/papers_'+representation+('_'+str(dimension) if dimension else '')+'.tsv.bz2', sep='\t', index_col=['url', 'title', 'popularity'])
	return cooc, papers, claims

def popular_clusters():
  NUM_CLUSTERS = 100 
  claims_clusters = pd.read_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
  papers_clusters = pd.read_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')
  
  claims_popularity = claims_clusters['popularity']
  papers_popularity = papers_clusters['popularity']
  
  claims_rank = [sum(claims_clusters[str(i)]*claims_popularity) for i in range(NUM_CLUSTERS)]
  claims_rank = [r/sum(claims_rank) for r in claims_rank]
  papers_rank = [sum(papers_clusters[str(i)]*papers_popularity) for i in range(NUM_CLUSTERS)]
  papers_rank = [r/sum(papers_rank) for r in papers_rank]

  rank = np.flip(np.argsort(np.array(claims_rank) + np.array(papers_rank)))

  for c in range(NUM_CLUSTERS):
    print('cluster: ', c)
    claims_centroid = claims_clusters.iloc[claims_clusters[str(c)].argmax()]['claim'].lower()
    print(claims_centroid)
    claims_centroid = set(claims_centroid.split()).intersection(hn_vocabulary)
    
    papers_centroid = papers_clusters.iloc[papers_clusters[str(c)].argmax()]['title'].lower()
    print(papers_centroid)
    papers_centroid = set(papers_centroid.split()).intersection(hn_vocabulary)

    print(claims_centroid.union(papers_centroid))

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
	clustering_types = ['LDA', 'GSDMM', 'GMM', 'PCA-GMM', 'KMeans', 'PCA-KMeans', 'compute_C_transform_P', 'compute_C_align_P', 'compute_P_transform_C', 'compute_P_align_C', 'coordinate-align', 'coordinate-transform', 'compute-align-0.1', 'compute-align-0.5', 'compute-align-0.9']
	results = []
	for NUM_CLUSTERS in [10, 20, 50, 100]:
		for clustering_type in clustering_types:
			papers_clusters, claims_clusters, cooc = compute_clusterings(clustering_type, 'GMM')
			p, asw = eval_clusters(papers_clusters, claims_clusters, cooc)
			results += [[NUM_CLUSTERS, clustering_type, p, asw]]
		print(results)
		
	pd.DataFrame(results, columns=['clusters', 'method', 'P@3', 'ASW']).to_csv(sciclops_dir + 'cache/clustering_results.tsv', sep='\t', index=None)

	df = pd.read_csv(sciclops_dir + 'cache/clustering_results.tsv', sep='\t')
	mapping = {'LDA':'LDA', 'GSDMM':'GSDMM', 'GMM':'GMM', 'PCA-GMM':'PCA/GMM', 'KMeans':'K-Means', 'PCA-KMeans':'PCA/K-Means', 'coordinate-align':'GBA-CP', 'compute_P_align_C':'GBA-C', 'compute_C_align_P':'GBA-P', 'coordinate-transform':'GBT-CP', 'compute_P_transform_C':'GBT-C', 'compute_C_transform_P':'GBT-P', 'compute-align-0.1':'AO-Content', 'compute-align-0.5':'AO-Balanced', 'compute-align-0.9':'AO-Graph'}
	df.method = df.method.apply(lambda x: mapping[x])

	df = df.pivot(index='method', columns='clusters', values=['ASW', 'P@3']).reindex(mapping.values()).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False).applymap(lambda x:'{0:04.1f}%'.format(100 * x))#.round(decimals=3) * 100
	print(df.to_latex())
	
	NUM_CLUSTERS = 100
	papers_clusters, claims_clusters, _ = compute_clusterings('compute-align-0.5', 'GMM')
	papers_clusters.to_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')
	claims_clusters.to_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
