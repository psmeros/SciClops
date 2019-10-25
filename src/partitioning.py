import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from pandarallel import pandarallel

import pke

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

############################### ######### ###############################

################################ HELPERS ################################

def keyphrase_extraction(text, n=10):	
	extractor = pke.unsupervised.MultipartiteRank()
	extractor.load_document(input=text)

	pos = {'NOUN', 'PROPN', 'ADJ'}
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += stopwords.words('english')
	extractor.candidate_selection(pos=pos, stoplist=stoplist)
	extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')

	keyphrases = extractor.get_n_best(n)
	return keyphrases

############################### ######### ###############################

pandarallel.initialize()

def extract_keyphrases():
	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	articles['keyphrases'] = articles.apply(lambda a: keyphrase_extraction(str(a.title) + ' ' + str(a.full_text)), axis=1)

	ght = [l.strip() for e in list(pd.read_csv(sciclops_dir + 'small_files/global_health_threats/ght.tsv', sep='\t').Keywords) for l in e.split('|')]
	articles['partition'] = articles.keyphrases.apply(lambda l: set([kph for sl in [k.split() for (k, _) in l] for kph in sl if kph in ght]))
	articles.to_csv(scilens_dir + 'article_details_v4.tsv.bz2', sep='\t', index=False)


def filter_partitions(keyphrase):
	articles = pd.read_csv(scilens_dir + 'article_details_v4.tsv.bz2', sep='\t')
	articles = articles[articles['partition'].apply(lambda s: keyphrase in s)]
	articles.to_csv(sciclops_dir + keyphrase + '_articles.tsv.bz2', sep='\t', index=False)

def partition_relations(clean_claims, partition):
	
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
	
