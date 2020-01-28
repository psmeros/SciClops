from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import spacy
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

CLAIM_THRESHOLD = 10
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())

def soft_labeling():	
	nlp = spacy.load('en_core_web_lg')

	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	titles = articles[['url', 'title']].drop_duplicates(subset='url')
	tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')

	G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')

	titles['prior'] = titles.apply(lambda x: [str(tweets.loc[t]['full_text']).split()[0] == str(x.title).split()[0] for t in G.predecessors(x.url) if t in tweets.index], axis=1).apply(lambda x: sum(x)/len(x) if len(x) else 0)

	#titles['prior'].hist()
	positive_samples = titles[titles.prior == 1].title

	#downsample negative samples
	negative_samples = articles['full_text'].sample(2*len(positive_samples))
	#split to list of sentences in list of paragraphs
	negative_samples = negative_samples.apply(lambda t: [list(nlp(p).sents) for p in t.split('\n')[2:-5] if p])
	#compute the probability of a sentence NOT to be a claim
	negative_samples = negative_samples.apply(lambda t: [(''.join(str(s)), (t.index(p)/len(t))*(p.index(s)/len(p))) for p in t for s in p if len(s) >= CLAIM_THRESHOLD])
	#keep the sentence with the max probability 
	negative_samples = negative_samples.apply(lambda s: max([('',0)]+s, key = lambda i : i[1])[0])

	positive_samples = pd.DataFrame(positive_samples).rename(columns={'title': 'sentence'})
	positive_samples['label'] = 1
	positive_samples

	negative_samples = pd.DataFrame(negative_samples).rename(columns={'full_text': 'sentence'})
	negative_samples = negative_samples[negative_samples.sentence.str.len() != 0][:len(positive_samples)]
	negative_samples['label'] = 0
	negative_samples

	pd.concat([positive_samples, negative_samples]).to_csv(sciclops_dir+'small_files/arguments/scientific.tsv', sep='\t', index=False)

def prepare_eval_dataset(gold_agreement):
	df = pd.read_csv(sciclops_dir + 'small_files/arguments/mturk_results.csv')

	df = df[['Input.sentence', 'Input.golden_label', 'Input.type', 'Answer.claim.label', 'LifetimeApprovalRate']]

	df = df.rename(columns={'Input.sentence':'sentence', 'Input.golden_label':'golden_label', 'Input.type':'type', 'Answer.claim.label':'label', 'LifetimeApprovalRate':'approval'})

	df = df.dropna()
	df = df[df.approval.apply(lambda x: int(re.sub('\%.*', '', x))) != 0]

	#aggregate results from crowdworkers
	df = pd.DataFrame(df.groupby(['sentence', 'type', 'golden_label'])['label'].apply(lambda x: (lambda c: (c.index[0], 'strong') if c.get(0) - c.get(1, default=0) > 1 else (c.index[0], 'weak') if c.get(0) - c.get(1, default=0) == 1 else np.nan)(x.value_counts())).apply(pd.Series))

	df = df.rename(columns={0:'label', 1: 'agreement'})
	df.label = df.label.map({'Yes':1, 'No':0})

	df = df.dropna().reset_index()


	
	if gold_agreement == 'strong':
		df =  df[(df.agreement == 'strong') & (df['golden_label']==df['label'])]
	elif gold_agreement == 'weak':
		df =  df[(df.agreement == 'weak')]

	return df[['sentence', 'label']]

############################### ######### ###############################

def train_BERT(model='bert-base-uncased'):
	df = pd.concat([pd.read_csv(sciclops_dir+'small_files/arguments/UKP_IBM.tsv', sep='\t').drop('topic', axis=1), pd.read_csv(sciclops_dir + 'small_files/arguments/scientific.tsv', sep='\t')])
	model = ClassificationModel('bert', model, use_cuda=False)
	model.train_model(df)

def eval_BERT(model, gold_agreement):
	df = prepare_eval_dataset(gold_agreement)
	model = ClassificationModel('bert', model, use_cuda=False)
	result, _, _ = model.eval_model(df)
	print (result)

def rule_based(gold_agreement):
	nlp = spacy.load('en_core_web_lg')
	
	def pattern_search(sentence):
		keywords = open(sciclops_dir + 'small_files/keywords/action.txt').read().splitlines()
		sentence = nlp(sentence)
		
		verbs = set()
		for v in sentence:
			if v.head.pos_ == 'VERB':
				verbs.add(v.head)

		if not verbs:
			return False

		rootVerb = ([w for w in sentence if w.head is w] or [None])[0]
		verbs = [rootVerb] + list(verbs)

		entities = [e.text for e in sentence.ents if e.label_ in ['PERSON', 'ORG']]

		for v in verbs:
			if v is not None and v.lemma_ in keywords:
				for np in v.children:
					if np.dep_ == 'nsubj':
						claimer = sentence[np.left_edge.i : np.right_edge.i+1].text
						if claimer in entities:
							return True

		return False

	df = prepare_eval_dataset(gold_agreement)
	df['pred'] = df.sentence.apply(lambda s: pattern_search(s))
	tn, fp, fn, tp = confusion_matrix(df['label'], df['pred']).ravel()
	print(tn, fp, fn, tp)

if __name__ == "__main__":
	rule_based(gold_agreement='strong')
	#eval_BERT(sciclops_dir + 'models/fine-tuned-bert-classifier', gold_agreement='weak')


