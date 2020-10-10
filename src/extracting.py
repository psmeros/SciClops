import os
import random
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from simpletransformers.classification import ClassificationModel
from simpletransformers.language_modeling import (LanguageModelingArgs,
                                                  LanguageModelingModel)
from sklearn.metrics import precision_recall_fscore_support

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

CLAIM_THRESHOLD = 10
LIFT_THRESHOLD = .8
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())


def annotation_sampling(num, max_sents=5):
	sentences = articles[['title', 'full_text']].sample(num)
	sentences = sentences.apply(lambda r: [r['title']] + [re.sub('\n', '', s.text) for _,s in zip(range(max_sents), nlp(r['full_text']).sents) if len(s) >= CLAIM_THRESHOLD and s[0].is_upper], axis=1)
	weights = sentences.apply(lambda l: [len(l) - l.index(s) for s in l])
	
	df = pd.DataFrame([random.choices(sentences[i], weights[i])[0] for i in range(num)], columns=['sentence'])
	df.to_csv(sciclops_dir + 'etc/arguments/validation_set.csv', index=False)

def negative_sampling(training, num, max_prob=True):
	#separate training and testing negative samples
	negative_samples = articles['full_text'][1000:].sample(num) if training else articles['full_text'][:1000].sample(num)
	#split to list of sentences in list of paragraphs
	negative_samples = negative_samples.apply(lambda t: [list(nlp(p).sents) for p in t.split('\n')[2:-5] if p])
	#compute the probability of a sentence NOT to be a claim
	negative_samples = negative_samples.apply(lambda t: [(''.join(str(s)), (t.index(p)/len(t))*(p.index(s)/len(p))) for p in t for s in p if len(s) >= CLAIM_THRESHOLD])
	#keep the sentence with the max probability
	i = -1 if max_prob else -2
	negative_samples = negative_samples.apply(lambda s: sorted([('',0)]+[('',0)]+s, key=lambda i: i[1])[i][0]).tolist()
	negative_samples = [s for s in negative_samples if s!= '']

	return negative_samples


def prepare_eval_dataset(gold_agreement):
	df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results.csv')

	df = df.rename(columns={'Input.sentence':'sentence', 'Answer.False.False':'False', 'Answer.NA.NA':'NA', 'Answer.True.True':'True'})
	df = df[['sentence', 'False', 'NA', 'True']]

	df = df.groupby('sentence').idxmax(axis=1).reset_index().rename(columns={ 0:'label'})[['sentence', 'label']]

	df = pd.DataFrame(df.groupby('sentence').apply(lambda x: (lambda c: (c.index[0][1], 'strong') if c.get(0) - c.get(1, default=0) > 1 else (c.index[0][1], 'weak') if c.get(0) - c.get(1, default=0) == 1 else np.nan)(x.value_counts())).apply(pd.Series)).dropna()

	df = df.rename(columns={0:'label', 1: 'agreement'}).reset_index()
	df.label = df.label.map({'True':1, 'False':0})

	return df[(df.agreement == gold_agreement)]


nlp = spacy.load('en_core_web_lg')
articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')

############################### ######### ###############################

def pretrain_BERT(model='bert-base-uncased', use_cuda=False):
	filename = '_df.csv' 
	df = pd.read_csv(sciclops_dir+'etc/million_headlines/abcnews.csv').drop('publish_date', axis=1)
	df.to_csv(filename, index=None, header=False)
	model_args = LanguageModelingArgs()
	model_args.fp16 = False
	model = LanguageModelingModel('bert', model, use_cuda=use_cuda, args=model_args)
	model.train_model(filename)
	os.remove(filename)


def train_BERT(model='bert-base-uncased', weak_labels=False, use_cuda=False):
	if weak_labels:
		df = pd.concat([pd.read_csv(sciclops_dir+'etc/arguments/UKP_IBM.tsv', sep='\t').drop('topic', axis=1), pd.read_csv(sciclops_dir + 'etc/arguments/scientific.tsv', sep='\t')])
	else:
		df = pd.read_csv(sciclops_dir+'etc/arguments/UKP_IBM.tsv', sep='\t').drop('topic', axis=1)
	
	model = ClassificationModel('bert', model, use_cuda)
	model.train_model(df)

def eval_BERT(model, gold_agreement):
	df = prepare_eval_dataset(gold_agreement)
	model = ClassificationModel('bert', model, use_cuda=False)
	result, _, _ = model.eval_model(df)
	p = result['tp']/(result['tp']+result['fp'])
	r = result['tp']/(result['tp']+result['fn'])
	f1 = 2*p*r/(p+r)
	print (p,r,f1)

def pred_BERT(model, claimKG=False):
	model = ClassificationModel('bert', model, use_cuda=False)

	if claimKG:
		claimsKG = pd.read_csv(sciclops_dir+'etc/claimKG/claims.csv') 
		claimsKG['label'], _ = model.predict(claimsKG.claimText)
		claimsKG = claimsKG[claimsKG.label == 1].drop('label', axis=1)
		claimsKG.to_csv(sciclops_dir+'etc/claimKG/claims_clean.csv', index=False)

	else:
		articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
		titles = articles[['url', 'title']].drop_duplicates(subset='url').rename(columns={'title': 'claim'})
		articles = articles[['url', 'quotes']].drop_duplicates(subset='url')
		articles.quotes = articles.quotes.apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
		articles = articles.explode('quotes').rename(columns={'quotes': 'claim'})
		articles = pd.concat([articles, titles])
		articles = articles[~articles['claim'].isna()]

		articles['label'], _ = model.predict(articles.claim)

		articles = articles[articles.label == 1].drop('label', axis=1)
		articles = articles.groupby('url')['claim'].apply(list).reset_index()
		articles.to_csv(sciclops_dir+'cache/claims_raw.tsv.bz2', sep='\t', index=False)

def rule_based(gold_agreement, how):

	def pattern_search(sentence):
		sentence = nlp(sentence)
		
		action = open(sciclops_dir + 'etc/keywords/action.txt').read().splitlines()
		person = open(sciclops_dir + 'etc/keywords/person.txt').read().splitlines()
		study = open(sciclops_dir + 'etc/keywords/study.txt').read().splitlines()
		vocabulary = open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
		entities = [e.text for e in sentence.ents if e.label_ in ['PERSON', 'ORG']]
		verbs = ([w for w in sentence if w.dep_=='ROOT'] or [None])

		for v in verbs:
			if v.text in action:
				for np in v.children:
					if np.dep_ in ['nsubj', 'dobj']:
						claimer = sentence[np.left_edge.i : np.right_edge.i+1].text
						for w in entities:
							if w in claimer:
								return True 
		
			for np in v.children:
				if np.dep_ in ['nsubj', 'dobj']:
					claimer = sentence[np.left_edge.i : np.right_edge.i+1].text
					for w in vocabulary+person+study:
						if w in claimer:
							return True 
	
		return False


	def max_lift(sentence):

		article_url = list(articles[articles['title'].str.find(sentence) != -1].dropna().index) + list(articles[articles['full_text'].str.find(sentence) != -1].dropna().index)

		if not article_url:
			return False

		related_tweets = [tweets.loc[t] for t in G.predecessors(article_url[0]) if t in tweets.index]

		if not related_tweets:
			return False

		overall_popularity =  sum([t['popularity'] for t in related_tweets])
		support = [t['popularity']/overall_popularity for t in related_tweets]

		confidence = [nlp(t['full_text']).similarity(nlp(sentence)) for t in related_tweets]

		max_lift = max([c/s for s,c in zip(support, confidence)])

		if max_lift > LIFT_THRESHOLD:
			return True

		return False


	df = prepare_eval_dataset(gold_agreement)

	if how == 'lift_only':
		f = lambda s: max_lift(s)
	elif how == 'pattern_only':
		f = lambda s: pattern_search(s)
	elif how == 'both_or':
		f = lambda s: max_lift(s) or pattern_search(s)
	elif how == 'both_and':
		f = lambda s: max_lift(s) and pattern_search(s)

	df['pred'] = df.sentence.apply(f)

	print(df.label.value_counts())
	print(precision_recall_fscore_support(df['label'], df['pred'], average='binary'))


if __name__ == "__main__":
	#pretrain_BERT(model='bert-base-uncased', use_cuda=True)
	rule_based(gold_agreement='weak', how='both_and')
	#eval_BERT(sciclops_dir + 'models/fine-tuned-bert-classifier', gold_agreement='weak')
	#pred_BERT(sciclops_dir + 'models/tuned-bert-classifier', claimKG=True)
