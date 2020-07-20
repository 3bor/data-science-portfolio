import random
import glob
import re
import numpy as np

from xml.etree import ElementTree
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif



#	Namespace for xml files
ns = {
	"OAI": "http://www.openarchives.org/OAI/2.0/", 
	"ARX": "http://arxiv.org/OAI/arXiv/"
}

def get_files(folder):
	"""
		get all files with .txt extension from a folder
		input: folder (string)
		output: list of files (strings)
	"""
	files = [f for f in glob.glob(folder + "*.txt")]
	return files


def get_records(file):
	"""
		get the records from an input file in xml format
		input: file (string)
		output: list of XMLrecords
	"""
	tree = ElementTree.parse(file)
	root = tree.getroot()
	records = tree.findall("OAI:ListRecords/OAI:record", ns)
	return records

	
def get_label(record):
	"""
		get the label of each record 
		input: XMLrecord 
		output: 0 (hep-ph) or 1 (hep-th)
	"""
	category = record.find('OAI:header/OAI:setSpec', ns).text
	if category == "physics:hep-ph": 
		label = 0
	elif category == "physics:hep-th":
		label = 1
	else:
		label = Null
	return label


def get_abstract(record):
	"""
		get abstract of each record
		input: XMLrecord 
		output: abstract (string)
	"""
	abstract = record.find("OAI:metadata/ARX:arXiv/ARX:abstract", ns).text
	return abstract


def parse_text(text, stemmer):
	"""
		Parse text by stemming words
		input: text (string), stemmer (SnowballStemmer())
		output: abstract with stemmed words
	"""
 	wordlist = re.sub("[^a-zA-Z_]", " ", text).lower().split()
	wordlist = [stemmer.stem(word) for word in wordlist]
	parsed = " ".join(wordlist)
	return parsed


def print_word_freq(trans_features_train, vectorizer):
	"""
		Print words and their frequencies
		input: trans_features_train, vectorizer
		output: printout of word frequencies
	"""
	wordfreq = np.sum(trans_features_train, axis=0)
	wordfreq = np.round(wordfreq, 4)
	wordfreq = [vectorizer.get_feature_names(), wordfreq.tolist()]
	wordfreq = map(list, zip(*wordfreq))
	wordfreq.sort(key = lambda x : x[1], reverse = True)
	for i in range(5):
		print(str(round(wordfreq[i][1],2))+" : "+wordfreq[i][0])
	print("(...)")
	for i in range(5,0,-1):
		print(str(round(wordfreq[-i][1],2))+" : "+wordfreq[-i][0])
