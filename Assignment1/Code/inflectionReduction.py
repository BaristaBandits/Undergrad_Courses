from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

class InflectionReduction:

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = []

		# Fill in code here
		stemmer = PorterStemmer()

		for sentence in text:
			stemmed_sentence = []
			for token in sentence:
				stemmed_sentence.append(stemmer.stem(token))
			reducedText.append(stemmed_sentence)

		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = []
		# Fill in code here
		lemmatizer = WordNetLemmatizer()

		for sentence in text:
			lemmatized_sentence = []
			for token in sentence:
				lemmatized_sentence.append(lemmatizer.lemmatize(token))
			reducedText.append(lemmatized_sentence)

		return reducedText



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None

		# Fill in code here
		reducedText = self.wordnetLemmatizer(text)

		return reducedText
