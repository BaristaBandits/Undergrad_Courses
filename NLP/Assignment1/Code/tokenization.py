from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import re
import spacy
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Fill in code here
		for sentence in text:
			tokens = re.findall(r"\w+|[^\w\s]", sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Fill in code here
		tokenizer = TreebankWordTokenizer()

		for sentence in text:
			tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		# Fill in code here

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		

		# Fill in code here
		tokenizedText = []

		# Fill in code here
		nlp = spacy.load("en_core_web_sm")

		for sentence in text:
			doc = nlp(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)

		return tokenizedText
