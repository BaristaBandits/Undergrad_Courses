from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize

nltk.download('punkt')



class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		# Split sentences on punctuation followed by whitespace
		sentences = re.split(r'(?<=[.!?])\s+', text)

		# Remove empty strings and strip whitespace
		segmentedText = [s.strip() for s in sentences if s.strip()]


		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# Use NLTK's pretrained Punkt tokenizer
		segmentedText = sent_tokenize(text)

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		doc = self.nlp(text)

		# Extract sentences detected by spaCy
		segmentedText = [sent.text.strip() for sent in doc.sents]

		return segmentedText
