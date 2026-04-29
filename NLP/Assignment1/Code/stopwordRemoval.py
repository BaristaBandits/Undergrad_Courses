from util import *
from nltk.corpus import stopwords
# Add your import statements here




class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		#Fill in code here
		stop_words = set(stopwords.words("english"))

		for sentence in text:
			filtered_sentence = []
			for token in sentence:
				if token.lower() not in stop_words:
					filtered_sentence.append(token)
			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText




	
