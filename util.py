# Add your import statements here
# Add your import statements here
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer



# Add any utility functions here

def compute_vocabulary_size(tokenized_docs):
    vocab = set()

    for doc in tokenized_docs:
        for sent in doc:
            for token in sent:
                vocab.add(token.lower())

    return len(vocab)



# ---------------------------
# Part 3: Stemming vs Lemmatization
# ---------------------------

def stem_documents(tokenized_docs):
    stemmer = PorterStemmer()
    stemmed_docs = []

    for doc in tokenized_docs:
        stem_doc = []
        for sent in doc:
            stem_doc.append([stemmer.stem(token) for token in sent])
        stemmed_docs.append(stem_doc)

    return stemmed_docs


def lemmatize_documents(tokenized_docs):
    lemmatizer = WordNetLemmatizer()
    lemma_docs = []

    for doc in tokenized_docs:
        lemma_doc = []
        for sent in doc:
            lemma_doc.append([lemmatizer.lemmatize(token) for token in sent])
        lemma_docs.append(lemma_doc)

    return lemma_docs


def compare_stemming_lemmatization(tokenized_docs):

    stemmed_docs = stem_documents(tokenized_docs)
    lemmatized_docs = lemmatize_documents(tokenized_docs)

    stem_vocab_size = compute_vocabulary_size(stemmed_docs)
    lemma_vocab_size = compute_vocabulary_size(lemmatized_docs)

    print("Vocabulary size after stemming:", stem_vocab_size)
    print("Vocabulary size after lemmatization:", lemma_vocab_size)

    print("\nExamples of possible over-stemming:")

    examples = ["studies", "analysis", "university", "connection"]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for word in examples:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        print(f"{word} -> stem: {stem} | lemma: {lemma}")



# ---------------------------
# Part 4: Data-driven Stopwords
# ---------------------------

def generate_data_driven_stopwords(tokenized_docs, k=50):

    all_tokens = []

    for doc in tokenized_docs:
        for sent in doc:
            for token in sent:
                all_tokens.append(token.lower())

    freq = Counter(all_tokens)

    stopword_list = set([word for word, count in freq.most_common(k)])

    return stopword_list


def compare_stopword_lists(tokenized_docs):

    nltk_list = set(stopwords.words("english"))
    data_list = generate_data_driven_stopwords(tokenized_docs)

    overlap = nltk_list.intersection(data_list)

    print("Number of NLTK stopwords:", len(nltk_list))
    print("Number of data-driven stopwords:", len(data_list))
    print("Overlap between lists:", len(overlap))

    print("\nWords only in data-driven list:")
    print(data_list - nltk_list)

    print("\nWords only in NLTK list:")
    print(nltk_list - data_list)




