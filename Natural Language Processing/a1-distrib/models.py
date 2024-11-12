# models.py

from sentiment_data import *
from utils import *
import numpy as np
import string
import math

from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from typing import List, Dict

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "\n"
])

def normalize_features(features: Counter) -> Counter:
        max_count = max(features.values(), default=1)
        return Counter({k: v / max_count for k, v in features.items()})

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def remove_punctuation(self, text: List[str]) -> List[str]:
        # Create a translation table that maps each punctuation character to None
        translator = str.maketrans('', '', string.punctuation)
        # Remove punctuation from each word
        return [word.translate(translator) for word in text]
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False):
        features = Counter()

        sentence = [word.lower() for word in self.remove_punctuation(sentence) if word.lower() not in STOP_WORDS]
        for word in sentence:
            if add_to_indexer:
                self.indexer.add_and_get_index(word, add=True)
            index = self.indexer.index_of(word)
            if index != -1:
                features[index] += 1
        return normalize_features(features)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def remove_punctuation(self, text: List[str]) -> List[str]:
        translator = str.maketrans('', '', string.punctuation)
        return [word.translate(translator) for word in text]
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False):
        features = Counter()
        
        # Preprocess sentence
        sentence = [word.lower() for word in self.remove_punctuation(sentence) if word.lower() not in STOP_WORDS]
        
        # Generate bigrams
        bigrams = [(sentence[i], sentence[i + 1]) for i in range(len(sentence) - 1)]
        
        # Extract features for bigrams
        for bigram in bigrams:
            bigram_feature = ' '.join(bigram)
            if add_to_indexer:
                self.indexer.add_and_get_index(bigram_feature, add=True)
            index = self.indexer.index_of(bigram_feature)
            if index != -1:
                features[index] += 1
        
        return normalize_features(features)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, min_df: int = 5):
        self.indexer = indexer
        self.min_df = min_df
        self.documents = []
        self.tf = []
        self.idf = defaultdict(float)
        self.vocab = defaultdict(int)

    def get_indexer(self):
        return self.indexer

    def fit_transform(self, documents: List[List[str]]):
        self.documents = documents
        num_docs = len(documents)
        doc_count = defaultdict(int)
        
        # Calculate term frequencies (TF)
        self.tf = []
        for doc in documents:
            term_count = Counter(doc)
            total_terms = len(doc)
            tf = {term: count / total_terms for term, count in term_count.items()}
            self.tf.append(tf)
            for term in term_count:
                doc_count[term] += 1
                self.vocab[term] += 1
        
        # Calculate inverse document frequency (IDF)
        for term, count in doc_count.items():
            if count >= self.min_df:
                self.idf[term] = math.log(num_docs / (1 + count))

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        sentence = [word.lower() for word in sentence if word.lower() not in STOP_WORDS]
        tf = Counter(sentence)
        total_terms = len(sentence)
        tf = {term: count / total_terms for term, count in tf.items()}
        
        features = Counter()
        for term, tf_value in tf.items():
            if term in self.idf:
                tfidf_value = tf_value * self.idf[term]
                if add_to_indexer:
                    self.indexer.add_and_get_index(term, add=True)
                index = self.indexer.index_of(term)
                if index != -1:
                    features[index] = tfidf_value

        return normalize_features(features)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor: FeatureExtractor, learning_rate: float = 0.0001):
        self.feature_extractor = feature_extractor
        self.weights = defaultdict(float)
        self.indexer = feature_extractor.get_indexer()
        self.initial_learning_rate = learning_rate

    def train(self, train_exs: List[SentimentExample], num_epochs: int):
        self.epochs = num_epochs
        learning_rate = self.initial_learning_rate
        for epoch in range(self.epochs):
            print(epoch)
            learning_rate = self.initial_learning_rate*(max(0.94**(epoch), 0.6))
            for example in train_exs:
                features = self.feature_extractor.extract_features(example.words, add_to_indexer=True)
                prediction = self.predict(example.words)
                if prediction != example.label:
                    for feature, count in features.items():
                        self.weights[feature] += learning_rate * (example.label - prediction) * count
            print(self.weights)

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights.get(feature, 0) * count for feature, count in features.items())
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor: FeatureExtractor, learning_rate: float = 0.0001):
        self.feature_extractor = feature_extractor
        self.weights = defaultdict(float)
        self.indexer = feature_extractor.get_indexer()
        self.initial_learning_rate = learning_rate

    def train(self, train_exs: List[SentimentExample], num_epochs: int):
        self.epochs = num_epochs
        learning_rate = self.initial_learning_rate
        for epoch in range(self.epochs):
            print(epoch)
            learning_rate = self.initial_learning_rate*(max(0.94**(epoch), 0.6)) 
            for example in train_exs:
                features = self.feature_extractor.extract_features(example.words, add_to_indexer=True)
                score = sum(self.weights.get(feature, 0) * count for feature, count in features.items())
                prediction = self.predict(example.words)
                error = example.label - prediction
                for feature, count in features.items():
                    self.weights[feature] += learning_rate * error * count * (-1 + (score)/(1+score))

    def predict(self, sentence: List[str]):
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights.get(feature, 0) * count for feature, count in features.items())
        sig_score = 1 / (1 + np.exp(score))
        if sig_score > 0.5:
            return 1
        else:
            return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs: int) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    model = PerceptronClassifier(feat_extractor)
    model.train(train_exs, epochs)
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs: int) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    model = LogisticRegressionClassifier(feat_extractor)
    model.train(train_exs, epochs)
    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
        num_epochs = 100
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
        num_epochs = 75
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        indexer = Indexer()
        better_fe = BetterFeatureExtractor(indexer)
        
        # Prepare the documents for TF-IDF vectorization
        documents = [example.words for example in train_exs]
        better_fe.fit_transform(documents)
        
        feat_extractor = better_fe
        num_epochs = 20
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor, num_epochs)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, num_epochs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model