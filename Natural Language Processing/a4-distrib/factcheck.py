# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer
from itertools import tee
import math
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification


nltk.download('punkt_tab')
nltk.download('wordnet')

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # move to gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        with torch.no_grad():
            # tokenize and get logits
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits

        # logits -> probs
        entailment_prob = torch.softmax(logits, dim=-1)[0][0].item() 
        del inputs, outputs, logits  # clean up
        gc.collect()
        return entailment_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def __init__(self, threshold=0.075, use_tfidf=True):
        self.threshold = threshold
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_tfidf = use_tfidf
        self.document_frequencies = defaultdict(int)

    def preprocess(self, text):
        # tokenize, stop words, and bigrams
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        bigrams = self.get_ngrams(filtered_tokens, n=2)
        return filtered_tokens + bigrams

    def get_ngrams(self, tokens, n=2):
        tokens, tokens_copy = tee(tokens)
        return [" ".join(ngram) for ngram in zip(*[tokens_copy] * n)]

    def compute_document_frequencies(self, all_passages):
        for tokens in all_passages:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.document_frequencies[term] += 1

    def tfidf_vectorize(self, tokens, vocabulary, total_docs):
        token_counts = Counter(tokens)
        vector = []
        for word in vocabulary:
            tf = token_counts.get(word, 0)
            idf = math.log((total_docs + 1) / (1 + self.document_frequencies[word])) + 1
            vector.append(tf * idf)
        return np.array(vector)

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self.preprocess(fact)
        passage_tokens_list = [self.preprocess(passage["text"]) for passage in passages]
        all_tokens = set(fact_tokens).union(*passage_tokens_list)
        
        if self.use_tfidf:
            self.compute_document_frequencies(passage_tokens_list + [fact_tokens])

        vocabulary = list(all_tokens)
        fact_vector = self.tfidf_vectorize(fact_tokens, vocabulary, total_docs=len(passage_tokens_list) + 1)

        max_similarity = 0
        for passage in passages:
            # passage -> sentences
            sentences = sent_tokenize(passage["text"])
            for sentence in sentences:
                sentence_tokens = self.preprocess(sentence)
                
                # check for overlap and skip stuff that isn't really relevant
                overlap_ratio = len(set(fact_tokens).intersection(sentence_tokens)) / len(set(fact_tokens))
                if overlap_ratio < 0.5:
                    continue

                sentence_vector = self.tfidf_vectorize(sentence_tokens, vocabulary, total_docs=len(passage_tokens_list) + 1)
                similarity = self.cosine_similarity(fact_vector, sentence_vector)

                # max similarity
                max_similarity = max(max_similarity, similarity)
        
        return "S" if max_similarity >= self.threshold else "NS"

class EntailmentFactChecker(object):
    def __init__(self, ent_model: EntailmentModel, entailment_threshold=0.75):
        self.ent_model = ent_model
        self.entailment_threshold = entailment_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        max_entailment_score = 0

        for passage in passages:
            sentences = sent_tokenize(passage["text"])
            for sentence in sentences:
                entailment_score = self.ent_model.check_entailment(premise=sentence, hypothesis=fact)
                max_entailment_score = max(max_entailment_score, entailment_score)

        return "S" if max_entailment_score >= self.entailment_threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

