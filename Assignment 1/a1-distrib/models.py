# models.py

import math
import random
from sentiment_data import *
from utils import *

from collections import Counter
from typing import List


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
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


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                if not self.indexer.contains(word):
                    continue
                idx = self.indexer.index_of(word)
            feats[idx] += 1
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()

        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                if not self.indexer.contains(bigram):
                    continue
                idx = self.indexer.index_of(bigram)
            feats[idx] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        # Bias feature
        bias_idx = self.indexer.add_and_get_index("BIAS", add=add_to_indexer)
        if bias_idx != -1:
            feats[bias_idx] += 1.0

        words = [t.lower() for t in sentence]
        # Unigrams
        for word in words:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                if not self.indexer.contains(word):
                    continue
                idx = self.indexer.index_of(word)
            feats[idx] += 1
        # Bigrams
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                if not self.indexer.contains(bigram):
                    continue
                idx = self.indexer.index_of(bigram)
            feats[idx] += 1
        # Negation
        neg_words = {"not", "n't", "no", "never"}
        if any(word in neg_words for word in words):
            negation_word = f"HAS_NEGATION_{words[0]}"
            idx = self.indexer.add_and_get_index(negation_word)
            if idx != -1:
                feats[idx] += 1.0
        # Punctuation
        punctuation_words = {".", ",", ":", ";", "?", "!"}
        if any(word in punctuation_words for word in words):
            punctuation_word = f"HAS_PUNCTUATION_{words[0]}"
            idx = self.indexer.add_and_get_index(punctuation_word)
            if idx != -1:
                feats[idx] += 1.0
        # All-caps
        if any(word.isupper() and len(word) > 1 for word in words):
            allcaps_word = f"HAS_ALLCAPS_{words[0]}"
            idx = self.indexer.add_and_get_index(allcaps_word)
            if idx != -1:
                feats[idx] += 1.0
        return feats


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

    def __init__(self, weights: List[float], feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(
            sentence, add_to_indexer=False)
        score = 0.0
        for idx, val in feats.items():
            if idx < len(self.weights):
                score += self.weights[idx] * val
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: List[float], feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(
            sentence, add_to_indexer=False)
        score = 0.0
        for idx, val in feats.items():
            if idx < len(self.weights):
                score += self.weights[idx] * val
        # Sigmoid
        prob = 1.0 / (1.0 + math.exp(-max(min(score, 20.0), -20.0)))
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Initialize weights
    indexer = feat_extractor.get_indexer()
    weights: List[float] = [0.0 for _ in range(len(indexer))]

    epochs = 5
    lr = 1.0
    for _ in range(epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(
                ex.words, add_to_indexer=True)
            if len(weights) < len(indexer):
                weights.extend([0.0] * (len(indexer) - len(weights)))
            # Compute score
            score = 0.0
            for idx, val in feats.items():
                score += weights[idx] * val
            pred = 1 if score >= 0 else 0
            y = ex.label
            if pred != y:
                # Update rule
                if y == 1:
                    for idx, val in feats.items():
                        weights[idx] += lr * val
                else:
                    for idx, val in feats.items():
                        weights[idx] -= lr * val

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights: List[float] = [0.0 for _ in range(len(indexer))]

    epochs = 5
    lr0 = 0.1
    l2 = 0.0
    for epoch in range(epochs):
        lr = lr0 / (1.0 + 0.1 * epoch)
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(
                ex.words, add_to_indexer=True)
            if len(weights) < len(indexer):
                weights.extend([0.0] * (len(indexer) - len(weights)))
            score = 0.0
            for idx, val in feats.items():
                score += weights[idx] * val
            score = max(min(score, 20.0), -20.0)
            p = 1.0 / (1.0 + math.exp(-score))
            y = ex.label
            err = (y - p)
            if l2 != 0.0:
                for i in range(len(weights)):
                    weights[i] *= (1.0 - lr * l2)
            for idx, val in feats.items():
                weights[idx] += lr * err * val

    return LogisticRegressionClassifier(weights, feat_extractor)


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
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception(
            "Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
