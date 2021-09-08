# models.py

from re import I
from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter
import math
class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self, sentence: List[str]):
        string_representation=type(self)
        if string_representation==UnigramFeatureExtractor:
            get_index= lambda word:self.feature_vector.index_of(f'Unigram:{word}')
            return [get_index(word) for word in sentence if get_index(word)!=-1]
        elif string_representation==BigramFeatureExtractor:
            get_index=lambda word:self.feature_vector.index_of(f'Bigram:{word}')
            return [get_index(bigram) for bigram in sentence if get_index(bigram)!=-1]
        else:
            get_index= lambda word:self.feature_vector.index_of(f'Better:{word}')
            return [get_index(word) for word in sentence if get_index(word)!=-1]

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
        c=Counter()
        c.update(sentence)
        return c


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self,train_exs: List[SentimentExample], indexer: Indexer):
        self.feature_vector=indexer
        self.stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        for sentiment_train in train_exs:
            filtered_sentence=[word.lower() for word in sentiment_train.words if not word.lower() in self.stop_words]
            for words in filtered_sentence:
                self.feature_vector.add_and_get_index(f'Unigram:{words}')
        self.weights=np.zeros(len(self.feature_vector))


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, train_exs: List[SentimentExample], indexer: Indexer):
        self.feature_vector=indexer
        self.stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        for sentiment_train in train_exs:
            filtered_sentence=[word.lower() for word in sentiment_train.words if not word.lower() in self.stop_words]
            filtered_sentence=[f'{filtered_sentence[i-1]}|{filtered_sentence[i]}' for i in range(1,len(filtered_sentence))]
            for words in filtered_sentence:
                self.feature_vector.add_and_get_index(f'Bigram:{words}')
        self.weights=np.zeros(len(self.feature_vector))
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False):
        filtered_sentence=[word.lower() for word in sentence if not word.lower() in self.stop_words]
        filtered_sentence=[f'{filtered_sentence[i-1]}|{filtered_sentence[i]}' for i in range(1,len(filtered_sentence))]
        return super().extract_features(filtered_sentence)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    #discard rare words
    def __init__(self, train_exs:List[SentimentExample],indexer: Indexer):
        self.feature_vector=indexer
        self.stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        c=Counter()
        for sentiment_train in train_exs:
            filtered_sentence=[word.lower() for word in sentiment_train.words if not word.lower() in self.stop_words]
            c.update(self.extract_features(filtered_sentence))
            for words in c:
                if c[words]>1:
                    self.feature_vector.add_and_get_index(f'Better:{words}')
        self.weights=np.zeros(len(self.feature_vector))


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
    def __init__(self,weights: np.ndarray,  feat_extractor: FeatureExtractor):
        self.weights=weights
        self.feat_extractor=feat_extractor
    def predict(self, sentence: List[str])->int:
        #create feature vector
        words = self.feat_extractor.extract_features(sentence)
        feature_vector=self.feat_extractor.feature_vector
        feature_vector_sentence=np.zeros(len(feature_vector))
        for indexes in [self.feat_extractor.get_indexer(words)]:
            feature_vector_sentence[indexes]=1
        return np.dot(self.weights,feature_vector_sentence)>0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights:np.ndarray, feat_extractor:FeatureExtractor):
        self.weights=weights
        self.feat_extractor=feat_extractor
    def predict(self,sentence:List[str])->int:
        words = self.feat_extractor.extract_features(sentence)
        feature_vector=self.feat_extractor.feature_vector
        feature_vector_sentence=np.zeros(len(feature_vector))
        for indexes in [self.feat_extractor.get_indexer(words)]:
            feature_vector_sentence[indexes]=1
        sigmoid=lambda z : np.exp(z)/(1+np.exp(z))
        predict=lambda x: sigmoid(np.dot(self.weights,x))
        return predict(feature_vector_sentence)>.5


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    feature_vector=feat_extractor.feature_vector
    weights=feat_extractor.weights
    #train the model
    epochs=10
    for i in range(0,epochs):
        random.shuffle(train_exs)
        learning_rate=1/(i+1)
        for sentence in train_exs:
            #create our f(x)
            feature_vector_sentence=np.zeros(len(feature_vector))
            words = feat_extractor.extract_features(sentence.words)
            for indexes in [feat_extractor.get_indexer(words)]:
                feature_vector_sentence[indexes]=1
            dot_product=np.dot(feature_vector_sentence,weights)
            if sentence.label==0 and dot_product>0:
                weights=np.subtract(weights, feature_vector_sentence*learning_rate)
            elif sentence.label==1 and dot_product<=0:
                weights=np.add(weights, feature_vector_sentence*learning_rate)
    return PerceptronClassifier(weights,feat_extractor)
    
def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    feature_vector=feat_extractor.feature_vector
    weights=feat_extractor.weights
    epochs =10
    sigmoid=lambda z : np.exp(z)/(1+np.exp(z))
    predict=lambda x: sigmoid(np.dot(weights,x))
    loss_func=lambda predicted: -math.log(predicted)
    gradient_func=lambda feature,predicted,actual:-np.multiply(feature,(1-predicted))if actual else np.multiply(feature,predicted)
    random.seed(0)
    for i in range(0,epochs):
        learning_rate=1/(1+i)
        random.shuffle(train_exs)
        for sentence in train_exs:
            feature_vector_sentence=np.zeros(len(feature_vector))
            words=feat_extractor.extract_features(sentence.words)
            for indexes in [feat_extractor.get_indexer(words)]:
                feature_vector_sentence[indexes]=1
            y_hat=predict(feature_vector_sentence)
            weights=np.subtract(weights,learning_rate*gradient_func(feature_vector_sentence,y_hat,sentence.label))
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
        feat_extractor = UnigramFeatureExtractor(train_exs, Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(train_exs,Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(train_exs,Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model