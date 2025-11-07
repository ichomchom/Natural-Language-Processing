# factcheck.py

from nltk.tokenize import sent_tokenize
import torch
from typing import List
import numpy as np
import spacy
import gc
import string
import re
import nltk
nltk.download('punkt')


class FactExample(object):
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


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda')
                          for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.
        probs = torch.softmax(logits, dim=1)[0]

        entailment_score = probs[0].item()

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something
        return entailment_score


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


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        table = str.maketrans('', '', string.punctuation)
        fact_list = fact.lower().translate(table).split()
        passage_text = " ".join([p["text"] for p in passages]).lower()
        passage_set = set(passage_text.translate(table).split())
        match_count = sum(1 for word in fact_list if word in passage_set)
        recall = match_count / len(fact_list) if fact_list else 0.0
        threshold = 0.80
        return "S" if recall >= threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.nlp = spacy.load("en_core_web_sm")

    def compute_word_overlap(self, fact: str, sentence: str) -> float:
        table = str.maketrans('', '', string.punctuation)
        fact_words = set(fact.lower().translate(table).split())
        sentence_words = set(sentence.lower().translate(table).split())
        if not fact_words:
            return 0.0
        match_count = sum(1 for word in fact_words if word in sentence_words)
        return match_count / len(fact_words)

    def predict(self, fact: str, passages: List[dict]) -> str:
        overlap_threshold = 0.1
        single_scores = []
        pair_scores = []
        triplet_scores = []

        for passage in passages:
            sentence_list = []
            passage_text = passage["text"]

            sentence_list = sent_tokenize(passage_text)

            if not sentence_list:
                re_sentences = re.findall(
                    r'<s>(.*?)</s>', passage_text, flags=re.DOTALL)
                if re_sentences:
                    sentence_list = re_sentences
            cleaned_sentences = []
            for sentence in sentence_list:
                sentence = sentence.strip()

                if len(sentence.split()) < 1:
                    continue
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                cleaned_sentences.append(sentence)

            for sentence in cleaned_sentences:
                overlap = self.compute_word_overlap(fact, sentence)
                if overlap < overlap_threshold:
                    continue
                entailment_score = self.ent_model.check_entailment(
                    sentence, fact)
                single_scores.append(entailment_score)

            for i in range(len(cleaned_sentences) - 1):
                sent_pair = cleaned_sentences[i] + ' ' + cleaned_sentences[i+1]
                overlap = self.compute_word_overlap(fact, sent_pair)
                if overlap < overlap_threshold:
                    continue
                entailment_score = self.ent_model.check_entailment(
                    sent_pair, fact)
                pair_scores.append(entailment_score)

        all_scores = single_scores + pair_scores + triplet_scores

        if not all_scores:
            final_score = 0.0
        else:
            final_score = max(all_scores)

        threshold = 0.65
        return "S" if final_score >= threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
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
            ignore_dep = ['punct', 'ROOT', 'root', 'det',
                          'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
