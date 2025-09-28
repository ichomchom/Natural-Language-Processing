# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk
from nltk.metrics import edit_distance


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(nn.Module, SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, word_embeddings: WordEmbeddings):
        super(NeuralSentimentClassifier, self).__init__()
        hidden_size = 700
        self.word_embeddings = word_embeddings
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(
            frozen=False, padding_idx=0)
        self.hidden_layer = nn.Linear(
            word_embeddings.get_embedding_length(), hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedding = self.embedding_layer(x)
        averaged_embedding = torch.mean(embedding, dim=1)
        hidden = self.hidden_layer(averaged_embedding)
        activated = self.relu(hidden)
        output = self.output_layer(activated)
        log_probs = self.log_softmax(output)
        return log_probs

    def convert_words_to_indices(self, ex_words: List[str], has_typos: bool) -> List[int]:
        indices = []
        for word in ex_words:
            if self.word_embeddings.word_indexer.index_of(word) != -1:
                indices.append(
                    self.word_embeddings.word_indexer.index_of(word))
            else:
                if has_typos:
                    corrected_word = self.spelling_correction(word)
                    indices.append(
                        self.word_embeddings.word_indexer.index_of(corrected_word))
                else:
                    indices.append(
                        self.word_embeddings.word_indexer.index_of("UNK"))
        return indices

    def pad_sequences(self, sequences, max_length, pad_value=0):
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                seq = seq + [pad_value] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            padded.append(seq)
        return padded

    def spelling_correction(self, word: str) -> str:
        if self.word_embeddings.word_indexer.index_of(word) != -1:
            return word
        else:
            min_distance = float('inf')
            corrected_word = ["UNK", "PAD"]
            count = 0
            for known_word in self.word_embeddings.word_indexer.objs_to_ints.keys():
                if known_word in ["UNK", "PAD"]:
                    continue
                count += 1
                if count > 6000:
                    break
                distance = edit_distance(word, known_word)
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = known_word
            return corrected_word

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.eval()
        with torch.no_grad():
            indices = self.convert_words_to_indices(ex_words, has_typos)
            list_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            log_probs = self.forward(list_tensor)
            prediction = torch.argmax(log_probs).item()
            return prediction


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    model = NeuralSentimentClassifier(word_embeddings)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 64

    for epoch in range(args.num_epochs):
        random.shuffle(train_exs)
        epoch_total_loss = 0
        number_of_batches = 0
        for i in range(0, len(train_exs), batch_size):
            batch = train_exs[i:i+batch_size]
            batch_words = [ex.words for ex in batch]
            batch_labels = [ex.label for ex in batch]
            indices_batch = [model.convert_words_to_indices(
                words, has_typos=False) for words in batch_words]

            max_length = max(len(seq) for seq in indices_batch)
            padded_batch = model.pad_sequences(indices_batch, max_length)
            input_tensor = torch.tensor(padded_batch, dtype=torch.long)
            labels = torch.tensor(batch_labels, dtype=torch.long)

            optimizer.zero_grad()
            log_probs = model(input_tensor)
            loss = loss_function(log_probs, labels)

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            number_of_batches += 1

        if epoch % 5 == 0:
            avg_loss = epoch_total_loss / number_of_batches
            print(f"Epoch {epoch}, Loss: {avg_loss}")

    return model
