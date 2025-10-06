# models.py

import numpy as np
import torch.nn as nn
import random
import torch
import torch.optim as optim


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, num_positions)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(d_model, d_internal) for i in range(num_layers)])
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x = self.emb_layer(indices)
        x = self.position_encoding(x)
        attn_maps = []

        for transformer_layer in self.transformer_layers:
            x1, att_weights = transformer_layer(x)
            attn_maps.append(att_weights)
            x = x1

        x = self.linear(x)
        log_softmax = nn.functional.log_softmax(x, dim=-1)

        return log_softmax, attn_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_model)
        self.seq = nn.Sequential(
            nn.Linear(d_model,  d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, input_vecs):
        queries = self.query(input_vecs)
        keys = self.key(input_vecs)
        values = self.value(input_vecs)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_attn_scores = attn_scores / \
            torch.sqrt(torch.tensor(queries.shape[-1], dtype=torch.float))

        mask = torch.triu(torch.ones_like(
            scaled_attn_scores), diagonal=1).bool()
        scaled_attn_scores = scaled_attn_scores.masked_fill(
            mask, float('-inf'))

        attn_weights = torch.softmax(scaled_attn_scores, dim=-1)
        attention = torch.matmul(attn_weights, values)

        x = input_vecs + attention
        ff_out = self.seq(x)
        x = x + ff_out

        return x, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(
            range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        self.model.eval()
        if len(context) >= 20:
            context = context[-(20-1):]
        text = ' ' + context
        indices = [self.vocab_index.index_of(c) for c in text]
        indices_tensor = torch.LongTensor(indices)
        long_prob, _ = self.model.forward(indices_tensor)
        return long_prob[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        if len(context) >= 20:
            context = context[-(20-1):]
        total_log_prob = 0.0
        current_context = context
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(current_context)
            char_index = self.vocab_index.index_of(char)
            log_prob_this_char = log_probs[char_index]
            total_log_prob += log_prob_this_char
            current_context = current_context + char
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    model = Transformer(27, 20, 64, 32, 27, 1)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    chunk_size = 20

    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)

        chunks = [train_text[i:i + chunk_size]
                  for i in range(0, len(train_text), chunk_size)]

        random.shuffle(chunks)
        loss_fcn = nn.NLLLoss()

        for chunk in chunks:
            input_text = ' ' + chunk[:19]
            target_text = chunk[:20]
            indices = [vocab_index.index_of(c) for c in input_text]
            target_indices = [vocab_index.index_of(c) for c in target_text]
            log_probs, attn_maps = model.forward(torch.LongTensor(indices))

            loss = loss_fcn(log_probs, torch.LongTensor(target_indices))
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

    model.eval()
    return NeuralLanguageModel(model, vocab_index)
