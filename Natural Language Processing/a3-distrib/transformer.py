# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
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

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        embedded = self.embedding(indices)  # [batch_size, seq_len, d_model]

        embedded = self.positional_encoding(embedded)

        attention_maps = []
        output = embedded
        for layer in self.layers:
            output, attn_map = layer(output)
            attention_maps.append(attn_map)

        logits = self.output_layer(output)  # [batch_size, seq_len, num_classes]
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]

        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.query_layer = nn.Linear(d_model, d_internal)
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)

        self.output_projection = nn.Linear(d_internal, d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

        self.scale = np.sqrt(d_internal)

    def forward(self, input_vecs):

        Q = self.query_layer(input_vecs)  # [batch_size, seq_len, d_internal]
        K = self.key_layer(input_vecs)  # [batch_size, seq_len, d_internal]
        V = self.value_layer(input_vecs)  # [batch_size, seq_len, d_internal]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = self.output_projection(attention_output)

        output = attention_output + input_vecs 
        output = self.feedforward(output) + output

        return output, attention_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
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
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):
    # Initialize the model
    vocab_size = 27  # 26 letters + space
    num_positions = 20
    d_model = 128
    d_internal = 64
    num_classes = 3
    num_layers = 1

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.NLLLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        random.shuffle(train)

        for example in train:
            optimizer.zero_grad()

            # Forward pass
            log_probs, _ = model(example.input_tensor.unsqueeze(0))

            loss = loss_fn(log_probs.view(-1, num_classes), example.output_tensor.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train)}')

    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))