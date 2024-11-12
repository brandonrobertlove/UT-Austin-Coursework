# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.nn.utils.rnn import pad_sequence
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


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, word_embeddings: WordEmbeddings, hidden_size: int, num_classes: int = 2, dropout_rate: float = 0.2):
        super(NeuralSentimentClassifier, self).__init__()
        self.word_embeddings = word_embeddings
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Define the feedforward layers
        self.network = nn.Sequential(
            nn.Linear(word_embeddings.get_embedding_length(), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_word_idxs: torch.Tensor):
        embedded = self.embedding_layer(batch_word_idxs)  # Shape: (batch_size, max_seq_len, embedding_dim)
        mask = (batch_word_idxs != 0).float().unsqueeze(-1)  # Mask for padding
        summed_embeddings = torch.sum(embedded * mask, dim=1)
        lengths = torch.sum(mask, dim=1)
        avg_embeddings = summed_embeddings / torch.clamp(lengths, min=1e-9)
        output = self.network(avg_embeddings)
        return output

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        if has_typos:
            # Correct spelling for each word in the input sentence
            corrected_words = [self.correct_spelling(word) for word in ex_words]
        else:
            # No typo correction
            corrected_words = ex_words

        # Convert words to their corresponding indices in the embedding layer
        word_indices = [self.word_embeddings.word_indexer.index_of(word) if self.word_embeddings.word_indexer.index_of(word) != -1 else self.word_embeddings.word_indexer.index_of("UNK") for word in corrected_words]
        
        # Prepare input tensor and make predictions
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)
        output = self.forward(word_indices_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        all_word_indices = [[self.word_embeddings.word_indexer.index_of(word) if self.word_embeddings.word_indexer.index_of(word) != -1 else self.word_embeddings.word_indexer.index_of("UNK") for word in ex_words] for ex_words in all_ex_words]
        padded_word_indices = pad_sequence([torch.tensor(w) for w in all_word_indices], batch_first=True, padding_value=0)
        output = self.forward(padded_word_indices)
        predictions = torch.argmax(output, dim=1).tolist()
        return predictions
    
    def correct_spelling(self, word: str) -> str:
        if self.word_embeddings.word_indexer.contains(word):
            return word

        # Compute edit distance for the closest word in the vocabulary
        closest_word = None
        min_distance = float('inf')
        for vocab_word in self.word_embeddings.word_indexer.objs_to_ints.keys():
            dist = edit_distance(word, vocab_word)
            if dist < min_distance and dist <= 2:  # Threshold to prevent distant matches
                min_distance = dist
                closest_word = vocab_word

        if closest_word is not None:
            return closest_word
        else:
            return "UNK"


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
    model = NeuralSentimentClassifier(word_embeddings, hidden_size=args.hidden_size)
    optimizer = optim.Adam(model.network.parameters(), lr=args.lr)

    # Print embedding matrix size
    vocab_size = word_embeddings.get_embedding_length()
    print(f"Embedding matrix size: {vocab_size}")

    batch_size = args.batch_size

    for epoch in range(args.num_epochs):
        model.network.train()
        total_loss = 0.0

        random.shuffle(train_exs)

        for i in range(0, len(train_exs), batch_size):
            batch_exs = train_exs[i:i + batch_size]
            all_word_indices = [[word_embeddings.word_indexer.index_of(word) if word_embeddings.word_indexer.index_of(word) != -1 else word_embeddings.word_indexer.index_of("UNK") for word in ex.words] for ex in batch_exs]

            padded_word_indices = pad_sequence([torch.tensor(w) for w in all_word_indices], batch_first=True, padding_value=0)

            labels = torch.tensor([ex.label for ex in batch_exs], dtype=torch.long)

            # Forward pass
            output = model.forward(padded_word_indices)
            loss = model.loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_exs)}")

    return model