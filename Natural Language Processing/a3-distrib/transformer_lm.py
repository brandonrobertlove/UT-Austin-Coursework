import torch
import torch.nn as nn
import numpy as np
import json

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        input_size = x.shape[-2]
        if input_size > self.emb.num_embeddings:
            raise ValueError(f"Input sequence length ({input_size}) exceeds maximum positional encoding size ({self.emb.num_embeddings}).")
        indices_to_embed = torch.arange(0, input_size).type(torch.LongTensor).to(x.device)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, vocab_index, d_model=32, nhead=4, num_layers=2, dim_feedforward=512, max_seq_len=20):
        super(NeuralLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_index = vocab_index
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, batched=True)

        # Transformer encoder with causal mask
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        embedded = self.positional_encoding(embedded)

        seq_len = input_tensor.size(1)

        # Creating a causal mask for the transformer
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input_tensor.device)
        transformer_output = self.transformer(embedded.permute(1, 0, 2), mask=causal_mask)

        logits = self.output_layer(transformer_output.permute(1, 0, 2))
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def get_next_char_log_probs(self, context):
        if len(context) == 0:
            context = " "

        padded_context = context[-20:].rjust(20)

        context_indices = torch.LongTensor([self.vocab_index.index_of(c) for c in padded_context]).unsqueeze(0).to(next(self.parameters()).device)

        self.eval()
        with torch.no_grad():
            log_probs = self.forward(context_indices)[0, -1, :]
        return log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        for i in range(len(next_chars)):
            log_probs = self.get_next_char_log_probs(context)
            next_char_idx = self.vocab_index.index_of(next_chars[i])
            total_log_prob += log_probs[next_char_idx]
            context += next_chars[i]
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda") if torch.cuda.is_available() else print("cpu")
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, vocab_index).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Lowered the learning rate for stability
    loss_fn = nn.NLLLoss()

    train_seq_len = 20
    num_epochs = 15
    batch_size = 16

    train_chunks = [train_text[((i*train_seq_len)):((i+1)*train_seq_len)] for i in range(0, int(np.floor((len(train_text) - train_seq_len)/train_seq_len)))]
    #print(train_chunks)
    train_indices = [[vocab_index.index_of(c) for c in (' '+chunk)] for chunk in train_chunks]
    train_tensors = torch.LongTensor(train_indices).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(train_tensors), batch_size):
            batch = train_tensors[i:i+batch_size]

            input_batch = batch[:, :-1]
            target_batch = batch[:, 1:]


            optimizer.zero_grad()
            log_probs = model(input_batch)

            loss = loss_fn(log_probs.reshape(-1, vocab_size), target_batch.reshape(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_chunks)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Perplexity: {perplexity}")
        print_evaluation(dev_text, model, vocab_index, args.output_bundle_path)

    return model


def read_text(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_text = ""
    for line in open(file):
        all_text += line
    print("%i chars read in" % len(all_text))
    return all_text


def run_sanity_check(lm, vocab_index):
    """
    Runs two sanity checks: (1) The language model must return valid probabilities for a few contexts. This checks that
    your model can take sequences of different lengths and contexts of different lengths without crashing.
    (2) Your reported next character distribution must agree with get_log_prob_sequence
    :param lm: the trained LM
    :return: True if the output is sane, false otherwise
    """
    contexts = [" ", " a person ", " some person "]
    next_seqs = ["s", "sits", "stands"]
    sane = True
    for context in contexts:
        for next_seq in next_seqs:
            log_prob = lm.get_log_prob_sequence(next_seq, context)
            if log_prob > 0.0:
                sane = False
                print("ERROR: sanity checks failed, LM log probability %f is invalid" % (log_prob))
            log_prob_from_single_probs = 0.0
            for i in range(0, len(next_seq)):
                #print(repr(context + next_seq[0:i]))
                #print(repr(next_seq[i]))
                next_char_log_probs = lm.get_next_char_log_probs(context + next_seq[0:i])
                #print(repr(next_char_log_probs))
                log_prob_from_single_probs += next_char_log_probs[vocab_index.index_of(next_seq[i])]
            if abs(log_prob_from_single_probs - log_prob) > 1e-3:
                sane = False
                print("ERROR: sanity checks failed, LM prob from sequence and single characters disagree: %f %f" % (log_prob, log_prob_from_single_probs))
    return sane


def normalization_test(lm, vocab_index):
    """
    Tests that LM normalizes, checks multiple contexts and sums over everything in the vocabulary to make sure it
    sums to one
    :param lm:
    :param voc:
    :return:
    """
    contexts = [" ", " a person "]
    normalizes = True
    for context in contexts:
        total_prob_single = np.sum(np.exp(lm.get_next_char_log_probs(context)))
        if total_prob_single < 0.99 or total_prob_single > 1.01:
            normalizes = False
            print("Probabilities sum to %f not 1.0 for context %s" % (total_prob_single, context))
        total_prob_seq = 0.0
        for char_idx in range(0, len(vocab_index)):
            total_prob_seq += np.exp(lm.get_log_prob_sequence(vocab_index.get_object(char_idx), context))
        if total_prob_seq < 0.99 or total_prob_seq > 1.01:
            normalizes = False
            print("Probabilities sum to %f not 1.0 for context %s" % (total_prob_seq, context))
    return normalizes

def perplexity_range_check(perplexity):
    if perplexity < 3.5:
        print("ERROR: checks failed, the perplexity is too low. Please make sure you are using causal mask and make sure you are scoring the entire next_chars (instead of a single chunk) in get_log_prob_sequence")
        return False
    return True

def print_evaluation(text, lm, vocab_index, output_bundle_path):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    sane = run_sanity_check(lm, vocab_index)
    log_prob = float(lm.get_log_prob_sequence(text, ""))
    avg_log_prob = log_prob/len(text)
    perplexity = np.exp(-log_prob / len(text))
    normalizes = normalization_test(lm, vocab_index)
    range_check = perplexity_range_check(perplexity)
    data = {'sane': sane, 'normalizes': normalizes, 'range': range_check, 'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    with open(output_bundle_path, 'w') as outfile:
        json.dump(data, outfile)
    return data