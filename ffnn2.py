import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import torch.nn.functional as F

unk = '<UNK>'

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, h, output_dim, max_length):
        super(FeedForwardNN, self).__init__()
        self.hidden_dim = h
        self.embedding = nn.Embedding(input_dim, 128)  # Embedding dimension set to 128
        self.fc1 = nn.Linear(128 * max_length, h)  # Adjust input size to match flattened embedding size
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(h, output_dim)  # Fully connected layer without flattening

        self.loss = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multiclass classification

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_sequence):
        # Embedding lookup
        embedded = self.embedding(input_sequence)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten the embedding

        # Fully connected layers
        hidden = F.relu(self.fc1(embedded))
        hidden = self.dropout1(hidden)
        output = self.fc2(hidden)

        return output

# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index, max_length=50):
    vectorized_data = []
    for document, y in data:
        vector = [word2index.get(word, word2index[unk]) for word in document]
        if len(vector) < max_length:
            vector += [0] * (max_length - len(vector))  # Pad shorter sequences
        else:
            vector = vector[:max_length]  # Truncate longer sequences
        vector = torch.tensor(vector)
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in test:
        tst.append((elt["text"].split(), int(elt["stars"]-1)))

    return tra, val, tst

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)  # Load train, val, test data
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    max_length = 50  # Define the maximum length for input sequences

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index, max_length)
    valid_data = convert_to_vector_representation(valid_data, word2index, max_length)
    test_data = convert_to_vector_representation(test_data, word2index, max_length)

    # Modify the output dimension to 5 classes (for rating predictions)
    model = FeedForwardNN(input_dim=len(vocab), h=args.hidden_dim, output_dim=5, max_length=max_length)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    print("========== Training for {} epochs ==========".format(args.epochs))
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 3
    for epoch in range(args.epochs):
        model.train()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")
        random.shuffle(train_data)
        minibatch_size = 8
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            batch_loss = 0
            for example_index in range(minibatch_size):
                input_sequence, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_sequence.unsqueeze(0))
                predicted_label = torch.argmax(predicted_vector, dim=1)  # Use argmax for class prediction
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                batch_loss += example_loss
            loss = batch_loss / minibatch_size
            loss.backward()
            optimizer.step()

        # Modified print to match the desired format
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{N}/{N} [============================>.] - loss: {loss.item():.4f} - acc: {correct / total:.4f}")
        print(f"Training completed for epoch {epoch + 1}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        print(f"Validation started for epoch {epoch + 1}")
        minibatch_size = 32
        N = len(valid_data)
        val_loss = 0
        with torch.no_grad():
            for minibatch_index in tqdm(range(N // minibatch_size)):
                for example_index in range(minibatch_size):
                    input_sequence, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_sequence.unsqueeze(0))
                    predicted_label = torch.argmax(predicted_vector, dim=1)  # Use argmax for class prediction
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                    val_loss += example_loss
        avg_val_loss = val_loss / (N // minibatch_size * minibatch_size)

        # Modified print to match the desired format
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{N}/{N} [============================>.] - val_loss: {avg_val_loss.item():.4f} - val_acc: {correct / total:.4f}")
        print(f"Validation completed for epoch {epoch + 1}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Early stopping check
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            print("Validation accuracy stagnated or dropped. Consider stopping training.")
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered after epoch {}".format(epoch + 1))
            break


    # Test the model
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    print("========== Testing ==========")
    minibatch_size = 32
    N = len(test_data)
    with torch.no_grad():
        for minibatch_index in tqdm(range(N // minibatch_size)):
            for example_index in range(minibatch_size):
                input_sequence, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_sequence.unsqueeze(0))  # Get the logits for the classes
                predicted_label = torch.argmax(predicted_vector, dim=1)  # Get the predicted class
                correct += int(predicted_label.item() == gold_label)  # Compare predicted and true class
                total += 1
    print("Test accuracy: {}".format(correct / total))
    print("Testing time: {:.4f}".format(time.time() - start_time))

