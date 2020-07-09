import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_data(data_file, train_size=0.7):
    '''
    Load data from the dataset and split into training and validation set

    :param data_file: Location of the dataset
    :param train_size: Size of the training set in percentage
    :return: training data and validation data
    '''
    f = open(data_file)
    content = f.read()

    data = []
    for line in content.split('\n'):
        word = list(line.split(' ')[0])
        pronunciation = line.split(' ')[1:]
        data.append([word, pronunciation])
    
    np.random.shuffle(data)
    total_samples = len(data)
    training_samples = round(total_samples * train_size)
    train_data = data[:training_samples]
    val_data = data[training_samples:]
    return train_data, val_data

def create_vocab(data):
    '''
    Create char to index mapping

    :param data: List of [words, pronunciation]
    :return: Vocabulary of word chars and vocabulary of pronunications
    '''
    char_vocab, pronunce_vocab = {}, {}
    char_vocab['<unk>'] = 0
    char_vocab['<pad>'] = 1
    char_vocab['<sos>']= 2
    char_vocab['<eos>'] = 3
    pronunce_vocab['<unk>'] = 0
    pronunce_vocab['<pad>'] = 1
    pronunce_vocab['<sos>'] = 2
    pronunce_vocab['<eos>'] = 3
    print('Creating Vocabularies')
    for ex in tqdm(data):
        word, pronunciation = ex
        for char in word:
            if char not in char_vocab:
                char_vocab[char] = len(char_vocab)
        for i in pronunciation:
            if i not in pronunce_vocab:
                pronunce_vocab[i] = len(pronunce_vocab)
    return char_vocab, pronunce_vocab

class G2PDataset(Dataset):
    '''PyTorch Dataset object fot G2PData'''
    def __init__(self, data, char_vocab, pronunce_vocab, max_len = 20):
        self.data = data
        self.char_vocab = char_vocab
        self.pronunce_vocab = pronunce_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode_words(self, word):
        word = word[:self.max_len - 2]
        word = [self.char_vocab['<sos>']] + [self.char_vocab.get(char, self.char_vocab['<unk>']) for char in word] + [self.char_vocab['<eos>']]
        padded_word = word + [self.char_vocab['<pad>']] * (self.max_len - len(word))
        padded_word = torch.tensor(padded_word)
        return padded_word

    def encode_pronunciation(self, pronunciation):
        pronunciation = pronunciation[:self.max_len - 2]
        pronunciation = [self.pronunce_vocab['<sos>']] + [self.pronunce_vocab.get(i, self.pronunce_vocab['<unk>']) for i in pronunciation] + [self.pronunce_vocab['<eos>']]
        padded_pronunciation = pronunciation + [self.pronunce_vocab['<pad>']] * (self.max_len - len(pronunciation))
        padded_pronunciation = torch.tensor(padded_pronunciation)
        return padded_pronunciation

    def __getitem__(self, idx):
        word, pronunciation = self.data[idx]
        word = self.encode_words(word)
        pronunciation = self.encode_pronunciation(pronunciation)
        return word, pronunciation

        




