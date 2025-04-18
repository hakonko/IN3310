import os
import numpy as np
import pandas as pd
import string
from nltk_tokenizer.tokenize import word_tokenize
from utils.coco import COCO


class Vocabulary:
    def __init__(self, size, save_file=None):
        self.words = []
        self.word2idx = {}
        self.word_frequencies = []
        self.size = size
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = {}
        for sentence in sentences:
            for w in word_tokenize(sentence.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1.0

        assert self.size-1 <= len(word_counts.keys())
        self.words.append('<pad>')
        self.word2idx['<pad>'] = 0
        self.word_frequencies.append(1.0)
        self.words.append('<start>')
        self.word2idx['<start>'] = 1
        self.word_frequencies.append(1.0)

        word_counts = sorted(list(word_counts.items()),
                             key=lambda x: x[1],
                             reverse=True)

        for idx in range(self.size-1):
            word, frequency = word_counts[idx]
            self.words.append(word)
            self.word2idx[word] = idx + 2
            self.word_frequencies.append(frequency)

        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = word_tokenize(sentence.lower())
        word_idxs = [self.word2idx[w] for w in words]
        return word_idxs

    def get_sentence(self, idxs):
        """ Translate a vector of indices into a sentence. """
        words = [self.words[i] for i in idxs]
        if words[-1] != '.':
            words.append('.')
        length = np.argmax(np.array(words) == '.') + 1
        words = words[:length]
        sentence = ''.join([' ' + w if not w.startswith("'") and w not in string.punctuation
                            else w for w in words]).strip()
        return sentence

    def save(self, save_file):
        """ Save the vocabulary to a file. """
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.size + 1)),  # + 1 for the <start> token
                             'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.word2idx = {self.words[i]: i for i in range(self.size)}
        self.word_frequencies = data['frequency'].values


def build_vocabulary(config, coco=None):
    """ Build the vocabulary from the training data and save it to a file. """
    if coco is None:
        coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)

    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)

    return vocabulary
