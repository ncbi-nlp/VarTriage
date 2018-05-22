import numpy as np
import re
import io
import itertools
from collections import Counter
from gensim.models import KeyedVectors, Word2Vec

def load_data_file(txt_filename, tgt_filename):
    txt = open(txt_filename, 'r')
    tgt = open(tgt_filename, 'r')
    X_txt = []
    Y = []
    for txt_line, tgt_line in zip(txt, tgt):
        X_txt.append(txt_line.strip())
        Y.append(tgt_line.strip())
    return X_txt, Y

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #return string.strip().lower()
    return string.strip()


def load_data_without_labels(pos_file):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(io.open(pos_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(io.open(neg_file, "r", encoding='latin-1').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples
    x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]
    # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    #return [x_text, y]
    return x_text

def load_data_and_labels(pos_file, neg_file):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(io.open(pos_file, "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(io.open(neg_file, "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_splited(pos_file, neg_file):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(io.open(pos_file, "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(io.open(neg_file, "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    #x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = np.array([[word for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(pos_file, neg_file):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_splited(pos_file, neg_file)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_data_eval(pos_file, neg_file, sequence_length):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(pos_file, neg_file)
    sentences_padded = pad_sentences_eval(sentences, sequence_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_pmids(pmid_file_addr):
    # Load data from files
    pmid_file = list(io.open(pmid_file_addr, "r").readlines())
    pmids = [s.strip() for s in pmid_file]
    pmids_list=list(pmids)
    return pmids_list

def pad_sentences_eval(sentences, sequence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def load_word2vec(word2vec_file_addr) :
    # Word2Vec loading (Glove to Word2vec: https://radimrehurek.com/gensim/scripts/glove2word2vec.html)
    print('Indexing word vectors.')
    embeddings_index = {}
    gensim_model = KeyedVectors.load_word2vec_format(word2vec_file_addr, binary=True)
    #gensim_model = KeyedVectors.load_word2vec_format(word2vec_file_addr, binary=False)
    word_counter = 0
    for word_in_w2v in gensim_model.vocab:
        # values = line.split()
        # word = word_in_w2v
        # coefs = np.asarray(values[1:], dtype='float32')
        coefs = np.asarray(gensim_model.word_vec(word_in_w2v), dtype='float32')
        embeddings_index[word_in_w2v] = coefs
    print('Found %s word vectors of word2vec' % len(gensim_model.vocab))
    test_word = "human"
    embedding_dim = len(embeddings_index[test_word])

    return embeddings_index, embedding_dim

