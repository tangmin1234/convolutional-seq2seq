# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import codecs
import regex
import itertools
import random

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source_train', 'corpora/train.tags.de-en.de', "source sequence for training")
flags.DEFINE_string('target_train', 'corpora/train.tags.de-en.en', "target sequence for training")
flags.DEFINE_string('source_test', 'corpora/IWSLT16.TED.tst2014.de-en.de.xml', "source sequence for test")
flags.DEFINE_string('target_test', 'corpora/IWSLT16.TED.tst2014.de-en.en.xml', "target sequence for test")
flags.DEFINE_integer('min_cnt', 20, "")
flags.DEFINE_integer('maxlen', 10, "")

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() \
                                if int(line.split()[1])>=FLAGS.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() \
                                if int(line.split()[1])>=FLAGS.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <=FLAGS.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), FLAGS.maxlen], np.int32)
    Y = np.zeros([len(y_list), FLAGS.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, FLAGS.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, FLAGS.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(FLAGS.source_train, 'r', 'utf-8').read().split("\n") \
                                                            if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(FLAGS.target_train, 'r', 'utf-8').read().split("\n") \
                                                            if line and line[0] != "<"]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y
    
def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(FLAGS.source_test, 'r', 'utf-8').read().split("\n") \
                                    if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(FLAGS.target_test, 'r', 'utf-8').read().split("\n") \
                                    if line and line[:4] == "<seg"]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // FLAGS.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=FLAGS.batch_size, 
                                capacity=FLAGS.batch_size*64,   
                                min_after_dequeue=FLAGS.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

def get_batch(num_epoch=None, shuffle=False):
    X, Y = load_train_data()
    num_batch = len(X) // FLAGS.batch_size
    valid_idxs = range(len(X))
    idx = itertools.chain.from_iterable(random.sample(valid_idxs, len(valid_idxs))\
                                        if shuffle else valid_idxs for _ in range(num_epoch))
    for i in range(num_batch * num_epoch):
        batch = {}
        batch_idxs = tuple(itertools.islice(idx, FLAGS.batch_size))
        batch['X'] = list(map(X.__getitem__, batch_idxs))
        batch['Y'] = list(map(Y.__getitem__, batch_idxs))
        yield batch

def get_batch_for_test(num_epoch=1, shuffle=False):
    X, source, target = load_test_data()
    num_batch = min(len(X) // FLAGS.batch_size, 10)
    valid_idxs = range(len(X))
    idx = itertools.chain.from_iterable(random.sample(valid_idxs, len(valid_idxs))\
                                        if shuffle else valid_idxs for _ in range(num_epoch))
    for i in range(num_epoch * num_batch):
        batch = {}
        batch_idxs = tuple(itertools.islice(idx, FLAGS.batch_size))
        batch['X'] = list(map(X.__getitem__, batch_idxs))
        batch['source'] = list(map(source.__getitem__, batch_idxs))
        batch['target'] = list(map(target.__getitem__, batch_idxs))
        yield batch

if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()
    batch = get_batch(num_epoch=1)
    batch = next(batch)