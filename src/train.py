import os
from tqdm import tqdm
import shutil
import codecs
from nltk.translate.bleu_score import corpus_bleu

import tensorflow as tf
from model import ConvSeq2Seq
from trainer import Trainer
from graph_handler import GraphHandler
from inferencer import Inferencer
from data_load import get_batch, get_batch_for_test, load_de_vocab, load_en_vocab

def main(config):
    if config.is_debug:
        _config_debug(config)
    create_directory(config)
    if config.mode == 'train':
        train(config)
    elif config.mode == 'eval':
        eval(config)
    elif config.mode == 'test':
        test(config)
    else:
        raise ValueError("invalid model value which should be either 'train' or 'eval'.")

def _config_debug(config):
    config.hidden_dim = 20
    config.hidden_dim_2 = 20
    config.log_period = 2
    config.save_period = 5
    config.eval_period = 10
    config.num_enc_block_1 = 1
    config.num_enc_block_2 = 1
    config.num_dec_block_1 = 1
    config.num_dec_block_2 = 1
    config.load_step = 5

def train(config):
    model = ConvSeq2Seq(config)
    trainer = Trainer(config, model)
    graph_handler = GraphHandler(config)
    sess = tf.Session()
    graph_handler.initialize(sess)

    for i, batch in tqdm(enumerate(get_batch(num_epoch=config.num_epoch))):
        global_step = sess.run(model.global_step) + 1
        loss, acc, summary = trainer.run_step(sess, batch)
        print "global_step: %d,    loss: %f,     acc: %f" % (global_step, loss, acc)

        get_summary = global_step % config.log_period == 0
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        if global_step % config.save_period == 0:
            graph_handler.save_model(sess, global_step)

            if global_step % config.eval_period == 0:
                pass

    if global_step % config.save_period != 0:
        graph_handler.save_model(sess)

def eval(config):
    pass

def _config_test(config):
    config.load_model = True
    config.load_step = 0

def test(config):
    _config_test(config)

    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    model = ConvSeq2Seq(config)
    graph_handler = GraphHandler(config)
    inferencer = Inferencer(config, model)
    sess = tf.Session()
    graph_handler.initialize(sess)

    global_step = 0
    refs = []
    hypotheses = []
    with codecs.open(os.path.join(config.eval_dir, config.model_name), "w", "utf-8") as fout:
        for i, batch in tqdm(enumerate(get_batch_for_test())):
            preds = inferencer.run(sess, batch)
            sources = batch['source']
            targets = batch['target']
            for source, target, pred in zip(sources, targets, preds):
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                fout.write("- source: " + source +"\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()

                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    refs.append([ref])
                    hypotheses.append(hypothesis)

        score = corpus_bleu(refs, hypotheses)
        fout.write("Bleu Score = " + str(100*score))

def create_directory(config):
    if not config.load_model:
        if os.path.exists(config.out_dir):
            shutil.rmtree(config.out_dir)
            os.makedirs(config.out_dir)
    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.eval_dir):
        os.makedirs(config.eval_dir)