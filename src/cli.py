import os
import tensorflow as tf
from train import main as m

# hyparameter settings

flags = tf.app.flags

flags.DEFINE_string('model_name', 'convs2s', 'convolutional seq2seq model')
flags.DEFINE_string('mode', 'train', "model running mode")
flags.DEFINE_string('out_base_dir', "/home/tangmin/mycode/reproduction.code/NLP/nmt/conv-s2s", "dictory storing model, log and etc")

flags.DEFINE_integer('batch_size', 60, "batch size")
flags.DEFINE_string('optimizer', 'adam', "Adam optimizer")
flags.DEFINE_integer('init_lr', 0.0001, "initial learning rate")
flags.DEFINE_float('keep_rate', 0.8, "dropout keep rate")
flags.DEFINE_integer('hidden_dim', 512, "hidden layer dimension for first 5 layers")
flags.DEFINE_integer('hidden_dim_2', 512, "hidder layer dimension for second 4 layers")

flags.DEFINE_integer('num_epoch', 100, "number of epoch")
flags.DEFINE_integer('num_enc_block_1', 5, "number of block in encoder ")
flags.DEFINE_integer('num_enc_block_2', 4, "")
flags.DEFINE_integer('num_dec_block_1', 5, "number of block in decoder")
flags.DEFINE_integer('num_dec_block_2', 4, "")
flags.DEFINE_integer('enc_kernel_width', 3, "kernel width for encoder")
flags.DEFINE_integer('dec_kernel_width', 5, "kernel width for decoder")
flags.DEFINE_float('grad_clip', 1, "value of gradient to be used for clipping")
flags.DEFINE_boolean('summary', True, "whether summary some tensor")
flags.DEFINE_boolean('load_model', False, "whether res``tore model")
flags.DEFINE_integer('log_period', 100, "logging period")
flags.DEFINE_integer('save_period', 500, "model saving period")
flags.DEFINE_integer('eval_period', 500, "model evaling period")
flags.DEFINE_integer('load_step', 0, "restore model from this step")
flags.DEFINE_boolean('is_debug', False, "whether debug code")

FLAGS = flags.FLAGS

def main(_):
    config = flags.FLAGS
    config.out_dir = os.path.join(config.out_base_dir, "out")
    m(config)

if __name__ == "__main__":
    tf.app.run()
