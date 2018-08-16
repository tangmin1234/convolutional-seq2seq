import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from data_load import get_batch_data, load_en_vocab, load_de_vocab
from utils import embedding, conv2d, dropout, label_smoothing


class ConvSeq2Seq(object):
    """docstring for ConvSeq2Seq"""
    def __init__(self, config):
        super(ConvSeq2Seq, self).__init__()
        self.config = config
        self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, \
                                            initializer=tf.constant_initializer(0), trainable=False)
        self.is_train = True if config.mode == 'train' else False
        self.logits = None
        self.loss = None
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.decode_input = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # add u"<S>" token
        self.build_network()
        if config.summary:
            self.summary = tf.summary.merge_all()

    def build_network(self):
        #import ipdb; ipdb.set_trace()
        config = self.config
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()
        
        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                len(de2idx),
                                num_units=config.hidden_dim,
                                scale=True,
                                scope='enc_embed')
            
            ## plus position embedding
            self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), \
                                            [tf.shape(self.x)[0], 1]),
                                config.maxlen,
                                config.hidden_dim,
                                zero_pad=False,
                                scale=False,
                                scope="enc_pe")
            
            self.enc = dropout(self.enc, config.keep_rate, is_train=self.is_train)

            self.enc_ = self.enc
            for block_idx in range(config.num_enc_block_1):
                scope = "encoder_block_{}".format(block_idx)
                enc_out = conv2d(self.enc,
                                    kernel_shape=(config.enc_kernel_width, 1),
                                    scope=scope)
                enc_out = batch_norm(enc_out, is_training=self.is_train, scope="lm"+scope)
                self.enc = enc_out

        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.decode_input,
                                len(en2idx),
                                config.hidden_dim,
                                scale=True,
                                scope='dec_embed')
            ## plus position embedding
            self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decode_input)[1]), 0), \
                                            [tf.shape(self.decode_input)[0], 1]),
                                config.maxlen,
                                config.hidden_dim,
                                zero_pad=False,
                                scale=False,
                                scope='dec_pe')

            self.dec_ = self.dec
            for block_idx in range(config.num_dec_block_1):
                scope = "decoder_block_conv_{}".format(block_idx)
                attention_scope = "decoder_block_att_{}".format(block_idx)
                dec_out = conv2d(self.dec,
                            kernel_shape=(config.dec_kernel_width, 1),
                            causal=True,
                            scope=scope)
                dec_out = attention_pool(self.enc_, self.dec, enc_out, dec_out, scope=attention_scope)
                dec_out = dec_out + self.dec
                dec_out = batch_norm(dec_out, is_training=self.is_train, scope="lm"+scope)
                self.dec = dec_out

        with tf.variable_scope('encoder'):
            for block_idx in range(config.num_enc_block_2):
                    scope = "encoder_block_{}".format(config.num_enc_block_1 + block_idx)
                    enc_out = conv2d(self.enc,
                                        kernel_shape=(config.enc_kernel_width, 1),
                                        num_outputs=config.hidden_dim_2,
                                        scope=scope)
                    enc_out = batch_norm(enc_out, is_training=self.is_train, scope="lm"+scope)
                    self.enc = enc_out

        with tf.variable_scope('decoder'):
            for block_idx in range(config.num_dec_block_2):
                scope = "decoder_block_conv_{}".format(config.num_dec_block_1 + block_idx)
                attention_scope = "decoder_block_att_{}".format(config.num_dec_block_1 + block_idx)
                dec_out = conv2d(self.dec,
                            kernel_shape=(config.dec_kernel_width, 1),
                            num_outputs=config.hidden_dim_2,
                            causal=True,
                            scope=scope)
                dec_out = attention_pool(self.enc_, self.dec, enc_out, dec_out, scope=attention_scope)
                dec_out = dec_out + self.dec
                dec_out = batch_norm(dec_out, is_training=self.is_train, scope="lm"+scope)
                self.dec = dec_out

        with tf.variable_scope("softmax_layer"):
            w = tf.get_variable('w', [config.hidden_dim, len(en2idx)])
            b = tf.get_variable('b', [len(en2idx)])
            w = tf.tile(tf.expand_dims(w, 0), [config.batch_size, 1, 1])
            self.logits = tf.matmul(dec_out, w) + b
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / tf.reduce_sum(self.istarget)
            tf.summary.scalar('acc', self.acc)
            
            if self.is_train:
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_mean(self.loss)
                tf.summary.scalar('mean_loss', self.mean_loss)

        self.tensors = {'source_sentence': self.enc_,
                        'target_sentence': self.dec_,
                        'enc_out': enc_out,
                        'dec_out':dec_out,
                        'predictions': self.preds,
                        'logits': self.logits}
        if self.is_train:
                self.tensors['loss'] = self.loss

        for key, value in self.tensors.items():
            tf.summary.histogram(key, value)

    def translate(self):
        pass
    
    def get_trainable_variables(self):
        return tf.trainable_variables()

    def get_feed_dict(self, batch):
        feed_dict = {}
        feed_dict = {self.x:batch['X'], self.y:batch['Y']}
        return feed_dict