import os
import tensorflow as tf

class GraphHandler(object):
    """docstring for GraphHandler"""
    def __init__(self, config):
        super(GraphHandler, self).__init__()
        self.config = config
        self.saver = tf.train.Saver()
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        if self.config.load_model:
            self._load(sess)
        
        if self.config.mode == 'train':
            self.writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

    def _load(self, sess):
        config = self.config
        if config.load_step > 0:
            save_path = os.path.join(config.save_dir, "{}-{}"\
                        .format(config.model_name, config.load_step))
        else:
            save_dir = config.save_dir
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "can not load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print "Loading saved model from {}".format(save_path)
        self.saver.restore(sess, save_path) 

    def save_model(self, sess, global_step):
        self.saver.save(sess, self.save_path, global_step=global_step)
        
    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.writer.add_summary(summary, global_step)