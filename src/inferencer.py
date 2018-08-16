import tensorflow as tf
import numpy as np

class Inferencer(object):
    """docstring for Inferencer"""
    def __init__(self, config, model):
        super(Inferencer, self).__init__()
        self.config = config
        self.model = model
        self.preds = model.preds

    def run(self, sess, batch):
        config = self.config
        preds = np.zeros((config.batch_size, config.maxlen), dtype=np.int32)
        for j in range(config.maxlen):
            feed_dict = {self.model.x: batch['X'],self.model.y: preds}
            _preds = sess.run([self.preds], feed_dict=feed_dict)
            preds[:, j] = _preds[0][:, j]

        return preds

