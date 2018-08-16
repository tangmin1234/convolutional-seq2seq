import tensorflow as tf

from utils import get_optimizer

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, config, model):
		super(Trainer, self).__init__()
		self.config = config
		self.model = model
		self.global_step = model.global_step
		self.acc = model.acc
		self.mean_loss = model.mean_loss
		self.summary = model.summary
		self.vars = self.model.get_trainable_variables()
		self.optimizer = get_optimizer(config.optimizer, self.config.init_lr)
		grads_and_vars = self.optimizer.compute_gradients(self.mean_loss, self.vars)
		self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

	def run_step(self, sess, batch):
		feed_dict = self.model.get_feed_dict(batch)
		if self.config.summary:
			loss, summary, acc, _ = sess.run([self.mean_loss, self.summary, self.acc, self.train_op], feed_dict=feed_dict)
		else:
			loss, acc, _ =  sess.run([self.mean_loss, self.acc, self.train_op], feed_dict=feed_dict)
			summary = None
		return loss, acc, summary