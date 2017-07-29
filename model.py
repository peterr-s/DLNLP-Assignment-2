from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, config, batch, lens_batch, label_batch, n_chars, phase = Phase.Predict):
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]

		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_size])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(
				tf.float32, shape=[batch_size, label_size])

		hidden_sz = 200

		layer = tf.reshape(tf.one_hot(self._x, n_chars), [batch_size, -1])
		hidden_w = tf.get_variable("w_hidden", shape = [layer.shape[0], hidden_sz])
		hidden_b = tf.get_variable("b_hidden", shape = [hidden_sz])
		for i in range(0, input_size) :
			hidden_outputs = tf.sigmoid(tf.matmul(layer, hidden_w) + hidden_b)
			layer = hidden_outputs
		
		w = tf.get_variable("w", shape = [layer.shape[1], label_size])
		b = tf.get_variable("b", shape = [label_size])
		logits = tf.matmul(layer, w) + b
        
	#	GRU_cell = rnn.GRUCell(100)
	#	hidden = tf.nn.dynamic_rnn(GRU_cell, tf.reshape(tf.one_hot(self._x, n_chars), [batch_size, -1]), sequence_length = self._lens, dtype = tf.float32)

	#	w = tf.get_variable("w", shape = [layer.shape[1], label_size])
	#	b = tf.get_variable("b", shape = [label_size])
	#	logits = tf.matmul(hidden, w) + b

		if phase == Phase.Train or Phase.Validation:
			losses = tf.nn.softmax_cross_entropy_with_logits(
				labels=self._y, logits=logits)
			self._loss = loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			start_lr = 0.01
			self._train_op = tf.train.AdamOptimizer(start_lr) \
				.minimize(losses)
			self._probs = probs = tf.nn.softmax(logits)

		if phase == Phase.Validation:
			# Highest probability labels of the gold data.
			hp_labels = tf.argmax(self.y, axis=1)

			# Predicted labels
			labels = tf.argmax(logits, axis=1)

			correct = tf.equal(hp_labels, labels)
			correct = tf.cast(correct, tf.float32)
			self._accuracy = tf.reduce_mean(correct)

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def lens(self):
		return self._lens

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y
