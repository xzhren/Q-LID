#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2022 Alibaba Inc. All Rights Reserved.
#
# History:
# 2022.07.05. Be created by xingzhang.rxz. Used for Query Language Identification.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

from config import *
from data_reader import *
import tensorflow as tf
import sys
import numpy as np
import common_attention
import common_layers
import subprocess
import re
from random import shuffle
from tensorflow.python.client import device_lib
from six.moves import xrange  # backward compatible with python2
from tensorflow.python.ops import array_ops
#from dp import GraphDispatcher
tf.logging.set_verbosity(tf.logging.INFO)

def prepare_encoder_input(src_wids, src_masks, params):
    src_vocab_size = params["src_vocab_size"]
    hidden_size = params["hidden_size"]
    with tf.variable_scope('Source_Side'):
        src_emb = common_layers.embedding(src_wids, src_vocab_size, hidden_size)
    src_emb *= hidden_size**0.5
    # encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(1-src_masks)
    encoder_input = common_attention.add_timing_signal_1d(src_emb)
    encoder_input = tf.multiply(encoder_input, tf.expand_dims(src_masks, 2))
    return encoder_input
    # return src_emb

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_sum(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def textcnn_model_loss(features, labels, params):
    last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64) # shape: [batch_size, 1], values=0
    src_wids = tf.concat([features,last_padding],1) 
    src_masks = tf.to_float(tf.not_equal(src_wids,0))
    shift_src_masks = src_masks[:,:-1]
    shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)

    encoder_input = prepare_encoder_input(src_wids, shift_src_masks, params)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - params['prepost_dropout'])
    print("encoder_input:", encoder_input)

    # params
    filter_sizes = [3,4,5]
    embedding_size = params["hidden_size"]
    num_filters = 128
    l2_reg_lambda = 0.0
    dropout_keep_prob = 1.0 - params['prepost_dropout']
    num_classes = params["number_of_classes"]
    # sequence_length = tf.shape(features)[1]
    # sequence_length = array_ops.shape(features)[1]
    # sequence_length = features.get_shape().as_list()[1]
    # input_y = labels
    embedded_chars_expanded = tf.expand_dims(encoder_input, -1)
    print("embedded_chars_expanded:", embedded_chars_expanded) # ?, ?, 512, 1
    print("input_y:", labels)
    # print("sequence_length:", sequence_length)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            print("conv:", conv) # ?, ?, 1, 128
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            print("h:", h) # ?, ?, 1, 128
            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(
            #     h,
            #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #     # ksize=[1, -1, 1, 1],
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name="pool")
            from tensorflow.python.ops import gen_nn_ops
            # pooled = gen_nn_ops.max_pool_v2(
            #     h,
            #     ksize=[1, tf.shape(features)[1] - filter_size + 1, 1, 1],
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name="pool")
            pooled = tf.reduce_max(h, axis=1, name="pool", keep_dims=True)
            print("pooled:", pooled) # ?, 1, 1, 128
            pooled_outputs.append(pooled)
            
    def layer_process(x, y, flag, dropout):
        if flag == None:
            return y
        for c in flag:
            if c == 'a':
                y = x+y
            elif c == 'n':
                y = common_layers.layer_norm(y)
            elif c == 'd':
                y = tf.nn.dropout(y, 1.0 - dropout)
        return y

    
    x = tf.concat(pooled_outputs, 3)
    x = tf.reshape(x, [-1, 3, num_filters])
    print("x", x)
    o,w = common_attention.multihead_attention(
            layer_process(None,x,'n',0.1),
            None,
            None,
            num_filters,
            num_filters,
            num_filters,
            8,
            0.1,
            summaries=False,
            name="encoder_self_attention")
    print("w", w)
    x = layer_process(x,o,'da',0.1)
    # o = transformer_ffn_layer(, params)
    o = common_layers.conv_hidden_relu(
            layer_process(None,x,'n',0.1),
            num_filters,
            num_filters,
            dropout=0.1)
    print("o", o)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool_flat = tf.reshape(o, [-1, num_filters_total])
    print("h_pool:", h_pool_flat) # ?, 384


    # # Combine all the pooled features
    # num_filters_total = num_filters * len(filter_sizes)
    # h_pool = tf.concat(pooled_outputs, 3)
    # h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # print("h_pool:", h_pool, h_pool_flat) # ?, 384

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        targets = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
        loss = tf.reduce_sum(losses) / tf.cast(tf.shape(features)[0], dtype=tf.float32)
        loss += l2_reg_lambda * l2_loss
        # loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # Accuracy
    with tf.name_scope("accuracy"):
        # correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        dist = tf.nn.softmax(scores)
        res = tf.argmax(dist, tf.rank(dist)-1)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=res, name='acc_op')
        print("acc:", accuracy)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])
    predictions = {"predict_score": dist }
    add_dict_to_collection("predictions", predictions)
    return loss, accuracy

def textcnn_train_fn(features, labels, mode, params):
    num_gpus = params['num_gpus']
    gradient_clip_value = params['gradient_clip_value']
    step = tf.to_float(tf.train.get_global_step())
    warmup_steps = params['warmup_steps']
    if params['learning_rate_decay'] == 'sqrt':
        lr_warmup = params['learning_rate_peak'] * tf.minimum(1.0,step/warmup_steps)
        lr_decay = params['learning_rate_peak'] * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
        lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
    elif params['learning_rate_decay'] == 'exp':
        lr = tf.train.exponential_decay(params['learning_rate_peak'],
                global_step=step,
                decay_steps=params['decay_steps'],
                decay_rate=params['decay_rate'])
    else:
        tf.logging.info("learning rate decay strategy not supported")
        sys.exit()
    if params['optimizer'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
    else:
        tf.logging.info("optimizer not supported")
        sys.exit()
    #optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-09)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, gradient_clip_value)

    def fill_until_num_gpus(inputs, num_gpus):
                outputs = inputs
                for i in range(num_gpus - 1):
                    outputs = tf.concat([outputs, inputs], 0)
                outputs= outputs[:num_gpus,]
                return outputs

    features = tf.cond(tf.shape(features)[0] < num_gpus, lambda: fill_until_num_gpus(features, num_gpus), lambda: features)
    labels = tf.cond(tf.shape(labels)[0] < num_gpus, lambda: fill_until_num_gpus(labels, num_gpus), lambda: labels)
    feature_shards = common_layers.approximate_split(features, num_gpus)
    label_shards = common_layers.approximate_split(labels, num_gpus)
    #loss_shards = dispatcher(get_loss, feature_shards, label_shards, params)
    devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    loss_shards = []
    grad_shards = []
    train_acc_shards = []
    for i, device in enumerate(devices):
        #if i > 0:
            #var_scope.reuse_variables()
        with tf.variable_scope( tf.get_variable_scope(), reuse=True if i > 0 else None):
            with tf.device(device):
                loss, train_acc = textcnn_model_loss(feature_shards[i], label_shards[i], params)
                grads = optimizer.compute_gradients(loss)
                #tf.get_variable_scope().reuse_variables()
                loss_shards.append(loss)
                grad_shards.append(grads)
                train_acc_shards.append(train_acc)
    #loss_shards = tf.Print(loss_shards,[loss_shards])
    loss = tf.reduce_mean(loss_shards)
    grad = average_gradients(grad_shards)
    train_acc = tf.reduce_mean(train_acc_shards)
    train_op = optimizer.apply_gradients(grad, global_step=tf.train.get_global_step())
    if params['ema_decay'] > 0.0:
        ema = tf.train.ExponentialMovingAverage(decay=params['ema_decay'], num_updates=tf.train.get_global_step())
        with tf.control_dependencies([train_op]):
            train_op = ema.apply(tf.trainable_variables())
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : train_acc}, every_n_iter=100)
    summary_hook = tf.train.SummarySaverHook(save_steps=100, summary_op=tf.summary.merge_all()) 
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks = [logging_hook, summary_hook])

def textcnn_eval_fn(features, labels, mode, params):
    loss, train_acc = textcnn_model_loss(features, labels, params)
    return tf.estimator.EstimatorSpec(mode=mode, loss=tf.constant(0.0))

def textcnn_pred_fn():
    pass


def textcnn_model_fn(features, labels, mode, params):
    with tf.variable_scope('TextCNNModel') as var_scope:
        if mode == tf.estimator.ModeKeys.TRAIN:
            return textcnn_train_fn(features, labels, mode, params)
        if mode == tf.estimator.ModeKeys.EVAL:
            return textcnn_eval_fn(features, labels, mode, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return textcnn_pred_fn()


def add_dict_to_collection(collection_name, dict_):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)

