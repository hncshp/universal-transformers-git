# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Author: Huang Ping
# Date: June 28, 2018
#
#                                          ****  MODEL DESCRIPTION  ****
#                                        ACT based Universal Transformers.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import division

from six.moves import range
from tensorflow.python.estimator.model_fn import ModeKeys
import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import functools
import math
import logging
from time import time

tf.logging.set_verbosity(tf.logging.DEBUG and tf.logging.INFO)


def log(msg, level=logging.INFO):
    tf.logging.log(level, msg)


def get_time():
    return time()


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------Global parameters definition-------------------------------------------------
w_split = True

PAD = '<PAD>'
EOS = '</S>'
UNK = '<UNK>'
data_dir = './en_vi/'
buckets = 5

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

FLAGS = None
encoder_word_int_map, encoder_vocab, encoder_vocab_size = None, None, None
decoder_word_int_map, decoder_vocab, decoder_vocab_size = None, None, None

src_bucket_width, tgt_bucket_width = None, None
src_max_length, tgt_max_length = 0, 0


# -----------------------------------------Global parameters definition-------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------functions definition for whole program utility----------------------------------------

def create_vocab_tables(src_vocab_file, tgt_vocab_file):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)

    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)

    return src_vocab_table, tgt_vocab_table


def np_vocab_processing(vocab_file):
    vocab = []
    vocab_size = 0
    with open(vocab_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            vocab_size += 1
            word = line.strip()
            vocab.append(word)
    word_2_int_map = {word: index for index, word in enumerate(vocab)}
    return word_2_int_map, vocab, vocab_size


def max_line_length(file_name):
    max_length = 0
    with open(file_name, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            if len(content) > max_length:
                max_length = len(content)
    return max_length


def average_length(file_name):
    total_length = 0
    item_num = 0
    with open(file_name, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            if len(content) > 0:
                content += [EOS]
                item_num += 1
                total_length += len(content)

    return round((total_length / item_num) * FLAGS.alpha)

def one_hot(one_data, vocab_size):
    length = len(one_data)
    one_hot_array = np.zeros(shape=(length, vocab_size), dtype=np.float32)
    for time in range(length):
        one_hot_array[time][one_data[time]] = 1.0
    return one_hot_array


def en_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: encoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with PAD_ID
            if len(feature_vector) > seq_length:
                feature_temp = feature_vector[:seq_length]
            else:
                feature_temp[:len(feature_vector)] = feature_vector
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome

def de_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: decoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with PAD_ID
            if len(feature_vector) > (seq_length - 1):
                feature_temp[:(seq_length - 1)] = feature_vector[:(seq_length - 1)]
                feature_temp[(seq_length - 1)] = EOS_ID
            else:
                feature_temp[:len(feature_vector)] = feature_vector
                feature_temp[len(feature_vector)] = EOS_ID
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome

def en_characters(probabilities):
    return [encoder_vocab[c] for c in np.argmax(probabilities, 1)]

def en_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, en_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, en_characters(b))]

    return s


def de_characters(probabilities):
    return [decoder_vocab[c] for c in np.argmax(probabilities, 1)]

def de_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, de_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, de_characters(b))]
    return s

def accuracy(labels, predictions):
    return np.sum(np.argmax(labels, axis=-1) == np.argmax(predictions, axis=-1)) / (labels.shape[0] * labels.shape[1])


# --------------------------------functions definition for whole program utility----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------define the FLAGS parameters for whole model usage.---------------------------------------

def add_Argumets(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--model_dir", type=str, default=data_dir + 'model_dir',
                        help='Output directory for model saving.')
    parser.add_argument("--train_src_file_path", type=str, default=data_dir + 'train_src.txt',
                        help='training src data path.')
    parser.add_argument("--train_tgt_file_path", type=str, default=data_dir + 'train_tgt.txt',
                        help='training tgt data path.')
    parser.add_argument("--eval_src_file_path", type=str, default=data_dir + 'eval_src.txt',
                        help='eval src data path.')
    parser.add_argument("--eval_tgt_file_path", type=str, default=data_dir + 'eval_tgt.txt',
                        help='eval tgt data path.')
    parser.add_argument("--infer_src_file_path", type=str, default=data_dir + 'infer_src.txt',
                        help='infer src data path.')
    parser.add_argument("--infer_tgt_file_path", type=str, default=data_dir + 'infer_tgt.txt',
                        help='infer tgt data path.')
    parser.add_argument("--src_vocab_file_path", type=str, default=data_dir + 'vocab_src.txt',
                        help='src vocab path.')
    parser.add_argument("--tgt_vocab_file_path", type=str, default=data_dir + 'vocab_tgt.txt',
                        help='tgt vocab path.')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size.')
    parser.add_argument("--input_vocab_size", type=int, default=0,
                        help='input vocabulary size.')
    parser.add_argument("--output_vocab_size", type=int, default=0,
                        help='output vocabulary size.')
    parser.add_argument("--max_input_time_steps", type=int, default=20,
                        help='input sequence length.')
    parser.add_argument("--max_output_time_steps", type=int, default=30,
                        help='output sequence length.')
    parser.add_argument("--num_buckets", type=int, default=buckets,
                        help='number of buckets.')
    parser.add_argument("--src_bucket_width", type=int, default=0,
                        help='src_bucket_width.')
    parser.add_argument("--tgt_bucket_width", type=int, default=0,
                        help='tgt_bucket_width.')
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--num_units", type=int, default=512,
                        help="hidden node number.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="head number.")
    parser.add_argument("--encoder_recurrent_steps", type=int, default=6,
                        help="encoder recurrent steps number.")
    parser.add_argument("--decoder_recurrent_steps", type=int, default=6,
                        help="decoder recurrent steps number.")
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate: 1.0 for sgd; Adam: 0.001 | 0.0001")
    parser.add_argument("--decay_factor", type=float, default=0.5,
                        help="Learning rate decay_factor")
    parser.add_argument("--reg_lambda", type=float, default=0.001,
                        help="L1 or L2 regularization lambda")
    parser.add_argument("--act_epsilon", type=float, default=0.01,
                        help="act epsilon")
    parser.add_argument("--train_steps", type=int, default=1600,
                        help="total training steps")
    parser.add_argument("--warmup_steps", type=int, default=50000,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="max_gradient_norm.")
    parser.add_argument("--lr_warmup", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate warmup schema")
    parser.add_argument("--lr_decay", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate decay schema")
    parser.add_argument("--whitespace_or_nonws_slip", type="bool", nargs='?', const=True, default=w_split,
                        help="True means whitespace slip; False means no whitespace slip")
    parser.add_argument("--label_smooth", type="bool", nargs='?', const=True, default=True,
                        help="if label smooth or not. True means label smooth; False means not")


# -----------------------------define the FLAGS parameters for whole model usage.---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------from here, kick off training, evaluation and prediction----------------------------------
def run_main(argv=None):
    # 0. ---------------------------------------(re)set or change  FLAGS------------------------------------------------

    #   reset partial FLAGS
    if FLAGS.max_input_time_steps == 0 and FLAGS.max_output_time_steps == 0:
        FLAGS.__setattr__('max_input_time_steps', src_max_length)
        FLAGS.__setattr__('max_output_time_steps', tgt_max_length)

    FLAGS.__setattr__('src_bucket_width', math.ceil(FLAGS.max_input_time_steps / FLAGS.num_buckets))
    FLAGS.__setattr__('tgt_bucket_width', math.ceil(FLAGS.max_output_time_steps / FLAGS.num_buckets))

    FLAGS.__setattr__('input_vocab_size', encoder_vocab_size)
    FLAGS.__setattr__('output_vocab_size', decoder_vocab_size)

    print("max_input_time_steps:", FLAGS.max_input_time_steps)
    print("max_output_time_steps:", FLAGS.max_output_time_steps)
    print("input_vocab_size:", FLAGS.input_vocab_size)
    print("output_vocab_size:", FLAGS.output_vocab_size)
    print("src_bucket_width:", FLAGS.src_bucket_width)
    print("tgt_bucket_width:", FLAGS.tgt_bucket_width)

    # ------------------------------------------------------------------------------------------------------------------
    # 1. ----------------------------------------self-defined HPARAM----------------------------------------------------
    params = tf.contrib.training.HParams(
        train_steps=FLAGS.train_steps,  # define training steps
        min_eval_frequency=1000,  # define the evaluation interval
        min_summary_frequency=200,  # define checkpoints save interval
        log_steps=200,  # define the log print out interval
        learning_rate=FLAGS.learning_rate,  # define learning rate
        warmup_steps=FLAGS.warmup_steps  # define the learning_rate warmup steps
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 2. ----------------------------define the training and evaluation env parameters----------------------------------
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=params.min_eval_frequency,
        save_summary_steps=params.min_summary_frequency,
        log_step_count_steps=params.log_steps)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. ------------------------------------------Define Estimator-----------------------------------------------------
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 4. ----------------------------------prepare the training and evluation specs-------------------------------------

    train_input_fn, train_input_hook = get_train_inputs(src=FLAGS.train_src_file_path, tgt=FLAGS.train_tgt_file_path,
                                                        batch_size=FLAGS.batch_size)
    eval_input_fn, eval_input_hook = get_eval_inputs(src=FLAGS.eval_src_file_path, tgt=FLAGS.eval_tgt_file_path,
                                                     batch_size=FLAGS.batch_size)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.train_steps, hooks=[train_input_hook])

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, hooks=[eval_input_hook])
    # ------------------------------------------------------------------------------------------------------------------
    # 5. ----------------------------------construct the train_and_evaluate---------------------------------------------
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # ------------------------------------------------------------------------------------------------------------------
    # 6. --------------------------------------------Prediction Process-------------------------------------------------

    pred_out = []
    pred_feature = en_translation_file_processing(FLAGS.infer_src_file_path, FLAGS.max_input_time_steps,
                                                  FLAGS.input_vocab_size)  # [B,T,V]

    pred_label = de_translation_file_processing(FLAGS.infer_tgt_file_path, FLAGS.max_output_time_steps,
                                                FLAGS.output_vocab_size)  # [B,T,V]

    pred_label = np.transpose(pred_label, [1, 0, 2])  # [T,B,V]

    predict_input_fn, pred_input_hook = get_predict_inputs(src=FLAGS.infer_src_file_path,
                                                           batch_size=FLAGS.batch_size)

    predictions = estimator.predict(input_fn=predict_input_fn, hooks=[pred_input_hook])

    for each in predictions:
        pred_out.append(each)

    pred_out = np.transpose(pred_out, [1, 0, 2])  # [T,B,V]

    def clean(inputs):
        out = []
        for each in inputs:
            if FLAGS.whitespace_or_nonws_slip:
                temp = ''
                for word in each.strip().split():
                    if word != EOS:
                        temp += word + ' '
                    else:
                        break
                out.append(temp.strip())
            else:
                temp = ''
                for word in each:
                    if word != EOS:
                        temp += word
                    else:
                        break
                out.append(temp.strip())
        return out

    pred_label_show = clean(de_batches2string(pred_label))
    pred_data_show = clean(de_batches2string(pred_out))

    for label_show, data_show in zip(pred_label_show, pred_data_show):
        print("----------------------------------------------------------------------------------------")
        print(' Expected out String:' + label_show)
        print('Predicted out String:' + data_show)
        print("----------------------------------------------------------------------------------------")

    print("Accuracy:", accuracy(pred_out, pred_label))


def model_fn(features, labels, mode, params):
    loss = None
    train_op = None
    eval_metric_ops = {}

    def label_smoothing(inputs, l_smooth=True):
        if l_smooth:
            epsilon = 0.01
        else:
            epsilon = 0.0
        K = inputs.get_shape().as_list()[-1]
        return ((1 - epsilon) * inputs) + (epsilon / K)

    if mode != ModeKeys.PREDICT:
        features, feature_len = features
        labels_in, label_out, label_len = labels

        tgt_seq_mask_original = tf.cast(tf.sequence_mask(label_len, FLAGS.max_output_time_steps), tf.float32)  # [B,T]
        tgt_seq_mask = tf.expand_dims(tgt_seq_mask_original, axis=2)  # [B,T,1]

        label_one_hot = tf.one_hot(indices=label_out, depth=FLAGS.output_vocab_size, axis=-1)  # [B, T, V]
        label_one_hot_mask = label_one_hot * tgt_seq_mask
        logits = architecture(features, feature_len, labels_in, label_len, mode)
        logits_mask = logits * tgt_seq_mask
        predictions = tf.nn.softmax(logits) * tgt_seq_mask

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_smoothing(label_one_hot_mask, FLAGS.label_smooth), logits=logits_mask)
        loss = tf.reduce_sum(loss) / tf.count_nonzero(tgt_seq_mask, dtype=tf.float32)

        train_op = get_train_op(loss, params)
        eval_metric_ops = get_eval_metric_ops(tf.argmax(label_one_hot_mask, axis=-1), tf.argmax(predictions, axis=-1),
                                              tgt_seq_mask_original)
    else:
        feature_len = features[:, -1]
        features = features[:, :-1]
        labels_in = None
        label_len = None
        predictions = architecture(features, feature_len, labels_in, label_len, mode)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op(loss, params):
    def get_learning_rate_warmup(hparam):
        warmup_steps = hparam.warmup_steps
        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor ** (
            tf.to_float(warmup_steps - tf.train.get_global_step()))

        return tf.cond(
            tf.train.get_global_step() < hparam.warmup_steps,
            lambda: inv_decay * learning_rate,
            lambda: learning_rate,
            name="learning_rate_warump_cond")

    def get_learning_rate_decay(hparam):
        decay_factor = FLAGS.decay_factor
        start_decay_step = int(
            hparam.train_steps * 4 / 5)
        decay_times = 4
        remain_steps = hparam.train_steps - start_decay_step
        decay_steps = int(remain_steps / decay_times)

        return tf.cond(
            tf.train.get_global_step() < start_decay_step,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate,
                (tf.train.get_global_step() - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    # learning_rate schema-----------------------------------
    learning_rate = tf.constant(params.learning_rate)
    # warm-up
    if FLAGS.lr_warmup:
        learning_rate = get_learning_rate_warmup(params)
    # decay
    if FLAGS.lr_decay:
        learning_rate = get_learning_rate_decay(params)

    tf.summary.scalar("learning_rate", learning_rate)
    trainable_params = tf.trainable_variables()
    opt = None
    if FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)

    gradients = tf.gradients(loss, trainable_params, colocate_gradients_with_ops=True)

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
    train_op = opt.apply_gradients(zip(clipped_gradients, trainable_params), tf.train.get_global_step())

    return train_op
    # ----------------------------------------------------------------


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer=tf.train.AdamOptimizer
    )


def get_eval_metric_ops(labels, predictions, mask):
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            weights=mask,
            name='Accuracy')
    }


def architecture(encoder_inputs, encoder_len, decoder_inputs, decoder_len, mode):

    regularizer = tf.keras.regularizers.l2(l=FLAGS.reg_lambda)

    def query_mask(query_batch_length, query_max_length):

        mask = tf.cast(tf.sequence_mask(query_batch_length, query_max_length), tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, axis=2)  # [B,T,1]

        return mask

    def key_mask(key_inputs):

        mask = tf.cast(tf.equal(key_inputs, PAD_ID), tf.float32)  # [B,T]
        mask *= (-2 ** 32 + 1)
        mask = tf.tile(mask, [FLAGS.num_heads, 1])
        mask = tf.expand_dims(mask, axis=1)  # [head*B, 1, T]

        return mask

    def normalization(inputs, scope):

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            epsilon = 1e-8
            params_shape = inputs.get_shape().as_list()[-1]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

            zeros = lambda: tf.zeros([params_shape], dtype=tf.float32)
            beta = tf.get_variable(name='beta', initializer=zeros, dtype=tf.float32)

            ones = lambda: tf.ones([params_shape], dtype=tf.float32)
            gamma = tf.get_variable(name='gamma', initializer=ones, dtype=tf.float32)
            """
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
            """
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

        return outputs

    with tf.variable_scope(name_or_scope='encoder_word_embedding', reuse=tf.AUTO_REUSE):
        encoder_embedding = tf.get_variable(name='en_embedding',
                                            shape=[FLAGS.input_vocab_size, FLAGS.num_units],
                                            dtype=tf.float32,
                                            initializer=tf.variance_scaling_initializer())

        encoder_embedding = tf.concat((tf.zeros([1, FLAGS.num_units]), encoder_embedding[1:, :]), axis=0)

    with tf.variable_scope(name_or_scope='decoder_word_embedding', reuse=tf.AUTO_REUSE):
        decoder_embedding = tf.get_variable(name='de_embedding',
                                            shape=[FLAGS.output_vocab_size, FLAGS.num_units],
                                            dtype=tf.float32,
                                            initializer=tf.variance_scaling_initializer())

        decoder_embedding = tf.concat((tf.zeros([1, FLAGS.num_units]), decoder_embedding[1:, :]), axis=0)

    def encoder_input_embedding(inputs):
        outputs = tf.nn.embedding_lookup(encoder_embedding, inputs)
        outputs = outputs * FLAGS.num_units ** 0.5
        return outputs

    def decoder_input_embedding(inputs):
        outputs = tf.nn.embedding_lookup(decoder_embedding, inputs)
        outputs = outputs * FLAGS.num_units ** 0.5
        return outputs

    position_length = max(FLAGS.max_input_time_steps, FLAGS.max_output_time_steps)

    position_table = np.full(shape=[position_length, FLAGS.num_units], fill_value=0.0, dtype=np.float32)

    for pos in range(position_length):
        for i in range(FLAGS.num_units):
            if i % 2 == 0:
                position_table[pos, i] = np.sin(pos / np.power(10000, 2.0 * i / FLAGS.num_units))
            else:
                position_table[pos, i] = np.cos(pos / np.power(10000, 2.0 * (i - 1) / FLAGS.num_units))

    position_table = tf.convert_to_tensor(position_table)

    def encoder_position_embedding(inputs):
        # input shape is [B,T,H]
        step = tf.range(FLAGS.max_input_time_steps)
        batch_step = tf.tile([step], [tf.shape(inputs)[0], 1])
        outputs = tf.nn.embedding_lookup(position_table, batch_step)
        outputs = outputs * FLAGS.num_units ** 0.5
        return outputs

    def decoder_position_embedding(inputs):
        # input shape is [B,T,H]
        step = tf.range(FLAGS.max_output_time_steps)
        batch_step = tf.tile([step], [tf.shape(inputs)[0], 1])
        outputs = tf.nn.embedding_lookup(position_table, batch_step)
        outputs = outputs * FLAGS.num_units ** 0.5
        return outputs


    recurrent_steps = max(FLAGS.encoder_recurrent_steps, FLAGS.decoder_recurrent_steps)
    act_step_table = np.full(shape=[recurrent_steps, FLAGS.num_units], fill_value=0.0, dtype=np.float32)

    for act_step in range(recurrent_steps):
        for i in range(FLAGS.num_units):
            if i % 2 == 0:
                act_step_table[act_step, i] = np.sin(act_step / np.power(10000, 2.0 * i / FLAGS.num_units))
            else:
                act_step_table[act_step, i] = np.cos(act_step / np.power(10000, 2.0 * (i - 1) / FLAGS.num_units))

    act_step_table = tf.convert_to_tensor(act_step_table)

    def recurrent_step_embedding(inputs, current_step):
        batch_steps = tf.fill(tf.shape(inputs)[:-1], current_step)
        outputs = tf.nn.embedding_lookup(act_step_table, batch_steps)
        outputs = outputs * FLAGS.num_units ** 0.5
        return outputs

    def projection_layer(inputs):
        with tf.variable_scope(name_or_scope='projection', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs=inputs, units=FLAGS.output_vocab_size, name='pl')
        return outputs

    def multihead_attention(query_inputs, key_inputs, value_inputs, key_mask, scope):

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(query_inputs, FLAGS.num_units, activation=tf.nn.relu, name='q')
            K = tf.layers.dense(key_inputs, FLAGS.num_units, activation=tf.nn.relu, name='k')
            V = tf.layers.dense(value_inputs, FLAGS.num_units, activation=tf.nn.relu, name='v')

            # [FLAGS.num_heads*B, T, FLAGS.num_units/FLAGS.num_heads]
            Q_head = tf.concat(tf.split(Q, FLAGS.num_heads, axis=-1), axis=0)
            K_head = tf.concat(tf.split(K, FLAGS.num_heads, axis=-1), axis=0)
            V_head = tf.concat(tf.split(V, FLAGS.num_heads, axis=-1), axis=0)

            attention = tf.matmul(Q_head, tf.transpose(K_head, [0, 2, 1])) / (
                    (FLAGS.num_units / FLAGS.num_heads) ** 0.5)  # [FLAGS.num_heads*B, Tq, Tk]

            attention += key_mask
            attention = tf.nn.softmax(attention)
            attention = tf.matmul(attention, V_head)  # [FLAGS.num_heads*B, Tq, FLAGS.num_units/FLAGS.num_heads]
            outputs = tf.concat(tf.split(attention, FLAGS.num_heads, axis=0), axis=-1)  # [B, Tq, FLAGS.num_units]

            #residual
            outputs = query_inputs + outputs
            # Dropout
            outputs = tf.layers.dropout(outputs, rate=FLAGS.dropout, training=(mode == ModeKeys.TRAIN))

            outputs = normalization(outputs, scope)

        return outputs

    def transition_function(inputs, scope):

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs=inputs, units=4 * FLAGS.num_units, name='l1')
            outputs = tf.layers.dense(inputs=outputs, units=FLAGS.num_units, name='l2')

            outputs = outputs + inputs
            outputs = tf.layers.dropout(outputs, rate=FLAGS.dropout, training=(mode == ModeKeys.TRAIN))

            outputs = normalization(outputs, scope)

        return outputs

    def encoder_act_unit(query_inputs, key_mask, current_step):
        ## Positional Encoding
        outputs = query_inputs
        outputs += encoder_position_embedding(outputs)

        # timesteps embedding
        outputs += recurrent_step_embedding(inputs=outputs, current_step=current_step)

        # multi_attention
        outputs = multihead_attention(query_inputs=outputs,
                                      key_inputs=outputs,
                                      value_inputs=outputs,
                                      key_mask=key_mask,
                                      scope='encoder_mul_attention')

        # transition_function
        outputs = transition_function(inputs=outputs,
                                      scope='encoder_transition')

        return outputs

    def decoder_act_unit(enc_outputs, query_inputs, dec_key_mask, enc_key_mask, current_step):
        ## Positional Encoding
        outputs = query_inputs
        outputs += decoder_position_embedding(outputs)
        # timesteps embedding
        outputs += recurrent_step_embedding(inputs=outputs, current_step=current_step)

        # self_multi_attention
        outputs = multihead_attention(query_inputs=outputs,
                                      key_inputs=outputs,
                                      value_inputs=outputs,
                                      key_mask=dec_key_mask,
                                      scope='decoder_self_mul_attention')
        # enc_dec_mul_attention
        outputs = multihead_attention(query_inputs=outputs,
                                      key_inputs=enc_outputs,
                                      value_inputs=enc_outputs,
                                      key_mask=enc_key_mask,
                                      scope='enc_dec_mul_attention')
        # transition_function
        outputs = transition_function(inputs=outputs,
                                      scope='decoder_transition')

        return outputs

    def act(inputs, max_steps, processing_unit):

        def if_continue(_prev_outputs, _final_outputs, _prev_accum_halt, _go_sign, _current_step):

            go = tf.logical_and(tf.reduce_any(tf.greater(_go_sign, 0.0)), tf.less(_current_step, max_steps))
            return go

        def step_in(_prev_outputs, _final_outputs, _prev_accum_halt, _go_sign, _current_step):

            _current_outputs = processing_unit(query_inputs=_prev_outputs, current_step=_current_step)
            with tf.variable_scope(name_or_scope='act_halt', reuse=tf.AUTO_REUSE):
                _current_halt = tf.layers.dense(inputs=_current_outputs, units=1, activation=tf.nn.sigmoid, name='halt')  # [B, T, 1]

            if _current_step == max_steps-1:

                R_N_t = (1.0 - _prev_accum_halt) * _go_sign

                _current_accum_halt = _prev_accum_halt * _go_sign + R_N_t

                _final_outputs += _current_outputs * R_N_t

                _go_sign = tf.zeros_like(_go_sign, dtype=tf.float32)

                _current_step += 1

            else:

                _current_accum_halt = (_current_halt + _prev_accum_halt) * _go_sign

                _current_RNT_sign = tf.cast(tf.greater_equal(_current_accum_halt, 1.0 - FLAGS.act_epsilon),
                                            dtype=tf.float32)

                R_N_t = (1.0 - _prev_accum_halt) * _current_RNT_sign

                _current_halt = (_current_halt * tf.cast(tf.less(_current_RNT_sign, 1.0), dtype=tf.float32) + R_N_t) * _go_sign

                _current_accum_halt = (_prev_accum_halt + _current_halt) * _go_sign

                _final_outputs += _current_outputs * _current_halt

                _go_sign -= _current_RNT_sign

                _current_step += 1

            return _current_outputs, _final_outputs, _current_accum_halt, _go_sign, _current_step

        prev_outputs = inputs  # [B,T,H]

        final_outputs = tf.zeros(tf.shape(inputs))  # [B,T,H]

        prev_accum_halt = tf.expand_dims(tf.zeros(tf.shape(inputs)[:-1]), axis=2)  # [B,T,1]

        go_sign = tf.expand_dims(tf.ones(tf.shape(inputs)[:-1]), axis=2)  # [B,T,1]

        current_step = 0

        prev_outputs, final_outputs, prev_accum_halt, go_sign, current_step = tf.while_loop(
            cond=if_continue,
            body=step_in,
            loop_vars=[prev_outputs, final_outputs, prev_accum_halt, go_sign, current_step],
            shape_invariants=[
                prev_outputs.get_shape(),
                final_outputs.get_shape(),
                prev_accum_halt.get_shape(),
                go_sign.get_shape(),
                tf.TensorShape([])
            ]
        )

        return final_outputs

    def encoder(encoder_in):
        k_mask = key_mask(encoder_in)

        with tf.variable_scope(name_or_scope='encoding', reuse=tf.AUTO_REUSE):
            ## Embedding
            enc = encoder_input_embedding(encoder_in)

            ## ACT
            processing_unit = functools.partial(
                encoder_act_unit,
                key_mask=k_mask
            )

            encoder_outputs = act(enc, FLAGS.encoder_recurrent_steps, processing_unit)

        return encoder_outputs

    def decoder_train_eval(encoder_in, decoder_in, encoder_out):

        batch_size = tf.shape(decoder_in)[0]
        outputs = tf.zeros([FLAGS.max_output_time_steps, FLAGS.num_units], dtype=tf.float32)  # [T, H]
        outputs = tf.tile([outputs], [batch_size, 1, 1])  # [B,T,H]
        tgt_length = 0

        def step_in(time_steps, last_outputs):
            length_mask = tf.cast(tf.sequence_mask([time_steps + 1], FLAGS.max_output_time_steps), tf.int32)
            length_mask = tf.tile(length_mask, [batch_size, 1])  # [B,T]

            pos_mask = tf.one_hot(indices=time_steps, depth=FLAGS.max_output_time_steps, axis=-1, dtype=tf.float32)  # [T]
            pos_mask = tf.tile([pos_mask], [batch_size, 1])  # [B,T]
            pos_mask = tf.expand_dims(pos_mask, axis=2)  # [B,T,1]

            tgt_inputs = decoder_in * length_mask

            dec_k_mask = key_mask(tgt_inputs)
            enc_k_mask = key_mask(encoder_in)

            with tf.variable_scope(name_or_scope='decoding', reuse=tf.AUTO_REUSE):
                ## Embedding
                dec = decoder_input_embedding(tgt_inputs)

                ## ACT
                processing_unit = functools.partial(
                    decoder_act_unit,
                    enc_outputs=encoder_out,
                    dec_key_mask=dec_k_mask,
                    enc_key_mask=enc_k_mask
                )

                prediction = act(dec, FLAGS.decoder_recurrent_steps, processing_unit)
                last_outputs += prediction * pos_mask

            time_steps +=1

            return time_steps, last_outputs

        tgt_length, outputs = tf.while_loop(
            cond=lambda tgt_length, *_: tgt_length < FLAGS.max_output_time_steps,
            body=step_in,
            loop_vars=[tgt_length, outputs],
            shape_invariants=[
                tf.TensorShape([]),
                outputs.get_shape()
            ]
        )

        logits = projection_layer(outputs)  # [B,T,V]
        return logits


    def decoder_predict(encoder_in, encoder_out):

        batch_size = tf.shape(encoder_in)[0]
        outputs = tf.zeros([1, FLAGS.max_output_time_steps], dtype=tf.int32)  # [1,T]
        outputs = tf.tile(outputs, [batch_size, 1]) #[B,T]
        tgt_length = 0

        def step_in(time_steps, last_outputs):

            paddings = tf.constant([[0, 0], [1, 0]])
            tgt_inputs = tf.pad(last_outputs, paddings, "CONSTANT", constant_values=1)
            tgt_inputs = tgt_inputs[:, :FLAGS.max_output_time_steps]

            length_mask = tf.cast(tf.sequence_mask([time_steps + 1], FLAGS.max_output_time_steps), tf.int32)
            length_mask = tf.tile(length_mask, [batch_size, 1])  # [B,T]

            pos_mask = tf.one_hot(indices=time_steps, depth=FLAGS.max_output_time_steps, axis=-1, dtype=tf.int32)  # [T]
            pos_mask = tf.tile([pos_mask], [batch_size, 1])  # [B,T]

            tgt_inputs = tgt_inputs * length_mask

            dec_k_mask = key_mask(tgt_inputs)
            enc_k_mask = key_mask(encoder_in)

            with tf.variable_scope(name_or_scope='decoding', reuse=tf.AUTO_REUSE):
                ## Embedding
                dec = decoder_input_embedding(tgt_inputs)

                ## ACT
                processing_unit = functools.partial(
                    decoder_act_unit,
                    enc_outputs=encoder_out,
                    dec_key_mask=dec_k_mask,
                    enc_key_mask=enc_k_mask
                )

                prediction = act(dec, FLAGS.decoder_recurrent_steps, processing_unit)

            prediction = tf.nn.softmax(projection_layer(prediction))  # [B,T,V]
            prediction = tf.argmax(prediction, axis=-1, output_type=tf.int32)  # [B,T]
            last_outputs += prediction * pos_mask

            time_steps += 1

            return time_steps, last_outputs

        tgt_length, outputs = tf.while_loop(
            cond=lambda tgt_length, *_: tgt_length < FLAGS.max_output_time_steps,
            body=step_in,
            loop_vars=[tgt_length, outputs],
            shape_invariants=[
                tf.TensorShape([]),
                outputs.get_shape()
            ]
        )

        prediction = tf.one_hot(indices=outputs, depth=FLAGS.output_vocab_size, axis=-1)  # [B, T, V]
        return prediction

    enc_outputs = encoder(encoder_inputs)
    if mode != ModeKeys.PREDICT:
        logits = decoder_train_eval(encoder_inputs, decoder_inputs, enc_outputs)
    else:
        logits = decoder_predict(encoder_inputs, enc_outputs)

    return logits


# ----------------------------------------------Define data loaders ----------------------------------------------------

class InitializerHook(tf.train.SessionRunHook):
    # Hook to initialise data iterator after Session is created.

    def __init__(self):
        super(InitializerHook, self).__init__()
        self.initializer_func = None

    def after_create_session(self, session, coord):
        # Initialise the iterator after the session has been created.
        self.initializer_func(session)


# Define the training inputs
def get_train_inputs(src, tgt, batch_size):

    initializer_hook = InitializerHook()

    def train_inputs():

        with tf.name_scope('Training_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            tgt_datasets = tf.data.TextLineDataset(tgt)
            src_tgt_datasets = tf.data.Dataset.zip((src_datasets, tgt_datasets))
            if FLAGS.whitespace_or_nonws_slip:
                src_tgt_datasets = src_tgt_datasets.map(
                    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values))
            else:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                    tf.string_split([src], delimiter='').values, tf.string_split([tgt], delimiter='').values))

            src_tgt_datasets = src_tgt_datasets.filter(
                lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src[:FLAGS.max_input_time_steps], tgt[:FLAGS.max_output_time_steps - 1]))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src, tf.concat([[EOS_ID], tgt], 0), tf.concat([tgt, [EOS_ID]], 0)))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))

            def batching_func(x):
                return x.padded_batch(
                    batch_size,
                    padded_shapes=(
                        tf.TensorShape([FLAGS.max_input_time_steps]),
                        # src_input
                        tf.TensorShape([FLAGS.max_output_time_steps]),
                        # tgt_input
                        tf.TensorShape([FLAGS.max_output_time_steps]),
                        # tgt_output
                        tf.TensorShape([]),  # src_len
                        tf.TensorShape([])),  # tgt_len
                    padding_values=(
                        PAD_ID,  # src_input
                        PAD_ID,  # tgt_input
                        PAD_ID,  # tgt_output
                        0,  # src_len -- unused
                        0))  # tgt_len -- unused

            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):

                bucket_id = tf.maximum(src_len // FLAGS.src_bucket_width, tgt_len // FLAGS.tgt_bucket_width)
                return tf.to_int64(tf.minimum(FLAGS.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = src_tgt_datasets.apply(
                tf.data.experimental.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

            batched_dataset = batched_dataset.shuffle(2000)
            batched_dataset = batched_dataset.repeat(None)

            iterator = batched_dataset.make_initializable_iterator()
            next_feature, next_label_in, next_label_out, feature_len, label_len = iterator.get_next()

            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)
            return (next_feature, feature_len), (next_label_in, next_label_out, label_len)

    return train_inputs, initializer_hook


def get_eval_inputs(src, tgt, batch_size):
    initializer_hook = InitializerHook()

    def eval_inputs():

        with tf.name_scope('Eval_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            tgt_datasets = tf.data.TextLineDataset(tgt)
            src_tgt_datasets = tf.data.Dataset.zip((src_datasets, tgt_datasets))
            if FLAGS.whitespace_or_nonws_slip:
                src_tgt_datasets = src_tgt_datasets.map(
                    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values))
            else:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                    tf.string_split([src], delimiter='').values, tf.string_split([tgt], delimiter='').values))

            src_tgt_datasets = src_tgt_datasets.filter(
                lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src[:FLAGS.max_input_time_steps], tgt[:FLAGS.max_output_time_steps - 1]))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src, tf.concat([[EOS_ID], tgt], 0), tf.concat([tgt, [EOS_ID]], 0)))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))

            def batching_func(x):
                return x.padded_batch(
                    batch_size,
                    padded_shapes=(
                        tf.TensorShape([FLAGS.max_input_time_steps]),
                        # src_input
                        tf.TensorShape([FLAGS.max_output_time_steps]),
                        # tgt_input
                        tf.TensorShape([FLAGS.max_output_time_steps]),
                        # tgt_output
                        tf.TensorShape([]),  # src_len
                        tf.TensorShape([])),  # tgt_len
                    padding_values=(
                        PAD_ID,  # src_input
                        PAD_ID,  # tgt_input
                        PAD_ID,  # tgt_output
                        0,  # src_len -- unused
                        0))  # tgt_len -- unused

            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):

                bucket_id = tf.maximum(src_len // FLAGS.src_bucket_width, tgt_len // FLAGS.tgt_bucket_width)
                return tf.to_int64(tf.minimum(FLAGS.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = src_tgt_datasets.apply(
                tf.data.experimental.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

            batched_dataset = batched_dataset.shuffle(2000)
            batched_dataset = batched_dataset.repeat(None)

            iterator = batched_dataset.make_initializable_iterator()
            next_feature, next_label_in, next_label_out, feature_len, label_len = iterator.get_next()

            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)

            return (next_feature, feature_len), (next_label_in, next_label_out, label_len)

    return eval_inputs, initializer_hook


def get_predict_inputs(src, batch_size):

    initializer_hook = InitializerHook()

    def predict_inputs():

        with tf.name_scope('Predict_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            if FLAGS.whitespace_or_nonws_slip:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src]).values)
            else:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src], delimiter='').values)

            src_datasets = src_datasets.filter(
                lambda src: tf.size(src) > 0)

            src_datasets = src_datasets.map(
                lambda src: src[:FLAGS.max_input_time_steps])

            src_datasets = src_datasets.map(
                lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

            src_datasets = src_datasets.map(lambda src: (src, tf.size(src)))

            src_datasets = src_datasets.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([FLAGS.max_input_time_steps]),  # src_input
                    tf.TensorShape([])),  # src_len
                padding_values=(
                    PAD_ID,  # src_input
                    0))  # src_len

            iterator = src_datasets.make_initializable_iterator()
            next_feature, feature_len = iterator.get_next()
            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)

            return tf.concat([next_feature, tf.expand_dims(feature_len, axis=1)], axis=-1)

    return predict_inputs, initializer_hook


# -----------------------------------------------------Run script ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_Argumets(parser)
    FLAGS, unparsered = parser.parse_known_args()

    src_max_length = max_line_length(FLAGS.train_src_file_path)
    tgt_max_length = max_line_length(FLAGS.train_tgt_file_path)
    encoder_word_int_map, encoder_vocab, encoder_vocab_size = np_vocab_processing(FLAGS.src_vocab_file_path)
    decoder_word_int_map, decoder_vocab, decoder_vocab_size = np_vocab_processing(FLAGS.tgt_vocab_file_path)

    tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsered)