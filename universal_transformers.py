# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Author: Huang Ping
# Date: Sept 30, 2018
#
#                                          ****  MODEL DESCRIPTION  ****
#                                        ACT based Universal Transformers.
# ----------------------------------------------------------------------------------------------------------------------

# the real division operations. if not import division, 5/3=1, if import division, 5/3=1.66666
# "from __future__  import division" shall be before all the import parts
from __future__ import division

from six.moves import range
from tensorflow.python.estimator.model_fn import ModeKeys
import argparse
import sys
import numpy as np
import tensorflow as tf
from tqdm import trange
from tensorflow.python.ops import lookup_ops
import functools
import math
import logging
from time import time


# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG and tf.logging.INFO)


def log(msg, level=logging.INFO):
    tf.logging.log(level, msg)


def get_time():
    return time()


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------Global parameters definition-------------------------------------------------

PAD = '<PAD>'
EOS = '</S>'
UNK = '<UNK>'
data_dir = './en_vi/'


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

# define the src and target vocabulary. the table constructed by lookup_ops need to be initialized.
def create_vocab_tables(src_vocab_file, tgt_vocab_file):
    # Creates vocab tables for src_vocab_file and tgt_vocab_file.
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)

    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)

    return src_vocab_table, tgt_vocab_table


# none tensor style vocab and word to id map. for data and image show in prediction stage.
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


# statistic the sentence max length for max_input and max_output parameters definition.
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


# self-defined average_length function for src and tgt average length estimation

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


# self-defined one_hot function.
def one_hot(one_data, vocab_size):
    length = len(one_data)
    one_hot_array = np.zeros(shape=(length, vocab_size), dtype=np.float32)
    for time in range(length):
        one_hot_array[time][one_data[time]] = 1.0
    return one_hot_array


# self-defined function to process the one_hot transition for files.
def en_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: encoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with UNK_ID
            if len(feature_vector) > seq_length:
                feature_temp = feature_vector[:seq_length]
            else:
                feature_temp[:len(feature_vector)] = feature_vector
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome


# self-defined function to process the one_hot transition for files.
def de_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: decoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with UNK_ID
            if len(feature_vector) > (seq_length - 1):
                feature_temp[:(seq_length - 1)] = feature_vector[:(seq_length - 1)]
                feature_temp[(seq_length - 1)] = EOS_ID
            else:
                feature_temp[:len(feature_vector)] = feature_vector
                feature_temp[len(feature_vector)] = EOS_ID
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome


# self-defined function for one_hot to word transition based on the vocabulary.
def en_characters(probabilities):
    return [encoder_vocab[c] for c in np.argmax(probabilities, 1)]


# self-defined function for batched one_hot data to string transition.
# NOTE: the input data shall be of [T,B,V] shape.
def en_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, en_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, en_characters(b))]

    return s


# self-defined function for one_hot to word transition based on the vocabulary.
def de_characters(probabilities):
    return [decoder_vocab[c] for c in np.argmax(probabilities, 1)]


# self-defined function for batched one_hot data to string transition.
# NOTE: the input data shall be of [T,B,V] shape.
def de_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, de_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, de_characters(b))]
    return s


# self-defined function for accuracy in prediction stage.
# labels and predictions shall be of [T,B,V] shape
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
    parser.add_argument("--max_input_time_steps", type=int, default=40,
                        help='input sequence length.')
    parser.add_argument("--max_output_time_steps", type=int, default=45,
                        help='output sequence length.')
    parser.add_argument("--num_buckets", type=int, default=5,
                        help='number of buckets.')
    parser.add_argument("--src_bucket_width", type=int, default=0,
                        help='src_bucket_width.')
    parser.add_argument("--tgt_bucket_width", type=int, default=0,
                        help='tgt_bucket_width.')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--num_units", type=int, default=128,
                        help="hidden node number.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="head number.")
    parser.add_argument("--encoder_maxsteps", type=int, default=12,
                        help="encoder maxsteps number.")
    parser.add_argument("--decoder_maxsteps", type=int, default=12,
                        help="decoder maxsteps number.")
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate: 1.0 for sgd; Adam: 0.001 | 0.0001")
    parser.add_argument("--decay_factor", type=float, default=0.5,
                        help="Learning rate decay_factor")
    parser.add_argument("--reg_lambda", type=float, default=0.001,
                        help="L1 or L2 regularization lambda")
    parser.add_argument("--act_epsilon", type=float, default=0.01,
                        help="act epsilon")
    parser.add_argument("--train_steps", type=int, default=5000,
                        help="total training steps")
    parser.add_argument("--warmup_steps", type=int, default=50000,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="max_gradient_norm.")
    parser.add_argument("--lr_warmup", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate warmup schema")
    parser.add_argument("--lr_decay", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate decay schema")
    parser.add_argument("--whitespace_or_nonws_slip", type="bool", nargs='?', const=True, default=True,
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
        min_eval_frequency=200,  # define the evaluation interval
        min_summary_frequency=200,  # define checkpoints save interval
        log_steps=200,  # define the log print out interval
        learning_rate=FLAGS.learning_rate,  # define learning rate
        warmup_steps=FLAGS.warmup_steps  # define the learning_rate warmup steps
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 2. ----------------------------define the training and evaluation env parameters----------------------------------
    # RunConfig is used to set the env parameters, like model saving dir, checkpoints save interval, etc.
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=params.min_eval_frequency,
        save_summary_steps=params.min_summary_frequency,
        log_step_count_steps=params.log_steps)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. ------------------------------------------Define Estimator-----------------------------------------------------
    # the standard estimator architecture, almost has no any need to modify it, in almost all of
    # the machine learning process.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 4. ----------------------------------prepare the training and evluation specs-------------------------------------
    # get the training, evaluation and prediction features and labels.
    # different task has different data architecture, so PLEASE take more care about the data type, dimension, etc.

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
            temp = ''
            for word in each.strip().split():
                if word != EOS:
                    temp += word + ' '
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

    # self-defined Accuracy function, to be used for self accuracy generating
    print("Accuracy:", accuracy(pred_out, pred_label))


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    # Loss, training and eval operations are not needed during inference.
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
        # features has shape of [B, T]
        features, feature_len = features
        labels_in, label_out, label_len = labels

        tgt_seq_mask_original = tf.cast(tf.sequence_mask(label_len, FLAGS.max_output_time_steps), tf.float32)  # [B,T]
        tgt_seq_mask = tf.expand_dims(tgt_seq_mask_original, axis=2)  # [B,T,1]

        label_one_hot = tf.one_hot(indices=label_out, depth=FLAGS.output_vocab_size, axis=-1)  # [B, T, V]
        label_one_hot_mask = label_one_hot * tgt_seq_mask  # only use the valuable data, 0 mask the non-valuable data
        logits = architecture(features, feature_len, labels_in, label_len, mode)
        logits_mask = logits * tgt_seq_mask  # only use the valuable data, 0 mask the non-valuable data
        predictions = tf.nn.softmax(logits) * tgt_seq_mask  # only use the valuable data, 0 mask the non-valuable data

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_smoothing(label_one_hot_mask, FLAGS.label_smooth), logits=logits_mask)
        loss = tf.reduce_sum(loss) / tf.count_nonzero(tgt_seq_mask, dtype=tf.float32)

        # train_op = get_train_op_fn(loss, params)
        train_op = get_train_op(loss, params)
        # only use the valuable data, 0 mask the non-valuable data
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
        # Get learning rate warmup.
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

    # compute gradients for params
    gradients = tf.gradients(loss, trainable_params, colocate_gradients_with_ops=True)

    # process gradients
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

    def normalization(inputs):

        epsilon = 1e-8
        params_shape = inputs.get_shape().as_list()[-1]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        zeros = lambda: tf.zeros([params_shape], dtype=tf.float32)
        beta = tf.Variable(
            initial_value=zeros,
            name='beta',
            dtype=tf.float32)

        ones = lambda: tf.ones([params_shape], dtype=tf.float32)
        gamma = tf.Variable(
            initial_value=ones,
            name='gamma',
            dtype=tf.float32)

        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

        return outputs

    def input_embedding(inputs, vocab_size, num_units):

        with tf.variable_scope(name_or_scope='word_embedding', reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embedding', shape=[vocab_size, num_units], dtype=tf.float32,
                                        initializer=tf.contrib.layers.variance_scaling_initializer())

            embedding = tf.concat((tf.zeros([1, num_units]), embedding[1:, :]), axis=0)

        outputs = tf.nn.embedding_lookup(embedding, inputs)
        return outputs

    def position_embedding(inputs, str_length, num_units):
        # input shape is [B,T,H]
        position_table = np.full(shape=[str_length, num_units], fill_value=0.0, dtype=np.float32)

        for pos in range(str_length):
            for i in range(num_units):
                if i % 2 == 0:
                    position_table[pos, i] = np.sin(pos / np.power(10000, 2.0 * i / num_units))
                else:
                    position_table[pos, i] = np.cos(pos / np.power(10000, 2.0 * (i - 1) / num_units))

        position_table = tf.convert_to_tensor(position_table)

        step = tf.reshape(tf.range(str_length), shape=[1, str_length])
        batch_step = tf.tile(step, [tf.shape(inputs)[0], 1])
        outputs = tf.nn.embedding_lookup(position_table, batch_step)
        return outputs

    def step_embedding(inputs, current_step, max_steps, num_units):
        act_step_table = np.full(shape=[max_steps, num_units], fill_value=0.0, dtype=np.float32)

        for act_step in range(max_steps):
            for i in range(num_units):
                if i % 2 == 0:
                    act_step_table[act_step, i] = np.sin(act_step / np.power(10000, 2.0 * i / num_units))
                else:
                    act_step_table[act_step, i] = np.cos(act_step / np.power(10000, 2.0 * (i - 1) / num_units))

        act_step_table = tf.convert_to_tensor(act_step_table)

        batch_steps = tf.fill(tf.shape(inputs)[:-1], current_step)

        outputs = tf.nn.embedding_lookup(act_step_table, batch_steps)
        return outputs

    def feed_forward(inputs):
        outputs = tf.layers.dense(inputs=inputs, units=4 * FLAGS.num_units)
        outputs = tf.layers.dense(inputs=outputs, units=FLAGS.num_units)

        outputs = outputs + inputs

        return outputs

    def projection_layer(inputs):
        with tf.variable_scope(name_or_scope='projection', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs=inputs, units=FLAGS.output_vocab_size)
        return outputs

    def multihead_attention(query_inputs, key_inputs, value_inputs, query_mask, key_mask, scope):

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(query_inputs, FLAGS.num_units, activation=tf.nn.relu)
            K = tf.layers.dense(key_inputs, FLAGS.num_units, activation=tf.nn.relu)
            V = tf.layers.dense(value_inputs, FLAGS.num_units, activation=tf.nn.relu)

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
            # mask
            outputs *= query_mask
            # Normalization
            outputs = normalization(outputs)
            # mask
            outputs *= query_mask

        return outputs

    def transition_function(inputs, query_mask, scope):

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            outputs = feed_forward(inputs)
            # Dropout
            outputs = tf.layers.dropout(outputs, rate=FLAGS.dropout, training=(mode == ModeKeys.TRAIN))
            # mask
            outputs *= query_mask
            # Normalization
            outputs = normalization(outputs)
            # mask
            outputs *= query_mask

        return outputs

    def encoder_act_unit(query_inputs, query_mask, key_mask, current_step):
        ## Positional Encoding
        query_inputs += position_embedding(query_inputs,
                                           str_length=FLAGS.max_input_time_steps,
                                           num_units=FLAGS.num_units)

        # timesteps embedding
        query_inputs += step_embedding(inputs=query_inputs,
                                       current_step=current_step,
                                       max_steps=FLAGS.encoder_maxsteps,
                                       num_units=FLAGS.num_units)

        # multi_attention
        outputs = multihead_attention(query_inputs=query_inputs,
                                      key_inputs=query_inputs,
                                      value_inputs=query_inputs,
                                      query_mask=query_mask,
                                      key_mask=key_mask,
                                      scope='encoder_mul_attention')

        # transition_function
        outputs = transition_function(inputs=outputs,
                                      query_mask=query_mask,
                                      scope='encoder_transition')

        return outputs

    def decoder_act_unit(enc_outputs, query_inputs, dec_query_mask, dec_key_mask, enc_key_mask, current_step, str_length):
        ## Positional Encoding
        query_inputs += position_embedding(query_inputs,
                                           str_length=str_length,
                                           num_units=FLAGS.num_units)
        # timesteps embedding
        query_inputs += step_embedding(inputs=query_inputs,
                                       current_step=current_step,
                                       max_steps=FLAGS.decoder_maxsteps,
                                       num_units=FLAGS.num_units)

        # self_multi_attention
        outputs = multihead_attention(query_inputs=query_inputs,
                                      key_inputs=query_inputs,
                                      value_inputs=query_inputs,
                                      query_mask=dec_query_mask,
                                      key_mask=dec_key_mask,
                                      scope='decoder_self_mul_attention')
        # enc_dec_mul_attention
        outputs = multihead_attention(query_inputs=outputs,
                                      key_inputs=enc_outputs,
                                      value_inputs=enc_outputs,
                                      query_mask=dec_query_mask,
                                      key_mask=enc_key_mask,
                                      scope='enc_dec_mul_attention')
        # transition_function
        outputs = transition_function(inputs=outputs,
                                      query_mask=dec_query_mask,
                                      scope='decoder_transition')

        return outputs

    def act(inputs, query_mask, max_steps, processing_unit):

        def if_continue(_prev_outputs, _final_outputs, _prev_accum_halt, _go_sign, _current_step):

            go = tf.logical_and(tf.reduce_any(tf.greater(_go_sign, 0.0)), tf.less(_current_step, max_steps))
            return go

        def step_in(_prev_outputs, _final_outputs, _prev_accum_halt, _go_sign, _current_step):

            _processing_outputs = processing_unit(query_inputs=_prev_outputs, current_step=_current_step)
            _current_outputs = \
                _prev_outputs * tf.cast(tf.equal(_go_sign, 0.0), dtype=tf.float32) + \
                _processing_outputs * _go_sign
            _current_halt = tf.layers.dense(
                inputs=_current_outputs,
                units=1,
                activation=tf.nn.sigmoid,
                name='halt'
            )  # [B, T, 1]
            # _current_halt = tf.squeeze(_current_halt) # [B, T]

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

                _current_halt = (_current_halt * tf.cast(tf.less(_current_RNT_sign, 1.0),
                                                         dtype=tf.float32) + R_N_t) * _go_sign

                _current_accum_halt = (_prev_accum_halt + _current_halt) * _go_sign

                _final_outputs += _current_outputs * _current_halt

                _go_sign -= _current_RNT_sign

                _go_sign *= query_mask

                _current_step += 1

            return _current_outputs, _final_outputs, _current_accum_halt, _go_sign, _current_step

        prev_outputs = inputs  # [B,T,H]

        final_outputs = tf.zeros(tf.shape(inputs))  # [B,T,H]

        prev_accum_halt = tf.expand_dims(tf.zeros(tf.shape(inputs)[:-1]), axis=2)  # [B,T,1]

        go_sign = tf.expand_dims(tf.ones(tf.shape(inputs)[:-1]), axis=2) * query_mask # [B,T,1]

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

    def encoder(encoder_in, encoder_in_batch_length):

        q_mask = query_mask(encoder_in_batch_length, FLAGS.max_input_time_steps)
        k_mask = key_mask(encoder_in)

        with tf.variable_scope(name_or_scope='encoding', reuse=tf.AUTO_REUSE):
            ## Embedding
            enc = input_embedding(encoder_in,
                                  vocab_size=FLAGS.input_vocab_size,
                                  num_units=FLAGS.num_units)

            ## ACT
            processing_unit = functools.partial(
                encoder_act_unit,
                query_mask=q_mask,
                key_mask=k_mask
            )

            encoder_outputs = act(enc, q_mask, FLAGS.encoder_maxsteps, processing_unit)

        return encoder_outputs

    def decoder_train_eval(encoder_in, encoder_out):

        batch_size = tf.shape(encoder_in)[0]
        tgt_inputs = tf.cast(tf.sequence_mask([1], 1), tf.int32)
        tgt_inputs = tf.tile(tgt_inputs, [batch_size, 1])
        logits = None
        for tgt_len in trange(FLAGS.max_output_time_steps):
            mask = tf.cast(tf.sequence_mask([tgt_len + 1], tgt_len + 1), tf.int32)
            mask = tf.tile(mask, [batch_size, 1])
            decoder_in_batch_length = [tgt_len + 1]
            decoder_in_batch_length = tf.tile(decoder_in_batch_length, [batch_size])
            tgt_inputs = tgt_inputs * mask

            dec_q_mask = query_mask(decoder_in_batch_length, tgt_len + 1)
            dec_k_mask = key_mask(tgt_inputs)
            enc_k_mask = key_mask(encoder_in)

            with tf.variable_scope(name_or_scope='decoding', reuse=tf.AUTO_REUSE):
                ## Embedding
                dec = input_embedding(tgt_inputs,
                                      vocab_size=FLAGS.output_vocab_size,
                                      num_units=FLAGS.num_units)

                ## ACT
                processing_unit = functools.partial(
                    decoder_act_unit,
                    enc_outputs=encoder_out,
                    dec_query_mask=dec_q_mask,
                    dec_key_mask=dec_k_mask,
                    enc_key_mask=enc_k_mask,
                    str_length=tgt_len+1
                )

                decoder_outputs = act(dec, dec_q_mask, FLAGS.decoder_maxsteps, processing_unit)

            logits = projection_layer(decoder_outputs)  # [B,T,V]
            prediction_outputs = tf.nn.softmax(logits)  # [B,T,V]
            prediction_outputs = tf.argmax(prediction_outputs, axis=-1, output_type=tf.int32)  # [B,T]
            paddings = tf.constant([[0, 0], [1, 0]])
            tgt_inputs = tf.pad(prediction_outputs, paddings, "CONSTANT", constant_values=1)

        return logits

    def decoder_predict(encoder_in, encoder_out):

        batch_size = tf.shape(encoder_in)[0]
        tgt_inputs = tf.cast(tf.sequence_mask([1], 1), tf.int32)
        tgt_inputs = tf.tile(tgt_inputs, [batch_size, 1])

        prediction = None

        for tgt_len in trange(FLAGS.max_output_time_steps):
            mask = tf.cast(tf.sequence_mask([tgt_len + 1], tgt_len + 1), tf.int32)
            mask = tf.tile(mask, [batch_size, 1])
            decoder_in_batch_length = [tgt_len + 1]
            decoder_in_batch_length = tf.tile(decoder_in_batch_length, [batch_size])
            tgt_inputs = tgt_inputs * mask

            dec_q_mask = query_mask(decoder_in_batch_length, tgt_len + 1)
            dec_k_mask = key_mask(tgt_inputs)
            enc_k_mask = key_mask(encoder_in)

            with tf.variable_scope(name_or_scope='decoding', reuse=tf.AUTO_REUSE):
                ## Embedding
                dec = input_embedding(tgt_inputs,
                                      vocab_size=FLAGS.output_vocab_size,
                                      num_units=FLAGS.num_units)

                ## ACT
                processing_unit = functools.partial(
                    decoder_act_unit,
                    enc_outputs=encoder_out,
                    dec_query_mask=dec_q_mask,
                    dec_key_mask=dec_k_mask,
                    enc_key_mask=enc_k_mask,
                    str_length=tgt_len+1
                )

                decoder_outputs = act(dec, dec_q_mask, FLAGS.decoder_maxsteps, processing_unit)

            prediction = tf.nn.softmax(projection_layer(decoder_outputs))  # [B,T,V]
            prediction_outputs = tf.argmax(prediction, axis=-1, output_type=tf.int32)  # [B,T]
            paddings = tf.constant([[0, 0], [1, 0]])
            tgt_inputs = tf.pad(prediction_outputs, paddings, "CONSTANT", constant_values=1)

        return prediction

    enc_outputs = encoder(encoder_inputs, encoder_len)
    if mode != ModeKeys.PREDICT:
        logits = decoder_train_eval(encoder_inputs, enc_outputs)
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
    """Return the input function to get the training data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    initializer_hook = InitializerHook()

    def train_inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every training
        """
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
            # Filter zero length input sequences.
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

            # Return function and hook

    return train_inputs, initializer_hook


def get_eval_inputs(src, tgt, batch_size):
    """Return the input function to get the eval data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    initializer_hook = InitializerHook()

    def eval_inputs():
        """Returns eval set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """

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
            # Filter zero length input sequences.
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

            # Set runhook to initialize iterator
            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)

            return (next_feature, feature_len), (next_label_in, next_label_out, label_len)

    # Return function and hook
    return eval_inputs, initializer_hook


def get_predict_inputs(src, batch_size):
    """Return the input function to get the predict data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    # initializer_hook = pre_InitializerHook()
    initializer_hook = InitializerHook()

    def predict_inputs():
        """Returns prediction set as Operations.

        Returns:
            features Operations that iterate over the dataset
            on every inference
        """

        with tf.name_scope('Predict_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            if FLAGS.whitespace_or_nonws_slip:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src]).values)
            else:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src], delimiter='').values)
            # Filter zero length input sequences.
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