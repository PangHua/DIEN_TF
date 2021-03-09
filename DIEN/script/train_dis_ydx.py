import os
import numpy
import collections
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
import json
from utils import *
import pdb

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--model", type=str, default='DIEN', help="model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
parser.add_argument("--num_accelerators", type=int, default=1, help="number of accelerators used for training")
parser.add_argument("--embedding_device", type=str, default='gpu', help="synthetic input embedding layer reside on gpu or cpu")
parser.add_argument("--saved_models", type=str, default='/workspace/save_models/', help="Saved path for models")
args = parser.parse_args()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2 * 128
ATTENTION_SIZE = 18 * 2 * 128
best_auc = 0.0

#TOTAL_TRAIN_SIZE = 256000
TOTAL_TRAIN_SIZE = 128000

INT_DATA_TYPE = 'int32'
FLOAT_DATA_TYPE = 'float32'


def prepare_data(input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    #pdb.set_trace()
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None
    
    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])


    mid_his = numpy.zeros((n_samples, maxlen_x)).astype(INT_DATA_TYPE)
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype(INT_DATA_TYPE)
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype(INT_DATA_TYPE)
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype(INT_DATA_TYPE)
    if args.data_type == 'FP32':
        data_type = 'float32'
    elif args.data_type == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)

def eval(sess, test_data, model, model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0
    for src, tgt in test_data:
        nums += 1
        sys.stdout.flush()
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        # print("begin evaluation")
        start_time = time.time()
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
        end_time = time.time()
        # print("evaluation time of one batch: %.3f" % (end_time - start_time))
        # print("end evaluation")
        eval_time += end_time - start_time
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        # print("nums: ", nums)
        # break
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        if args.mode == 'train':
            model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, eval_time, nums

def train_synthetic(   
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2,
        n_uid = 543060,
        n_mid = 100000 * 300,
        n_cat = 1601,
        embedding_device = 'gpu'      
):
    print("batch_size: ", batch_size)
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    synthetic_input = True
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
        # parameters needs to put in config file
       
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type = data_type, 
            synthetic_input = synthetic_input, batch_size = batch_size, max_length = maxlen, device = embedding_device)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type, 
            synthetic_input = synthetic_input, batch_size = batch_size, max_length = maxlen, device = embedding_device)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        
        iter = 0
        train_size = 0
        approximate_accelerator_time = 0

        for itr in range(1):
            for i in range(500):   
                start_time = time.time()
                _, _, _ = model.train_synthetic_input(sess)
                end_time = time.time()
                # print("training time of one batch: %.3f" % (end_time - start_time))
                one_iter_time = end_time - start_time   
                approximate_accelerator_time += one_iter_time                 
                iter += 1
                sys.stdout.flush()
                if (iter % 100) == 0:
                    print('iter: %d ----> speed: %.4f  QPS' % 
                                        (iter, 1.0 * batch_size /one_iter_time ))    
         
        print("Total recommendations: %d" % (iter * batch_size))
        print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
        print("Approximate accelerator performance in recommendations/second is %.3f" % (float(iter * batch_size)/float(approximate_accelerator_time)))


@tf.function
def input_fn_v2(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, training):
    batchsize = 128
    features = collections.OrderedDict()
    feature_keys = ['uids', 'mids', 'cats', 'mid_his', 'cat_his', 'mid_mask', 'sl', 'noclk_mids', 'noclk_cats']
    def get_generator():
        train_generator = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
        for src, tgt in train_generator:
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
            features['uids'] = uids
            features['mids'] = mids
            features['cats' ] = cats
            features['mid_his']= mid_his
            features['cat_his'] = cat_his
            features['mid_mask'] = mid_mask
            features['sl'] = sl
            features['noclk_mids'] = noclk_mids
            features['noclk_cats'] = noclk_cats
            '''
            print("features uids:{}".format(features['uids']))
            print("features mids:{}".format(features['uids']))
            print("features cats:{}".format(features['uids']))
            print("features mid_his:{}".format(features['mid_his']))
            print("features cat_his:{}".format(features['cat_his']))
            print("features mid_mask:{}".format(features['mid_mask']))
            print("features sl:{}".format(features['sl']))
            print("features noclk_mids:{}".format(features['noclk_mids']))
            print("features noclk_cats:{}".format(features['noclk_cats']))
            print("target:{}".format(target))
            print("features:{}".format(features))
            '''
            #features =[uids, mids, cats, mid_his, cat_his, mid_mask, sl, noclk_mids, noclk_cats]
            yield features, target
    output_types = ({k: tf.int32 for k in feature_keys}, tf.float32)

    features_shape = [None]
    features_shape_2 = [None, 5]
    labels_shape = [2]
    output_shapes = ({'uids':tf.TensorShape([]),
                      'mids':tf.TensorShape([]),
                      'cats':tf.TensorShape([]),
                      'mid_his': tf.TensorShape(features_shape),
                      'cat_his': tf.TensorShape(features_shape),
                      'mid_mask': tf.TensorShape(features_shape),
                      'sl': tf.TensorShape([]),
                      'noclk_mids': tf.TensorShape(features_shape_2),
                      'noclk_cats': tf.TensorShape(features_shape_2)},
                     tf.TensorShape(labels_shape))

    dataset = tf.data.Dataset.from_generator(get_generator,output_types, output_shapes).batch(1)
    iterator = tf.data.make_initializable_iterator(dataset)
    #iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    #x,y = iterator.get_next()
    return iterator.get_next()


def model_fn(features, labels, mode, params):
    training = mode == tf.estimator.ModeKeys.TRAIN

    n_uid, n_mid, n_cat = params['n_uid'], params['n_mid'], params['n_cat']
    model_type, batch_size, maxlen, data_type = params['model_type'], params['batch_size'], params['maxlen'], params['data_type']
    print("Number of uid = %i, mid = %i, cat = %i" % (n_uid, n_mid, n_cat)) #Number of uid = 543060, mid = 367983, cat = 1601 for Amazon dataset
    if model_type == 'DNN':
        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type = data_type, 
        batch_size = batch_size, max_length = maxlen)
    elif model_type == 'PNN':
        model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'Wide':
        model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN':
        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIEN':
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type, 
        batch_size = batch_size, max_length = maxlen)
    else:
        print ("Invalid model_type : %s", model_type)
        return

    model(features, labels)
    lr = 0.001
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(model.loss, global_step=tf.train.get_global_step())

    logging_hook = tf.estimator.LoggingTensorHook({"loss" : model.loss, "accuracy" : model.accuracy}, every_n_iter=10, 
                                                  every_n_secs=None, at_end=False, formatter=None)
    profiling_hook = tf.train.ProfilerHook(save_steps=200, output_dir='/workspace/timeline/', show_dataflow=True, show_memory=True)
    return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op, training_hooks=[logging_hook, profiling_hook],scaffold=None)
    #return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op, training_hooks=[logging_hook],scaffold=None)


@tf.function
def input_fn(features, y, read_samples_per_batch, shuffle):
    return tf.estimator.inputs.numpy_input_fn(x=features, y=y, batch_size=read_samples_per_batch, shuffle=shuffle)

def preprocess(train_data, maxlen):
    features = collections.OrderedDict()
    tgt_list = []
    feature_keys = ['uids', 'mids', 'cats', 'mid_his', 'cat_his', 'mid_mask', 'sl', 'noclk_mids', 'noclk_cats']
    
    for k in feature_keys:
        features[k] = []
    for src, tgt in train_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
        features['uids'].append(uids)
        features['mids'].append(mids)
        features['cats' ].append(cats)
        features['mid_his'].append(mid_his)
        features['cat_his'].append(cat_his)
        features['mid_mask'].append(mid_mask)
        features['sl'].append(sl)
        features['noclk_mids'].append(noclk_mids)
        features['noclk_cats'].append(noclk_cats)
        tgt_list.append(tgt)
        break

    for k in features:
        features[k] = np.squeeze(np.vstack(features[k]))
        if features[k].dtype == 'float64':
            features[k] = np.float32(features[k])
        if features[k].dtype == 'int64':
            features[k] = np.int32(features[k])

        y = np.squeeze(np.vstack(tgt_list))
        if y.dtype == 'int64':
            y = np.int32(y)
        if y.dtype == 'float64':
            y = np.float32(y)

    return features, y

def input_fn_v3(train_data, maxlen, batch_size=128):
    def train_input_fn():
        SHUFFLE_BUFFER_SIZE = 8192
        features, y = preprocess(train_data, maxlen)
        features_dataset = tf.data.Dataset.from_tensor_slices(features)
        labels_dataset = tf.data.Dataset.from_tensor_slices(y)
        train_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
        #train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        train_dataset = train_dataset.batch(batch_size=batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=SHUFFLE_BUFFER_SIZE)
        iterator = train_dataset.make_one_shot_iterator()
        #iterator = tf.data.make_initializable_iterator(train_dataset)
        #iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
        return iterator.get_next()
    return train_input_fn

def train(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2,
	saved_models = args.saved_models):
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)

    end_2_end_start_time = time.time()

    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    if tf_config['task']['type'] == 'ps':
        read_total_size = 128 #TOTAL_TRAIN_SIZE #1086120
    else:
        read_total_size = TOTAL_TRAIN_SIZE #1086120

    print("batch_size: ", batch_size)
    print("Trainning size:{}".format(read_total_size))

    print("Loading and preprocessing data ...")
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, read_total_size, maxlen, shuffle_each_epoch=False)
    n_uid, n_mid, n_cat = train_data.get_n()
    print("n_uid:{}, n_mid:{}, n_cat:{}".format(n_uid, n_mid, n_cat))

    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    ps_nodes = tf_config.get("cluster", {})['ps']
    worker_nodes = tf_config.get("cluster", {})['worker']
    task_env = tf_config.get("task", {})
    task_type = task_env.get("type", {})
    task_index = task_env.get("index", {})
    rpc_layer = tf_config.get("rpc_layer", {})
    if task_type == 'ps':
        gpu_count = 0
    else:
        gpu_count = len(worker_nodes)
    num_accelerators = {"GPU": gpu_count}

    '''
    import pdb
    pdb.set_trace()
    cluster_spec = tf.train.ClusterSpec({
        "ps": ps_nodes,
        "worker": worker_nodes,
        })
    simple_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, task_type=task_type, 
                                                                           task_id=task_index, 
                                                                           num_accelerators=num_accelerators,
                                                                           rpc_layer=rpc_layer)

    # DistributedStrategy
    #strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver=simple_resolver)
    '''
    strategy = tf.distribute.experimental.ParameterServerStrategy()

    # Create Estimator
    config = {}
    config['total_steps'] = 40000
    config['n_uid'] = n_uid
    config['n_mid'] = n_mid
    config['n_cat'] = n_cat
    config['maxlen'] = maxlen
    config['model_type'] = model_type
    config['data_type'] = data_type
    config['batch_size']= batch_size

    run_config = tf.estimator.RunConfig(train_distribute=strategy, 
                                        session_config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=True),
                                        protocol='grpc+verbs')
    estimator = tf.estimator.Estimator(
        model_dir=saved_models,
        model_fn=model_fn,
        config=run_config,
        params=config)

    # The default is 128
    read_samples_per_batch = batch_size 
    start_time = time.time()

    # Training only
    #train_input_fn = tf.estimator.inputs.numpy_input_fn(x=features, y=y, batch_size=read_samples_per_batch, shuffle=True)
    #evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(x=features, y=y, batch_size=read_samples_per_batch, shuffle=False)
    #estimator.train(input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None) 

    #features, y = preprocess(train_data, maxlen)


    train_spec = tf.estimator.TrainSpec(input_fn= lambda: input_fn_v3(train_data, maxlen, batch_size=batch_size), max_steps=config["total_steps"])
    eval_spec = tf.estimator.EvalSpec(input_fn= lambda: input_fn_v3(train_data, maxlen, batch_size=batch_size))

    '''
    # Train and Evaluate
    train_spec = tf.estimator.TrainSpec(input_fn= lambda: tf.estimator.inputs.numpy_input_fn(x=features, y=y, batch_size=read_samples_per_batch, shuffle=True),max_steps=config["total_steps"])
    eval_spec = tf.estimator.EvalSpec(input_fn= lambda: tf.estimator.inputs.numpy_input_fn(x=features, y=y, batch_size=read_samples_per_batch, shuffle=False))
    '''

    # Generator Input
    #train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, True),
    #                                    max_steps=config["total_steps"])
    #eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen,False))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    end_time = time.time()
    approximate_accelerator_time = end_time - start_time
    end2end_time = end_time - end_2_end_start_time
    print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
    print("Approximate end2end accelerator time in seconds is %.3f" % end2end_time)
    print("Approximate accelerator performance in recommendations/second is %.3f" % (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))


def train_generator(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        data_type = 'FP32',
        seed = 2,
	saved_models = args.saved_models):
    print("model: ", model_type)
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)

    end_2_end_start_time = time.time()

    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    if tf_config:
        if tf_config['task']['type'] == 'ps':
            read_total_size = 128 #TOTAL_TRAIN_SIZE #1086120
        else:
            read_total_size = 128 #TOTAL_TRAIN_SIZE #1086120
    else:
        read_total_size = 128 #TOTAL_TRAIN_SIZE #1086120

    print("batch_size: ", batch_size)
    print("Trainning size:{}".format(read_total_size))

    print("Loading and preprocessing data ...")
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, read_total_size, maxlen, shuffle_each_epoch=False)
    n_uid, n_mid, n_cat = train_data.get_n()
    print("n_uid:{}, n_mid:{}, n_cat:{}".format(n_uid, n_mid, n_cat))

    # DistributedStrategy
    strategy = tf.distribute.experimental.ParameterServerStrategy()

    # Create Estimator
    config = {}
    config['total_steps'] = 40000
    config['n_uid'] = n_uid
    config['n_mid'] = n_mid
    config['n_cat'] = n_cat
    config['maxlen'] = maxlen
    config['model_type'] = model_type
    config['data_type'] = data_type
    config['batch_size']= batch_size

    run_config = tf.estimator.RunConfig(train_distribute=strategy, session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    estimator = tf.estimator.Estimator(
        model_dir=saved_models,
        model_fn=model_fn,
        config=run_config,
        params=config)

    # The default is 128
    read_samples_per_batch = batch_size 
    start_time = time.time()

    # Generator Input
    def train_input_fn():
        return input_fn_v2(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, True)
    
    def eval_input_fn():
        return input_fn_v2(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen,False)

    # Training only
    estimator.train(input_fn=lambda: train_input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None) 

    '''
    # Train and Evaluate
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn, max_steps=config["total_steps"])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    '''

    end_time = time.time()
    approximate_accelerator_time = end_time - start_time
    end2end_time = end_time - end_2_end_start_time
    print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
    print("Approximate end2end accelerator time in seconds is %.3f" % end2end_time)
    print("Approximate accelerator performance in recommendations/second is %.3f" % (float(TOTAL_TRAIN_SIZE)/float(approximate_accelerator_time)))


if __name__ == '__main__':
    SEED = args.seed
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type, saved_models=args.saved_models)
    elif args.mode == 'train_generator':
        train_generator(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type, saved_models=args.saved_models)
    elif args.mode == 'test':
        test(model_type=args.model, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
    elif args.mode == 'synthetic':
        train_synthetic(model_type=args.model, seed=SEED, batch_size=args.batch_size,
        data_type=args.data_type, embedding_device = args.embedding_device
        )
    else:
        print('do nothing...')
