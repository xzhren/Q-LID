import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import tag_constants
import sys

from transformer import get_assignment_map_from_checkpoint
from transformer import GenerateSignature

from transformer import transformer_model_fn as model_fn
from config import *

for k in params.keys():
    if 'dropout' in k:
        params[k] = 0.0
print(params)

tf.logging.set_verbosity(tf.logging.INFO)

def export_single(model_dir, export_dir):
    #graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:  
        signature_def_map = GenerateSignature()
        saver_for_restore = tf.train.Saver(sharded=True)
        saver_for_restore.restore(sess, model_dir)
        
        tvars = tf.trainable_variables()
        #initialized_variable_names = {}
        #init_checkpoint = model_dir
        #(assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        ##print("assignment_map:", assignment_map)
        #tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        var_size = 0
        for var in tvars:
            init_string = ""
            #if var.name in initialized_variable_names:
            #    init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
            var_shape = var.shape.as_list()
            assert len(var_shape) <= 2
            var_size += var_shape[0] if len(var_shape) == 1 else var_shape[0] * var_shape[1]
        tf.logging.info(" variables size: %d ", var_size)
        #tf.train.init_from_checkpoint(model_dir)
     
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.SERVING],
                                       signature_def_map=signature_def_map,
                                       assets_collection=ops.get_collection(
                                       ops.GraphKeys.ASSET_FILEPATHS))
        builder.save()


#def serving_input_receiver_fn():
#    """Serving input_fn that builds features from placeholders
#
#    Returns
#    -------
#    tf.estimator.export.ServingInputReceiver
#    """
#    src_tensor_test_ph = tf.placeholder(tf.int64, [None, None], name="src_wid")
#    #receiver_tensors = {'src_wid': src_tensor_test_ph}
#    receiver_tensors = src_tensor_test_ph
#    return tf.estimator.export.ServingInputReceiver(src_tensor_test_ph, receiver_tensors)

if __name__ == '__main__':
    #model_dir = "./logs/langs23v8_trans_best/model.ckpt-442962"
    #model_dir = "./logs/langs23v8_trans_best/model.ckpt-196270"
    #model_dir = "./logs/langs23v8_trans_vocab_2l_net_best/model.ckpt-140001"
    #model_dir = "./logs/langs23v8_trans_best/"
    #export_dir = "./logs/langs23v8_trans_vocab_2l_net_best_export/"
    #model_dir, export_dir = sys.argv[1], sys.argv[2]
    
    export_dir = model_dir + "/export"
    export_model = model_dir + "/ema/model.ckpt-" + exportstep
    #print(sys.argv)
    print("export_model:",export_model,"export_dir:",export_dir)
    export_single(export_model, export_dir)

    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #session_config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True, log_device_placement=False)
    #transformer = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params, config=tf.estimator.RunConfig(session_config=session_config))
    #transformer.export_savedmodel(export_dir, serving_input_receiver_fn)
