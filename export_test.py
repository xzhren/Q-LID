#-*- coding:utf-8 -*-
import argparse
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="Langident",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozenmodel.pb", type=str, help="Frozen model file to import")

    args = parser.parse_args()
    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)
    #for op in graph.get_operations():
    #    print(op.name,op.values())

    input_data = [[19,13,5,9,3,2,6,7,8,12,9,6,3]]
    input_mask = [[1, 1, 1,1,1,1,1,1,1,1, 1,1,1]]
    input_data = [[19,285,471]]
    input_mask = [[1,1,1]]
    input_data = [[ 19,   8,   4,  10,   8],[ 19,   8,   4,  10,   8],[ 19,   8,   4,  10,   8],[ 19,   8,   4,  10,   8],[ 19,   8,   4,  10,   8],[ 19,   8,   4,  10,   8]]
    input_mask = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
    #input_data = [[4386,0,6,0,532,40393,514,656,1072,6666,1670,42000,514,485,1072,1564,1670,519,4,48,11156,4446,4,638,0,2818,24852,4,11908,16,759,0,743,25,759,0,743,25,759,545,150,743,25,759,1851,986,150,743,25,759]]
    #input_length = [50]
    #input_padding_mask = [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]
    #enc_batch_extend_vocab = [[4386,50000,6,50000,532,40393,514,656,1072,6666,1670,42000,514,485,1072,1564,1670,519,4,48,11156,4446,4,638,50001,2818,24852,4,11908,16,759,50002,743,25,759,50003,743,25,759,545,150,743,25,759,1851,986,150,743,25,759]]
    #max_art_oovs = [4]
    # 模型的五个输入参数
    prefix = "Langident/"
    y = graph.get_tensor_by_name(prefix+'output_label:0')
    #y = graph.get_tensor_by_name(prefix+'ArgMax:0')
    score = graph.get_tensor_by_name(prefix+'predict_score:0')
    #score = graph.get_tensor_by_name(prefix+'NmtModel/Softmax:0')
    x1 = graph.get_tensor_by_name(prefix+'src_wid:0')
    #x2 = graph.get_tensor_by_name(prefix+'src_mask:0')
    #x1 = graph.get_tensor_by_name(prefix+'Placeholder:0')
    #x2 = graph.get_tensor_by_name(prefix+'Placeholder_1:0')

    with tf.Session(graph=graph) as sess:
        result = sess.run(fetches=[y,score], feed_dict = {x1:input_data})
    print(result)
