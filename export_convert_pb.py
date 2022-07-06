import sys
import tensorflow as tf

def frozen(export_dir):
    #export_dir = "./model-var-3.0"
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph,config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        for op in graph.get_operations():
            print(op.name,op.values())
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            #['output_label', 'predict_score', 'src_wid', 'src_mask']  # The output node names are used to select the usefull nodes
            ['output_label', 'predict_score']  # The output node names are used to select the usefull nodes
        )
        output_graph = export_dir+"/frozenmodel.pb"
        # # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
if __name__ == '__main__':
    export_dir = sys.argv[1]
    print("export_dir:"+export_dir)
    frozen(export_dir)
