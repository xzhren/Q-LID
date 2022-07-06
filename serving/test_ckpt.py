# coding=utf-8
import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import tensorflow as tf
import time

from transformer import GenerateSignature
from config import *

VOCAB_SIZE = 26000
supported_lang = set(["ar","zh","zh-tw","nl","en","fr","de","he","hi","id","it","ja","ko","ms","pl","pt","ru","es","th","tr","ug","uk","vi"])

vocabs = None
with open(vocab_src) as vocabfile:
    vocabs = [line.strip() for line in vocabfile.readlines()]
vocabs = vocabs[:VOCAB_SIZE]
print("load vocab size: {}".format(str(len(vocabs))))

labels = None
with open(vocab_trg) as labelfile:
    labels = [line.strip() for line in labelfile.readlines()]
print("load label size: {}".format(str(len(labels))))

vocab2id = {}
id2vocab = {}
for index, v in enumerate(vocabs):
    vocab2id[v] = index
    id2vocab[index] = v
print("example: vocab2id {} id2vocab {}".format(str(list(vocab2id.items())[:5]), str(list(id2vocab.items())[:5])))

START_TOKEN = vocab2id['']
SPACE_TOKEN = vocab2id['']
UNK_TOKEN = vocab2id['<UNK>']
# MASK_TOKEN = vocab2id['']
PAD_TOKEN = vocab2id['</S>']
print("START_TOKEN:", START_TOKEN, id2vocab[START_TOKEN], "SPACE_TOKEN:", SPACE_TOKEN, id2vocab[SPACE_TOKEN], "UNK_TOKEN:", UNK_TOKEN, id2vocab[UNK_TOKEN], "PAD_TOKEN:", PAD_TOKEN, id2vocab[PAD_TOKEN])

def convert_one_sample(text_item, vocab2id):
    wids = [START_TOKEN]
    text_item = text_item.lower()
    for word in text_item:
        if word == " ": tmp = SPACE_TOKEN
        else: tmp = UNK_TOKEN if word not in vocab2id else vocab2id[word]
        wids.append(tmp)
    wids.append(PAD_TOKEN)
    return wids, len(wids)

def pre_processor(text, vocab2id):
    if type(text) == str:
        wids, pad_size = convert_one_sample(text, vocab2id)
        return np.array([wids]), pad_size, 1
    if type(text) == list:
        widslt = []
        for text_item in text:
            wids, item_pad_size = convert_one_sample(text_item, vocab2id)
            if item_pad_size > pad_size: pad_size = item_pad_size
            widslt.append(wids)
        
        for wids in widslt:
            if len(wids) < pad_size:
                wids.extend([PAD_TOKEN]* (pad_size - len(wids)) )

        return np.array(widslt), pad_size, len(text)
    else:
        return None # raise 

def post_processor(output, output_score, labels, batch_size): 
    final_out = {}
    for i in range(batch_size):
        tmp_res = output[i]
        tmp_score = output_score[i]
        output_label = labels[tmp_res]
       
        prob_index = np.argsort(tmp_score)[::-1]
        prob_value = np.sort(tmp_score)[::-1]
        prob_index_lang = [labels[item] for item in prob_index]
        if supported_lang != None and prob_index_lang[0] not in supported_lang:
            for pid, prob in zip(prob_index_lang, prob_value):
                if pid in supported_lang:
                     prob_index_lang[2], prob_value[2] = prob_index_lang[1], prob_value[1]
                     prob_index_lang[1], prob_value[1] = prob_index_lang[0], prob_value[0]
                     prob_index_lang[0], prob_value[0] = pid, prob
                     break
        prob_index_lang = prob_index_lang[:3]
        prob_value = prob_value[:3]

        predict_score = "\t".join(["%s\t%.2f" %(l,s*100) for l, s in zip(prob_index_lang, prob_value)])
        # tmp_output = "{}\t{}".format(output_label, predict_score)
        # final_out += tmp_output + "\n"
        final_out["label-"+str(i)] = output_label
        final_out["score-"+str(i)] = predict_score
        print(predict_score)
    return final_out

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    print("python main.py export_dir")
    #    exit()

    #export_dir = sys.argv[1]
    export_dir = model_dir + "/model.ckpt-" + exportstep
    print("export_dir: {}".format(export_dir))

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        #tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        signature_def_map = GenerateSignature()
        #saver = tf.train.import_meta_graph(export_dir+'.meta')
        saver = tf.train.Saver()
        saver.restore(sess, export_dir)
        #for op in graph.get_operations():
        #    print(op.name,op.values())
        
        #y = graph.get_tensor_by_name('output_label:0')
        #score = graph.get_tensor_by_name('predict_score:0')
        #x = graph.get_tensor_by_name('src_wid:0')
        y = signature_def_map['output_label']
        score = signature_def_map['predict_score']
        x = signature_def_map['src_wid']
        if use_script_embedding or use_word_embedding:
            x_s = signature_def_map['src_sid']

        print("please input text ...")
        for text in sys.stdin:
            input_data, pad_size, batch_size = pre_processor(text.strip(), vocab2id) # seg + w2id + return feed_dict
            print(input_data, pad_size, batch_size)

            s0 = time.time()
            output, output_score = sess.run(fetches=[y,score], feed_dict = {x:input_data})
            s1 = time.time()

            result = post_processor(output, output_score, labels, batch_size)
            print(result)
            print("query time: {} ms, text lens: {} token".format(100*(s1 - s0), pad_size))
