#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2022 Alibaba Inc. All Rights Reserved.
#
# History:
# 2022.07.05. Be created by xingzhang.rxz. Used for Query Language Identification.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

supported_lang=set(["ar","zh","zh-tw","nl","en","fr","de","he","hi","id","it","ja","ko","ms","pl","pt","ru","es","th","tr","ug","uk","vi"])
supported_lang=set(["ar","zh","nl","en","fr","de","he","hi","id","it","ja","ko","pl","pt","ru","es","th","tr","vi"])
#supported_lang=set(['ar', 'ru', 'ko', 'ja', 'zh', 'hi', 'ug', 'he','th','en','es','pt','fr','it','id','vi','pl','nl','tr','de'])
# supported_lang=set(['en','es','pt','fr','it','id','vi','pl','nl','tr','de'])

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("python get_confusiont_matrix eval.out.x.res")
    resfile = sys.argv[1]

    label_dict = {}
    label_dict_reverse = {}
    for index, line in enumerate(supported_lang):
        line = line.strip()
        label_dict[index] = line
        label_dict_reverse[line] = index
        class_num = index+1

    confusion_matrix = [[0]*class_num for _ in range(class_num)]
    with open(resfile) as target:
        for line in target:
            infos = line.strip().split("\t")
            #if infos[2] == "he": infos[2] = "iw"
            #if infos[2] == "zhtw": infos[2] = "zh-tw"
            #if infos[1] == "he": infos[1] = "iw"
            #if infos[1] == "zhtw": infos[1] = "zh-tw"
            if infos[1] not in supported_lang: continue
            #if infos[1] not in label_dict_reverse: continue
            #if infos[2] not in label_dict_reverse: continue
            true_label, pred_label = label_dict_reverse[infos[1]], label_dict_reverse[infos[2]]
            confusion_matrix[true_label][pred_label] += 1

    right_cnt = 0
    for i in range(class_num):
        if sum(confusion_matrix[i]) != 0:
            acc = confusion_matrix[i][i]/sum(confusion_matrix[i])
        else:
            acc = -1
        right_cnt += confusion_matrix[i][i]
        # print(label_dict[i]+"\t"+"\t".join([str(_) for _ in confusion_matrix[i]])+"\n")
        print(label_dict[i]+"\t"+"\t".join([str(_) for _ in confusion_matrix[i]])+"\t"+format(acc*100, "0.2f"))
    acc = right_cnt/sum(map(sum, confusion_matrix))
    print("-\t"+"\t".join(label_dict.values())+"\t"+format(acc*100, "0.2f"))

        
