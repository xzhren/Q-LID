#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2022 Alibaba Inc. All Rights Reserved.
#
# History:
# 2022.07.05. Be created by xingzhang.rxz. Used for Query Language Identification.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python get_confusiont_matrix eval.out.x.res exp1.label.txt")
    resfile = sys.argv[1]
    labelfile = sys.argv[2]

    label_dict = {}
    label_dict_reverse = {}
    with open(labelfile) as target:
        for index, line in enumerate(target):
            line = line.strip()
            label_dict[index] = line
            label_dict_reverse[line] = index
            class_num = index+1

    confusion_matrix = [[0]*class_num for _ in range(class_num)]
    with open(resfile) as target:
        for line in target:
            infos = line.split("\t")
            true_label, pred_label = label_dict_reverse[infos[1]], label_dict_reverse[infos[2]]
            confusion_matrix[true_label][pred_label] += 1

    right_cnt = 0
    for i in range(class_num):
        acc = confusion_matrix[i][i]/sum(confusion_matrix[i])
        right_cnt += confusion_matrix[i][i]
        # print(label_dict[i]+"\t"+"\t".join([str(_) for _ in confusion_matrix[i]])+"\n")
        print(label_dict[i]+"\t"+"\t".join([str(_) for _ in confusion_matrix[i]])+"\t"+format(acc*100, "0.2f"))
    acc = right_cnt/sum(map(sum, confusion_matrix))
    print("-\t"+"\t".join(label_dict.values())+"\t"+format(acc*100, "0.2f"))

        