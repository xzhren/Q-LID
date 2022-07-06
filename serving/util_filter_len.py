#-*-coding=utf8-*-
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


for index, line in enumerate(sys.stdin):
    line = line.strip().decode('utf-8')
    tokens = line.split('\t')
    if len(tokens) != 2:
        continue
    lang = tokens[0].lower()   #input odps table should have same format
    text = tokens[1].lower()   #clean_text processd by  lg_text_clean udf
   

    if len(text) < 500: print(line.rstrip())
