#-*-coding=utf8-*-
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

for line in sys.stdin:
    #line = line.strip().decode('utf-8', errors="ignore")
    line = line.strip().decode('utf-8')
    tokens = line.split('\t')
    if len(tokens) != 2:
        continue
    text = tokens[0].lower()   #input odps table should have same format
    lang = tokens[1].lower()   #clean_text processd by  lg_text_clean udf

    txt_len = len(text.split(" "))
    if txt_len > 500:
        continue
    if txt_len < 5:
        continue

    print text + '\t' + lang
