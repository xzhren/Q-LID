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
    lang = tokens[0].lower()   #input odps table should have same format
    text = tokens[1].lower()   #clean_text processd by  lg_text_clean udf
    # domain = tokens[2]
    # only for latin language family, we filter characters that not belong to latin.
    #text = re.sub(ur'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', text)
    
    if len(text) > 500: continue

    new_line = " "
    for s in text:
        if s == ' ':
            new_line += ' '
        else:
            new_line += s + ' '
    new_line = new_line[0:-1]
    print new_line + '\t' + lang
