#-*-coding=utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

cur_word = None
cur_count = 0
for line in sys.stdin:
    tokens = line.strip().split('\t')
    if len(tokens) != 2:
        continue
    word,count = tokens
    if word == cur_word:
        cur_count += int(count)
    else:
        if cur_word is not None:
            print '%s\t%d' % (cur_word, cur_count)
        cur_word = word
        cur_count = int(count)

if cur_word is not None:
    print '%s\t%d' % (cur_word, cur_count)

