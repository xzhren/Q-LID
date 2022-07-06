#-*-coding=utf-8-*-
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
print >> sys.stderr, 'reporter:counter:MyCounter,EnterScript,1'

for line in sys.stdin:
    tokens = line.strip().lower().split('\t')
    #if len(tokens) <= 2: print line
    assert len(tokens) == 2
    text = tokens[1].decode('utf-8')
    # words = text
    for w in list(text):
        print '%s\t%d' % (w, 1)

print >> sys.stderr, 'reporter:counter:MyCounter,LeaveScript,1'
