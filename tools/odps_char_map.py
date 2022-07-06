#-*-coding=utf-8-*-
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
print >> sys.stderr, 'reporter:counter:MyCounter,EnterScript,1'

for line in sys.stdin:
    tokens = line.strip().split('\t')
    text = tokens[0].decode('utf-8')
    words = text.split(' ')
    for w in words:
        print '%s\t%d' % (w, 1)

print >> sys.stderr, 'reporter:counter:MyCounter,LeaveScript,1'
