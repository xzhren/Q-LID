from __future__ import print_function

import sys
lineNum=0
for line in sys.stdin:
    line = line.strip()
    print(line + '' + str(lineNum))
    lineNum+=1
