import sys
import cPickle as pkl

if len(sys.argv) == 1:
    print 'python build_vocab.py train_file vocab_size pkl_file > txt_file'
    sys.exit()

d = {}
for s in open(sys.argv[1]):
    tokens = s.strip().split(' ')
    for w in tokens:
        d[w] = d.get(w,0) + 1
dd = {'</S>':0,'':1,'<UNK>':2}
idx = 2
for k,v in sorted(d.items(),key=lambda d:d[1],reverse=True):
    idx += 1
    dd[k] = idx
    if len(d) >= int(sys.argv[2]):
        break
pkl.dump(dd, open(sys.argv[3],'wb'))

d = pkl.load(open(sys.argv[3]))
for k,v in sorted(d.items(),key=lambda d:d[1]):
    #print k + "" + str(v)
    print k 
