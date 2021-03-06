#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import print_function
from __future__ import unicode_literals, division

import sys
import codecs
import argparse
from collections import defaultdict

# hack for python2/3 compatibility
from io import open
argparse.open = open

# python 2/3 compatibility
#if sys.version_info < (3, 0):
#  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
#  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
#  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
#else:
#  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
#  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
#  sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

import codecs

class BPE(object):

    def __init__(self, codes, separator='@@'):

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])
        #bpe_codes format: key:code, value:index
        self.separator = separator
        self.cache = {}

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = []
        for word in sentence.split():
            new_word = encode(word, self.bpe_codes, self.cache)
            #print  "====================="
            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return ' '.join(output)

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--src_codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--tgt_codes', '-t', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")

    return parser

def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def encode(orig, bpe_codes, cache={}):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    word = tuple(orig) + ('</w>',) # split word into characters
    pairs = get_pairs(word)
    #for x in pairs:
    #    print x[0]
    #    print x[1]


    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        #print "bigram:", bigram
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                #print "i:%d, j:%d, new_word:%s" % (i,j,word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                #print "append 1:", first+second
                i += 2
            else:
                new_word.append(word[i])
                #print "append 2: word[i]"
                i += 1
        new_word = tuple(new_word)
        word = new_word
        #print(word)
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    cache[orig] = word
    return word


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # read/write files as UTF-8
    src_codes = codecs.open(args.src_codes.name, encoding='utf-8')
    tgt_codes = codecs.open(args.tgt_codes.name, encoding='utf-8')
    src_bpe = BPE(src_codes, args.separator)
    tgt_bpe = BPE(tgt_codes, args.separator)

    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    for line in args.input:
        parts = line.strip().split("\t")
        if len(parts) < 2: continue
        src = parts[0].strip()
        tgt = parts[1].strip()
        src_words_bpe = []
        for word in src.split():
            if (word.startswith("$number")
                or word.startswith("$date")
                or word.startswith("$time")
                or word.startswith("$person")
                or word.startswith("$location")
                or word.startswith("$organization")
                or word.startswith("$brand")
                or word.startswith("$product")
                or word.startswith("$term")
                or word.startswith("$any")):
                    src_words_bpe.append(word)
            else:
                src_words_bpe.append(src_bpe.segment(word))
        print(" ".join(src_words_bpe) + "\t" + tgt)
