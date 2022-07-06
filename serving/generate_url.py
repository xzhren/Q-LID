#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :   Xingzhang Ren
@Contact :   xingzhang.rxz@alibaba-inc.com
@Time:   Sep. 19,2019
@Desc:   测试语种识别线上服务
'''


#import urllib.request
#import urllib.parse
import json
import requests
import time
import sys
import urllib
import random

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python main.py ip port")    
        exit()
 
    ip = sys.argv[1]
    port = sys.argv[2]
    #print("server address: {}:{}".format(ip, port))

    vocab = [line.strip() for line in open("vocab/vocab.txt")]

    #print("please input:")
    #for line in sys.stdin:
    for index in range(30000):
        #text = line.strip()
        #text_len = 1 + index//2
        text_len = 500
        text = "".join(random.sample(vocab, text_len))
        #params = {'service':'languagedect', 'app':'lippi-translate', 'query':text}
        url = "http://"+ip+":"+port+"/ld?service=languagedect&app=lippi-translate&query="+text
        #url = "http://"+ip+":"+port+"/ld?" + urllib.urlencode(params)
        #url = "http://"+ip+":"+port+"/ld?" + urllib.parse.urlencode(params)
        print(url)    
        #print(text)

        #s0 = time.time()
        #response = requests.post(url, data = {'query':text})
        #response = requests.get(url)
        #s1 = time.time()
        
        #data = response.content.decode()
        #data = json.loads(data)
        #print("{}\t{}".format(data['result']['lang'], 1000*(s1 - s0)))
        #print("{}\t{}\t{}".format(text_len, 1000*(s1 - s0), data['result']['time']))
        #print(data)
        #print("query time: {} ms".format(100*(s1 - s0)))
        #print("please input")0
