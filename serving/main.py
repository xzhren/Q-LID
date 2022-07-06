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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python main.py ip port")    
        exit()
 
    ip = sys.argv[1]
    port = sys.argv[2]
    #print("server address: {}:{}".format(ip, port))

    #print("please input:")
    for line in sys.stdin:
        text = line.strip()

        params = {'service':'languagedect', 'app':'lippi-translate', 'query':text}
        #url = "http://"+ip+":"+port+"/ld?service=languagedect&app=lippi-translate&query="+text
        url = "http://"+ip+":"+port+"/ld?" + urllib.urlencode(params)
        #url = "http://"+ip+":"+port+"/ld?" + urllib.parse.urlencode(params)
        print(url)    

        s0 = time.time()
        #response = requests.post(url, data = {'query':text})
        response = requests.get(url)
        s1 = time.time()
        
        data = response.content.decode()
        data = json.loads(data)
        #print("{}\t{}".format(data['result']['lang'], 1000*(s1 - s0)))
        #print(data)
        #print("query time: {} ms".format(100*(s1 - s0)))
        #print("please input")0
