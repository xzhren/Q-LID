#-*-coding=utf8-*-
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

langtype = sys.argv[1]

if langtype == "23":
    supported_lang=set(["ar","zh","zh-tw","nl","en","fr","de","iw","hi","id","it","ja","ko","ms","pl","pt","ru","es","th","tr","ug","uk","vi"])
    supported_lang=set(["ar","zh","zh-tw","nl","en","fr","de","he","hi","id","it","ja","ko","ms","pl","pt","ru","es","th","tr","ug","uk","vi"])
if langtype == "19":
    supported_lang=set(['ar', 'ru', 'ko', 'ja', 'zh', 'hi', 'he','th','en','es','pt','fr','it','id','vi','pl','nl','tr','de'])
# supported_lang=set(['en','es','pt','fr','it','id','vi','pl','nl','tr','de'])
if langtype == "104": 
    supported_lang=set(["af","am","ar","az","be","bg","bn","bs","ca","ce","co","cs","cy","da","de","el","en","eo","es","et","eu","fa","fi","fr","fy","ga","gd","gl","gu","ha","haw","he","hi","hmn","hr","ht","hu","hy","id","ig","is","it","ja","jv","ka","kk","km","kn","ko","ku","ky","la","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","no","ny","pa","pl","ps","pt","ro","ru","sd","si","sk","sl","sm","sn","so","sq","sr","st","su","sv","sw","ta","te","tg","th","tl","tr","ug","uk","ur","uz","vi","xh","yi","yo","zh","zh-tw","zu"])
if langtype == "222":
   supported_lang=set(["ab","ace","af","ak","am","an","ang","ar","as","ast","ay","az","ba","bal","be","bem","ber","bg","bho","bi","bm","bn","bo","br","bs","byn","ca","cbk","ceb","ch","chm","chr","cnr","co","crh","cs","csb","cv","cy","da","de","dtp","dv","ee","el","en","eo","es","et","eu","fa","fi","fil","fj","fo","fr","fur","fvr","fy","ga","gd","gil","gl","gn","gos","gu","gv","ha","haw","hbs","he","hi","hil","hmn","hsb","ht","hu","hup","hy","ia","iba","id","ie","ig","ii","ilo","inh","io","is","it","iu","ja","jbo","jv","ka","kab","kdx","kek","kg","kha","kk","kl","km","kn","ko","kr","ks","ku","kw","ky","la","lb","lfn","lg","li","ln","lo","lt","ltg","lv","mai","mfe","mg","mh","mi","mk","ml","mn","mr","ms","mt","mus","my","nch","nds","ne","ngu","niu","nl","no","nv","ny","oc","oj","om","or","os","pa","pag","pap","pl","pmn","ps","pt","qu","quc","rm","rn","ro","rom","ru","rue","rw","sa","sco","sd","se","sg","shn","si","sk","sl","sm","sn","so","sq","sr","st","su","sv","sw","syr","szl","ta","te","tet","tg","th","ti","tk","tl","tlh","to","toi","tpi","tr","ts","tt","tvl","tw","ty","tyv","udm","ug","uk","umb","ur","uz","ve","vi","vo","wa","war","wo","xh","yi","yo","za","zh","zh-tw","zu","zza"])


supported_lang_all = set(["ab","ace","af","ak","am","an","ang","ar","as","ast","ay","az","ba","bal","be","bem","ber","bg","bho","bi","bm","bn","bo","br","bs","byn","ca","cbk","ceb","ch","chm","chr","cnr","co","crh","cs","csb","cv","cy","da","de","dtp","dv","ee","el","en","eo","es","et","eu","fa","fi","fil","fj","fo","fr","fur","fvr","fy","ga","gd","gil","gl","gn","gos","gu","gv","ha","haw","hbs","he","hi","hil","hmn","hsb","ht","hu","hup","hy","ia","iba","id","ie","ig","ii","ilo","inh","io","is","it","iu","ja","jbo","jv","ka","kab","kdx","kek","kg","kha","kk","kl","km","kn","ko","kr","ks","ku","kw","ky","la","lb","lfn","lg","li","ln","lo","lt","ltg","lv","mai","mfe","mg","mh","mi","mk","ml","mn","mr","ms","mt","mus","my","nch","nds","ne","ngu","niu","nl","no","nv","ny","oc","oj","om","or","os","pa","pag","pap","pl","pmn","ps","pt","qu","quc","rm","rn","ro","rom","ru","rue","rw","sa","sco","sd","se","sg","shn","si","sk","sl","sm","sn","so","sq","sr","st","su","sv","sw","syr","szl","ta","te","tet","tg","th","ti","tk","tl","tlh","to","toi","tpi","tr","ts","tt","tvl","tw","ty","tyv","udm","ug","uk","umb","ur","uz","ve","vi","vo","wa","war","wo","xh","yi","yo","za","zh","zh-tw","zu","zza"])
print(langtype, ":", supported_lang)

lang_dict = {"布尔语":"af","阿姆哈拉语":"am","阿拉伯语":"ar","阿塞拜疆语":"az","白俄罗斯语":"be","保加利亚语":"bg","孟加拉语":"bn","藏语":"bo","波斯尼亚语":"bs","加泰罗尼亚语":"ca","宿务语":"ce","科西嘉语":"co","捷克语":"cs","威尔士语":"cy","丹麦语":"da","德语":"de","希腊语":"el","英语":"en","世界语":"eo","西班牙语":"es","爱沙尼亚语":"et","巴斯克语":"eu","波斯语":"fa","芬兰语":"fi","法语":"fr","弗里西语":"fy","爱尔兰语":"ga","苏格兰盖尔语":"gd","加利西亚语":"gl","古吉拉特语":"gu","豪萨语":"ha","夏威夷语":"haw","希伯来语":"he","印地语":"hi","苗语":"hmn","克罗地亚语":"hr","海地克里奥尔语":"ht","匈牙利语":"hu","亚美尼亚语":"hy","印尼语":"id","伊博语":"ig","彝文":"ii","冰岛语":"is","意大利语":"it","日语":"ja","印尼爪哇语":"jv","格鲁吉亚语":"ka","哈萨克语":"kk","高棉语":"km","卡纳达语":"kn","韩语":"ko","库尔德语":"ku","吉尔吉斯语":"ky","拉丁语":"la","卢森堡语":"li","老挝语":"lo","立陶宛语":"lt","拉脱维亚语":"lv","马尔加什语":"mg","毛利语":"mi","马其顿语":"mk","马拉雅拉姆语":"ml","蒙古语":"mn","马拉地语":"mr","马来语":"ms","马耳他语":"mt","缅甸语":"my","尼泊尔语":"ne","荷兰语":"nl","挪威语":"no","齐切瓦语":"ny","旁遮普语":"pa","波兰语":"pl","普什图语":"ps","葡萄牙语":"pt","罗马尼亚语":"ro","俄语":"ru","信德语":"sd","僧伽罗语":"si","斯洛伐克语":"sk","斯洛文尼亚语":"sl","萨摩亚语":"sm","修纳语":"sn","索马里语":"so","阿尔巴尼亚语":"sq","塞尔维亚语":"sr","塞索托语":"st","印尼巽他语":"su","瑞典语":"sv","斯瓦希里语":"sw","泰米尔语":"ta","泰卢固语":"te","塔吉克语":"tg","泰语":"th","菲律宾语":"tl","土耳其语":"tr","维吾尔语":"ug","乌克兰语":"uk","乌尔都语":"ur","乌兹别克语":"uz","越南语":"vi","南非科萨语":"xh","意第绪语":"yi","约鲁巴语":"yo","粤语":"yue","壮语":"za","中文":"zh","繁体中文":"zh-tw","南非祖鲁语":"zu"}

lang_p = {}
lang_rec_corr = {}
lang_recall = {}
for lang in supported_lang:
    lang_p[lang] = 0
    lang_recall[lang] = 0
    lang_rec_corr[lang] = 0
lang_p["null"]=0
lang_recall["null"]=0
lang_rec_corr["null"]=0

def compute_f_score(input_file):
    counter_all, counter_right = 0, 0
    with open(input_file, 'r') as f:
        for line in f:
            #tokens = line.strip().decode('utf-8').split('\t')
            tokens = line.strip().split('\t')
            # if len(tokens) < 3:
            #     continue
            assert len(tokens) >= 3
            query = tokens[0]
            human_tag = tokens[1]
            rec_lang = tokens[2]
            if human_tag not in supported_lang: continue
            if rec_lang == human_tag: counter_right+=1
            counter_all += 1

            # for not supported language
            #if human_tag not in supported_lang:
            #    continue
            #if rec_lang not in supported_lang:
            #    rec_lang = "null"

            ##if recogize language equals human tagging result
            if rec_lang == human_tag:
                lang_rec_corr[rec_lang] += 1
            if rec_lang not in lang_p: lang_p[rec_lang] = 0
            lang_p[rec_lang] += 1
            lang_recall[human_tag] += 1


    supported_langlt = sorted(supported_lang)
    #compute P、R、F
    print("language\tP\tR\tF")
    counter_rec_zero, counter_hum_zero = 0, 0
    for lang in supported_langlt:
         p_lang = lang_rec_corr[lang] / (lang_p[lang] + 0.00001) #in case of denominator is zero
         r_lang = lang_rec_corr[lang] / (lang_recall[lang] + 0.00001)
         f_lang = (2 * p_lang * r_lang) / ( p_lang + r_lang + 0.00001)
         #lang_str = list(lang_dict.keys())[list(lang_dict.values()).index(lang)]
         #print("%s\t%s\t%.2f\t%.2f\t%.2f" % (lang_str, lang, p_lang*100, r_lang * 100, f_lang * 100))
         print("%s\t%.2f\t%.2f\t%.2f" % (lang, p_lang*100, r_lang * 100, f_lang * 100))

    for lang in supported_lang_all:
         if lang in lang_p and  lang_p[lang] != 0: counter_rec_zero += 1
         if lang in lang_recall and lang_recall[lang] != 0: counter_hum_zero += 1
    print("human eval langs: %d, model eval langs: %d " % (counter_hum_zero, counter_rec_zero) )
    print("ACC:", "%.2f, right: %d, all: %d" % (100.0*counter_right/counter_all, counter_right, counter_all) )


if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_f_score.py [19/23/104] [input_file]")
        sys.exit(1)
    input_file = sys.argv[2]
    compute_f_score(input_file)
