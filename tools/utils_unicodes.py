#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

class UNICODE_BLOCK:
    def __init__(self):
        self.id2name = {0:"UNK"}
        self.name2id = {"UNK":0}
        self.unkindex = 0
        self.blocks = collections.OrderedDict()
        index = 1
        lastid = -1
        for line in unicode_str.strip().split("\n"):
            _, _, _, startid, endid, name = line.split("\t")
            startid, endid = int(startid), int(endid)
            if name not in self.name2id:
                self.name2id[name] = index
                self.id2name[index] = name
            else:
                assert False
            # if endid <= lastid: print(line, endid, startid)
            assert endid > lastid
            if startid != lastid + 1:
                self.blocks[lastid + 1] = self.unkindex
            self.blocks[startid] = index
            index += 1
            lastid = endid
        self.blocks[endid] = index
        print("[INFO]: load %d unicode blocks" % index)
        assert index == 309
        print(self.id2name)
        print(self.blocks)

    def get_char_ublock_id(self, stritem):
        if len(stritem) != 1: print(stritem)
        assert len(stritem) == 1
        utf_8_code = ord(stritem)
        res_codeindex = -1
        for code, codeindex in self.blocks.items():
            #print(code, utf_8_code, res_codeindex)
            if utf_8_code < code:
                #print(stritem, utf_8_code, res_codeindex)
                if res_codeindex <= 0: print(utf_8_code, res_codeindex)
                assert res_codeindex >= 0
                return res_codeindex
            res_codeindex = codeindex
        print("ERROR UBLOCL INDEX", stritem, utf_8_code)
        assert False
        # return 0

            
    def get_str_ublock_id(self, strlist):
        unicode_script_ids = []
        for item in strlist:
            codeindex = self.get_char_ublock_id(item)
            unicode_script_ids.append(codeindex)
        return unicode_script_ids


unicode_str = """
U+0000	U+007F	128	0	127	Basic Latin
U+0080	U+00FF	128	128	255	Latin-1 Supplement
U+0100	U+017F	128	256	383	Latin Extended-A
U+0180	U+024F	208	384	591	Latin Extended-B
U+0250	U+02AF	96	592	687	IPA Extensions
U+02B0	U+02FF	80	688	767	Spacing Modifier Letters
U+0300	U+036F	112	768	879	Combining Diacritical Marks
U+0370	U+03FF	144	880	1023	Greek and Coptic
U+0400	U+04FF	256	1024	1279	Cyrillic
U+0500	U+052F	48	1280	1327	Cyrillic Supplement
U+0530	U+058F	96	1328	1423	Armenian
U+0590	U+05FF	112	1424	1535	Hebrew
U+0600	U+06FF	256	1536	1791	Arabic
U+0700	U+074F	80	1792	1871	Syriac
U+0750	U+077F	48	1872	1919	Arabic Supplement
U+0780	U+07BF	64	1920	1983	Thaana
U+07C0	U+07FF	64	1984	2047	NKo
U+0800	U+083F	64	2048	2111	Samaritan
U+0840	U+085F	32	2112	2143	Mandaic
U+0860	U+086F	16	2144	2159	Syriac Supplement
U+08A0	U+08FF	96	2208	2303	Arabic Extended-A
U+0900	U+097F	128	2304	2431	Devanagari
U+0980	U+09FF	128	2432	2559	Bengali
U+0A00	U+0A7F	128	2560	2687	Gurmukhi
U+0A80	U+0AFF	128	2688	2815	Gujarati
U+0B00	U+0B7F	128	2816	2943	Oriya
U+0B80	U+0BFF	128	2944	3071	Tamil
U+0C00	U+0C7F	128	3072	3199	Telugu
U+0C80	U+0CFF	128	3200	3327	Kannada
U+0D00	U+0D7F	128	3328	3455	Malayalam
U+0D80	U+0DFF	128	3456	3583	Sinhala
U+0E00	U+0E7F	128	3584	3711	Thai
U+0E80	U+0EFF	128	3712	3839	Lao
U+0F00	U+0FFF	256	3840	4095	Tibetan
U+1000	U+109F	160	4096	4255	Myanmar
U+10A0	U+10FF	96	4256	4351	Georgian
U+1100	U+11FF	256	4352	4607	Hangul Jamo
U+1200	U+137F	384	4608	4991	Ethiopic
U+1380	U+139F	32	4992	5023	Ethiopic Supplement
U+13A0	U+13FF	96	5024	5119	Cherokee
U+1400	U+167F	640	5120	5759	Unified Canadian Aboriginal Syllabics
U+1680	U+169F	32	5760	5791	Ogham
U+16A0	U+16FF	96	5792	5887	Runic
U+1700	U+171F	32	5888	5919	Tagalog
U+1720	U+173F	32	5920	5951	Hanunoo
U+1740	U+175F	32	5952	5983	Buhid
U+1760	U+177F	32	5984	6015	Tagbanwa
U+1780	U+17FF	128	6016	6143	Khmer
U+1800	U+18AF	176	6144	6319	Mongolian
U+18B0	U+18FF	80	6320	6399	Unified Canadian Aboriginal Syllabics Extended
U+1900	U+194F	80	6400	6479	Limbu
U+1950	U+197F	48	6480	6527	Tai Le
U+1980	U+19DF	96	6528	6623	New Tai Lue
U+19E0	U+19FF	32	6624	6655	Khmer Symbols
U+1A00	U+1A1F	32	6656	6687	Buginese
U+1A20	U+1AAF	144	6688	6831	Tai Tham
U+1AB0	U+1AFF	80	6832	6911	Combining Diacritical Marks Extended
U+1B00	U+1B7F	128	6912	7039	Balinese
U+1B80	U+1BBF	64	7040	7103	Sundanese
U+1BC0	U+1BFF	64	7104	7167	Batak
U+1C00	U+1C4F	80	7168	7247	Lepcha
U+1C50	U+1C7F	48	7248	7295	Ol Chiki
U+1C80	U+1C8F	16	7296	7311	Cyrillic Extended-C
U+1C90	U+1CBF	48	7312	7359	Georgian Extended
U+1CC0	U+1CCF	16	7360	7375	Sundanese Supplement
U+1CD0	U+1CFF	48	7376	7423	Vedic Extensions
U+1D00	U+1D7F	128	7424	7551	Phonetic Extensions
U+1D80	U+1DBF	64	7552	7615	Phonetic Extensions Supplement
U+1DC0	U+1DFF	64	7616	7679	Combining Diacritical Marks Supplement
U+1E00	U+1EFF	256	7680	7935	Latin Extended Additional
U+1F00	U+1FFF	256	7936	8191	Greek Extended
U+2000	U+206F	112	8192	8303	General Punctuation
U+2070	U+209F	48	8304	8351	Superscripts and Subscripts
U+20A0	U+20CF	48	8352	8399	Currency Symbols
U+20D0	U+20FF	48	8400	8447	Combining Diacritical Marks for Symbols
U+2100	U+214F	80	8448	8527	Letterlike Symbols
U+2150	U+218F	64	8528	8591	Number Forms
U+2190	U+21FF	112	8592	8703	Arrows
U+2200	U+22FF	256	8704	8959	Mathematical Operators
U+2300	U+23FF	256	8960	9215	Miscellaneous Technical
U+2400	U+243F	64	9216	9279	Control Pictures
U+2440	U+245F	32	9280	9311	Optical Character Recognition
U+2460	U+24FF	160	9312	9471	Enclosed Alphanumerics
U+2500	U+257F	128	9472	9599	Box Drawing
U+2580	U+259F	32	9600	9631	Block Elements
U+25A0	U+25FF	96	9632	9727	Geometric Shapes
U+2600	U+26FF	256	9728	9983	Miscellaneous Symbols
U+2700	U+27BF	192	9984	10175	Dingbats
U+27C0	U+27EF	48	10176	10223	Miscellaneous Mathematical Symbols-A
U+27F0	U+27FF	16	10224	10239	Supplemental Arrows-A
U+2800	U+28FF	256	10240	10495	Braille Patterns
U+2900	U+297F	128	10496	10623	Supplemental Arrows-B
U+2980	U+29FF	128	10624	10751	Miscellaneous Mathematical Symbols-B
U+2A00	U+2AFF	256	10752	11007	Supplemental Mathematical Operators
U+2B00	U+2BFF	256	11008	11263	Miscellaneous Symbols and Arrows
U+2C00	U+2C5F	96	11264	11359	Glagolitic
U+2C60	U+2C7F	32	11360	11391	Latin Extended-C
U+2C80	U+2CFF	128	11392	11519	Coptic
U+2D00	U+2D2F	48	11520	11567	Georgian Supplement
U+2D30	U+2D7F	80	11568	11647	Tifinagh
U+2D80	U+2DDF	96	11648	11743	Ethiopic Extended
U+2DE0	U+2DFF	32	11744	11775	Cyrillic Extended-A
U+2E00	U+2E7F	128	11776	11903	Supplemental Punctuation
U+2E80	U+2EFF	128	11904	12031	CJK Radicals Supplement
U+2F00	U+2FDF	224	12032	12255	Kangxi Radicals
U+2FF0	U+2FFF	16	12272	12287	Ideographic Description Characters
U+3000	U+303F	64	12288	12351	CJK Symbols and Punctuation
U+3040	U+309F	96	12352	12447	Hiragana
U+30A0	U+30FF	96	12448	12543	Katakana
U+3100	U+312F	48	12544	12591	Bopomofo
U+3130	U+318F	96	12592	12687	Hangul Compatibility Jamo
U+3190	U+319F	16	12688	12703	Kanbun
U+31A0	U+31BF	32	12704	12735	Bopomofo Extended
U+31C0	U+31EF	48	12736	12783	CJK Strokes
U+31F0	U+31FF	16	12784	12799	Katakana Phonetic Extensions
U+3200	U+32FF	256	12800	13055	Enclosed CJK Letters and Months
U+3300	U+33FF	256	13056	13311	CJK Compatibility
U+3400	U+4DBF	6592	13312	19903	CJK Unified Ideographs Extension A
U+4DC0	U+4DFF	64	19904	19967	Yijing Hexagram Symbols
U+4E00	U+9FFF	20992	19968	40959	CJK Unified Ideographs
U+A000	U+A48F	1168	40960	42127	Yi Syllables
U+A490	U+A4CF	64	42128	42191	Yi Radicals
U+A4D0	U+A4FF	48	42192	42239	Lisu
U+A500	U+A63F	320	42240	42559	Vai
U+A640	U+A69F	96	42560	42655	Cyrillic Extended-B
U+A6A0	U+A6FF	96	42656	42751	Bamum
U+A700	U+A71F	32	42752	42783	Modifier Tone Letters
U+A720	U+A7FF	224	42784	43007	Latin Extended-D
U+A800	U+A82F	48	43008	43055	Syloti Nagri
U+A830	U+A83F	16	43056	43071	Common Indic Number Forms
U+A840	U+A87F	64	43072	43135	Phags-pa
U+A880	U+A8DF	96	43136	43231	Saurashtra
U+A8E0	U+A8FF	32	43232	43263	Devanagari Extended
U+A900	U+A92F	48	43264	43311	Kayah Li
U+A930	U+A95F	48	43312	43359	Rejang
U+A960	U+A97F	32	43360	43391	Hangul Jamo Extended-A
U+A980	U+A9DF	96	43392	43487	Javanese
U+A9E0	U+A9FF	32	43488	43519	Myanmar Extended-B
U+AA00	U+AA5F	96	43520	43615	Cham
U+AA60	U+AA7F	32	43616	43647	Myanmar Extended-A
U+AA80	U+AADF	96	43648	43743	Tai Viet
U+AAE0	U+AAFF	32	43744	43775	Meetei Mayek Extensions
U+AB00	U+AB2F	48	43776	43823	Ethiopic Extended-A
U+AB30	U+AB6F	64	43824	43887	Latin Extended-E
U+AB70	U+ABBF	80	43888	43967	Cherokee Supplement
U+ABC0	U+ABFF	64	43968	44031	Meetei Mayek
U+AC00	U+D7AF	11184	44032	55215	Hangul Syllables
U+D7B0	U+D7FF	80	55216	55295	Hangul Jamo Extended-B
U+D800	U+DB7F	896	55296	56191	High Surrogates
U+DB80	U+DBFF	128	56192	56319	High Private Use Surrogates
U+DC00	U+DFFF	1024	56320	57343	Low Surrogates
U+E000	U+F8FF	6400	57344	63743	Private Use Area
U+F900	U+FAFF	512	63744	64255	CJK Compatibility Ideographs
U+FB00	U+FB4F	80	64256	64335	Alphabetic Presentation Forms
U+FB50	U+FDFF	688	64336	65023	Arabic Presentation Forms-A
U+FE00	U+FE0F	16	65024	65039	Variation Selectors
U+FE10	U+FE1F	16	65040	65055	Vertical Forms
U+FE20	U+FE2F	16	65056	65071	Combining Half Marks
U+FE30	U+FE4F	32	65072	65103	CJK Compatibility Forms
U+FE50	U+FE6F	32	65104	65135	Small Form Variants
U+FE70	U+FEFF	144	65136	65279	Arabic Presentation Forms-B
U+FF00	U+FFEF	240	65280	65519	Halfwidth and Fullwidth Forms
U+FFF0	U+FFFF	16	65520	65535	Specials
U+10000	U+1007F	128	65536	65663	Linear B Syllabary
U+10080	U+100FF	128	65664	65791	Linear B Ideograms
U+10100	U+1013F	64	65792	65855	Aegean Numbers
U+10140	U+1018F	80	65856	65935	Ancient Greek Numbers
U+10190	U+101CF	64	65936	65999	Ancient Symbols
U+101D0	U+101FF	48	66000	66047	Phaistos Disc
U+10280	U+1029F	32	66176	66207	Lycian
U+102A0	U+102DF	64	66208	66271	Carian
U+102E0	U+102FF	32	66272	66303	Coptic Epact Numbers
U+10300	U+1032F	48	66304	66351	Old Italic
U+10330	U+1034F	32	66352	66383	Gothic
U+10350	U+1037F	48	66384	66431	Old Permic
U+10380	U+1039F	32	66432	66463	Ugaritic
U+103A0	U+103DF	64	66464	66527	Old Persian
U+10400	U+1044F	80	66560	66639	Deseret
U+10450	U+1047F	48	66640	66687	Shavian
U+10480	U+104AF	48	66688	66735	Osmanya
U+104B0	U+104FF	80	66736	66815	Osage
U+10500	U+1052F	48	66816	66863	Elbasan
U+10530	U+1056F	64	66864	66927	Caucasian Albanian
U+10600	U+1077F	384	67072	67455	Linear A
U+10800	U+1083F	64	67584	67647	Cypriot Syllabary
U+10840	U+1085F	32	67648	67679	Imperial Aramaic
U+10860	U+1087F	32	67680	67711	Palmyrene
U+10880	U+108AF	48	67712	67759	Nabataean
U+108E0	U+108FF	32	67808	67839	Hatran
U+10900	U+1091F	32	67840	67871	Phoenician
U+10920	U+1093F	32	67872	67903	Lydian
U+10980	U+1099F	32	67968	67999	Meroitic Hieroglyphs
U+109A0	U+109FF	96	68000	68095	Meroitic Cursive
U+10A00	U+10A5F	96	68096	68191	Kharoshthi
U+10A60	U+10A7F	32	68192	68223	Old South Arabian
U+10A80	U+10A9F	32	68224	68255	Old North Arabian
U+10AC0	U+10AFF	64	68288	68351	Manichaean
U+10B00	U+10B3F	64	68352	68415	Avestan
U+10B40	U+10B5F	32	68416	68447	Inscriptional Parthian
U+10B60	U+10B7F	32	68448	68479	Inscriptional Pahlavi
U+10B80	U+10BAF	48	68480	68527	Psalter Pahlavi
U+10C00	U+10C4F	80	68608	68687	Old Turkic
U+10C80	U+10CFF	128	68736	68863	Old Hungarian
U+10D00	U+10D3F	64	68864	68927	Hanifi Rohingya
U+10E60	U+10E7F	32	69216	69247	Rumi Numeral Symbols
U+10E80	U+10EBF	64	69248	69311	Yezidi
U+10F00	U+10F2F	48	69376	69423	Old Sogdian
U+10F30	U+10F6F	64	69424	69487	Sogdian
U+10FB0	U+10FDF	48	69552	69599	Chorasmian
U+10FE0	U+10FFF	32	69600	69631	Elymaic
U+11000	U+1107F	128	69632	69759	Brahmi
U+11080	U+110CF	80	69760	69839	Kaithi
U+110D0	U+110FF	48	69840	69887	Sora Sompeng
U+11100	U+1114F	80	69888	69967	Chakma
U+11150	U+1117F	48	69968	70015	Mahajani
U+11180	U+111DF	96	70016	70111	Sharada
U+111E0	U+111FF	32	70112	70143	Sinhala Archaic Numbers
U+11200	U+1124F	80	70144	70223	Khojki
U+11280	U+112AF	48	70272	70319	Multani
U+112B0	U+112FF	80	70320	70399	Khudawadi
U+11300	U+1137F	128	70400	70527	Grantha
U+11400	U+1147F	128	70656	70783	Newa
U+11480	U+114DF	96	70784	70879	Tirhuta
U+11580	U+115FF	128	71040	71167	Siddham
U+11600	U+1165F	96	71168	71263	Modi
U+11660	U+1167F	32	71264	71295	Mongolian Supplement
U+11680	U+116CF	80	71296	71375	Takri
U+11700	U+1173F	64	71424	71487	Ahom
U+11800	U+1184F	80	71680	71759	Dogra
U+118A0	U+118FF	96	71840	71935	Warang Citi
U+11900	U+1195F	96	71936	72031	Dives Akuru
U+119A0	U+119FF	96	72096	72191	Nandinagari
U+11A00	U+11A4F	80	72192	72271	Zanabazar Square
U+11A50	U+11AAF	96	72272	72367	Soyombo
U+11AC0	U+11AFF	64	72384	72447	Pau Cin Hau
U+11C00	U+11C6F	112	72704	72815	Bhaiksuki
U+11C70	U+11CBF	80	72816	72895	Marchen
U+11D00	U+11D5F	96	72960	73055	Masaram Gondi
U+11D60	U+11DAF	80	73056	73135	Gunjala Gondi
U+11EE0	U+11EFF	32	73440	73471	Makasar
U+11FB0	U+11FBF	16	73648	73663	Lisu Supplement
U+11FC0	U+11FFF	64	73664	73727	Tamil Supplement
U+12000	U+123FF	1024	73728	74751	Cuneiform
U+12400	U+1247F	128	74752	74879	Cuneiform Numbers and Punctuation
U+12480	U+1254F	208	74880	75087	Early Dynastic Cuneiform
U+13000	U+1342F	1072	77824	78895	Egyptian Hieroglyphs
U+13430	U+1343F	16	78896	78911	Egyptian Hieroglyph Format Controls
U+14400	U+1467F	640	82944	83583	Anatolian Hieroglyphs
U+16800	U+16A3F	576	92160	92735	Bamum Supplement
U+16A40	U+16A6F	48	92736	92783	Mro
U+16AD0	U+16AFF	48	92880	92927	Bassa Vah
U+16B00	U+16B8F	144	92928	93071	Pahawh Hmong
U+16E40	U+16E9F	96	93760	93855	Medefaidrin
U+16F00	U+16F9F	160	93952	94111	Miao
U+16FE0	U+16FFF	32	94176	94207	Ideographic Symbols and Punctuation
U+17000	U+187FF	6144	94208	100351	Tangut
U+18800	U+18AFF	768	100352	101119	Tangut Components
U+18B00	U+18CFF	512	101120	101631	Khitan Small Script
U+18D00	U+18D8F	144	101632	101775	Tangut Supplement
U+1B000	U+1B0FF	256	110592	110847	Kana Supplement
U+1B100	U+1B12F	48	110848	110895	Kana Extended-A
U+1B130	U+1B16F	64	110896	110959	Small Kana Extension
U+1B170	U+1B2FF	400	110960	111359	Nushu
U+1BC00	U+1BC9F	160	113664	113823	Duployan
U+1BCA0	U+1BCAF	16	113824	113839	Shorthand Format Controls
U+1D000	U+1D0FF	256	118784	119039	Byzantine Musical Symbols
U+1D100	U+1D1FF	256	119040	119295	Musical Symbols
U+1D200	U+1D24F	80	119296	119375	Ancient Greek Musical Notation
U+1D2E0	U+1D2FF	32	119520	119551	Mayan Numerals
U+1D300	U+1D35F	96	119552	119647	Tai Xuan Jing Symbols
U+1D360	U+1D37F	32	119648	119679	Counting Rod Numerals
U+1D400	U+1D7FF	1024	119808	120831	Mathematical Alphanumeric Symbols
U+1D800	U+1DAAF	688	120832	121519	Sutton SignWriting
U+1E000	U+1E02F	48	122880	122927	Glagolitic Supplement
U+1E100	U+1E14F	80	123136	123215	Nyiakeng Puachue Hmong
U+1E2C0	U+1E2FF	64	123584	123647	Wancho
U+1E800	U+1E8DF	224	124928	125151	Mende Kikakui
U+1E900	U+1E95F	96	125184	125279	Adlam
U+1EC70	U+1ECBF	80	126064	126143	Indic Siyaq Numbers
U+1ED00	U+1ED4F	80	126208	126287	Ottoman Siyaq Numbers
U+1EE00	U+1EEFF	256	126464	126719	Arabic Mathematical Alphabetic Symbols
U+1F000	U+1F02F	48	126976	127023	Mahjong Tiles
U+1F030	U+1F09F	112	127024	127135	Domino Tiles
U+1F0A0	U+1F0FF	96	127136	127231	Playing Cards
U+1F100	U+1F1FF	256	127232	127487	Enclosed Alphanumeric Supplement
U+1F200	U+1F2FF	256	127488	127743	Enclosed Ideographic Supplement
U+1F300	U+1F5FF	768	127744	128511	Miscellaneous Symbols and Pictographs
U+1F600	U+1F64F	80	128512	128591	Emoticons
U+1F650	U+1F67F	48	128592	128639	Ornamental Dingbats
U+1F680	U+1F6FF	128	128640	128767	Transport and Map Symbols
U+1F700	U+1F77F	128	128768	128895	Alchemical Symbols
U+1F780	U+1F7FF	128	128896	129023	Geometric Shapes Extended
U+1F800	U+1F8FF	256	129024	129279	Supplemental Arrows-C
U+1F900	U+1F9FF	256	129280	129535	Supplemental Symbols and Pictographs
U+1FA00	U+1FA6F	112	129536	129647	Chess Symbols
U+1FA70	U+1FAFF	144	129648	129791	Symbols and Pictographs Extended-A
U+1FB00	U+1FBFF	256	129792	130047	Symbols for Legacy Computing
U+20000	U+2A6DF	42720	131072	173791	CJK Unified Ideographs Extension B
U+2A700	U+2B73F	4160	173824	177983	CJK Unified Ideographs Extension C
U+2B740	U+2B81F	224	177984	178207	CJK Unified Ideographs Extension D
U+2B820	U+2CEAF	5776	178208	183983	CJK Unified Ideographs Extension E
U+2CEB0	U+2EBEF	7488	183984	191471	CJK Unified Ideographs Extension F
U+2F800	U+2FA1F	544	194560	195103	CJK Compatibility Ideographs Supplement
U+30000	U+3134F	4944	196608	201551	CJK Unified Ideographs Extension G
U+E0000	U+E007F	128	917504	917631	Tags
U+E0100	U+E01EF	240	917760	917999	Variation Selectors Supplement
U+F0000	U+FFFFF	65536	983040	1048575	Supplementary Private Use Area-A
U+100000	U+10FFFF	65536	1048576	1114111	Supplementary Private Use Area-B
"""

if __name__ == "__main__":
    ublock_utls = UNICODE_BLOCK()
    test_strlt = "this is test data."
    res = ublock_utls.get_str_ublock_id(test_strlt)
    print(test_strlt, res)
    test_strlt = "Clé USB 这是测试 数据.ﻯ"
    res = ublock_utls.get_str_ublock_id(test_strlt)
    print(test_strlt, res)
    test_strlt = u"\u0897"
    res = ublock_utls.get_str_ublock_id(test_strlt)
    print("txt:", test_strlt, "res:", res)

    index = 1
    lastid = -1
    id2name = {}
    name2id = {}
    # self.unkindex = 0
    for line in unicode_str.strip().split("\n"):
        startid_code, endid_code, _, startid, endid, name = line.split("\t")
        startid, endid = int(startid), int(endid)
        if name not in name2id:
            name2id[name] = index
            id2name[index] = name
            startid_code = startid_code.replace("U+","0x")
            endid_code = endid_code.replace("U+","0x")
            #print("if (unicode >= %s and unicode <= %s) return %d;" %(startid_code, endid_code, index))
        else:
            assert False
        # if endid <= lastid: print(line, endid, startid)
        assert endid > lastid
        index += 1
        lastid = endid
