#ODPS_CONFIG=/home/yaoliang.yl/tools/odps_clt_release_64/conf/odps_config.ini
#ODPS_BIN="/home/yaoliang.yl/tools/odps_clt_release_64/bin/odpscmd --config=${ODPS_CONFIG}"
# ODPS_BIN="myodpscmd"
ODPS_BIN="odpscmd --config=~/tools/clt/conf/odps_config.ini"
#source ~/.bash_profile

input_table=$1
char_train_table=${input_table}
char_vocab_table="${input_table}_word_vocab"

#This function is used for generating final train data from input_table,
#After get the 'char_train_table', you should split the table into train.src & train.trg;
#for example: awk -F "\t" '{print $1}' $char_train_table > train.src 
# awk -F "\t" '{print $2}' $char_train_table > train.trg


#get character vocab, you can set the maximum number of vocabs according to the situation
#In latin lanuguages, we only keep top 360 characters, 
get_vocab(){
${ODPS_BIN} -e "
    use icbu_translate;
    -- udf_nlptk('tokenize', text, src_lang, tgt_lang, is_lower,is_genera,is_net, is_human_name_reg)
    --create table if not exists ${char_vocab_table}_tok as
    --select lang, udf_nlptk('tokenize', lower(text), lang, '*', true, false, false, false) as text_tok, text from ${char_train_table}
    --where lang in ('ar','zh','zh-tw','nl','en','fr','de','he','hi','id','it','ja','ko','ms','pl','pt','ru','es','th','tr','ug','uk','vi');

    --get char vocab
    drop table if exists ${char_vocab_table}_tmp;
    create table  ${char_vocab_table}_tmp(vocab string, cnt bigint);
    jar com.aliyun.odps.mapred.bridge.streaming.StreamJob \
        -jobconf stream.map.input.field.separator=\t     \
        -jobconf stream.map.output.field.separator=\t    \
        -jobconf stream.reduce.input.field.separator=\t  \
        -jobconf stream.reduce.output.field.separator=\t \
        -input  ${char_train_table}   \
        -output ${char_vocab_table}_tmp \
        -mapper \"python odps_word_map.py\" \
        -reducer \"python odps_wc_cnt.py\" \
        -jobconf odps.stage.mapper.split.size=8 \
        -file odps_word_map.py \
        -file odps_wc_cnt.py;
   
    insert into table ${char_vocab_table}_tmp(vocab, cnt) values('</S>', '10000000001');
    insert into table ${char_vocab_table}_tmp(vocab, cnt) values('<UNK>','10000000000');
    drop table if exists ${char_vocab_table};
    create table ${char_vocab_table} 
    as
    select vocab, cnt
    from ${char_vocab_table}_tmp 
    order by cnt desc; 
    -- order by cnt desc limit 50000; 
   " 
} 
get_vocab
