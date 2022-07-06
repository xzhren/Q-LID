#NMT 新干预实验
#odpscmd=~/tools/odps_clt_release_64/bin/odpscmd
odpscmd=odpscmd
table_name=$1
src_bpe=$2

#orig_table="icbu_translate.yl141858_es2en_train_data_outdomain_v1"
orig_table="icbu_translate.${table_name}"
#orig_bpe_table="icbu_translate.yl141858_es2en_train_data_outdomain_v1_bpe"
orig_bpe_table="icbu_translate.${table_name}_bpe"
bpe_origin_table(){
${odpscmd} -e "

   set odps.instance.priority=1;
   drop table if exists ${orig_bpe_table};
   create table if not exists ${orig_bpe_table}(src_bpe string, tgt string) lifecycle 180;
   jar com.aliyun.odps.mapred.bridge.streaming.StreamJob \
       -input ${orig_table} \
       -output ${orig_bpe_table} \
       -mapper \"python mapper_src_bpe.py -c ${src_bpe} -t ${src_bpe}\" \
       -reducer NONE  \
       -jobconf stream.map.input.field.separator=\t \
       -jobconf stream.map.output.field.separator=\t \
       -file mapper_src_bpe.py \
       -file ../corpus_bpe/${src_bpe} \
       ;
"
}
bpe_origin_table
