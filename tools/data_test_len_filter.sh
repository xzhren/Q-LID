#ODPS_CONFIG=/home/yaoliang.yl/tools/odps_clt_release_64/conf/odps_config.ini
#ODPS_BIN="/home/yaoliang.yl/tools/odps_clt_release_64/bin/odpscmd --config=${ODPS_CONFIG}"
# ODPS_BIN="myodpscmd"
ODPS_BIN="odpscmd --config=~/tools/odps_clt/conf/odps_config_xzhren.ini"
#source ~/.bash_profile

input_table=$1
char_train_table="icbu_translate.xf207793_langident_format_${input_table}"
char_train_table_filter="icbu_translate.xf207793_langident_format_${input_table}_filter"

#This function is used for generating final train data from input_table,
#After get the 'char_train_table', you should split the table into train.src & train.trg;
#for example: awk -F "\t" '{print $1}' $char_train_table > train.src 
# awk -F "\t" '{print $2}' $char_train_table > train.trg
filter_data(){
${ODPS_BIN} -e "
DROP TABLE IF EXISTS ${char_train_table_filter};
CREATE TABLE ${char_train_table_filter}(text string, lang string);
jar com.aliyun.odps.mapred.bridge.streaming.StreamJob \
            -jobconf stream.map.input.field.separator=\t     \
            -jobconf stream.map.output.field.separator=\t    \
            -jobconf stream.reduce.input.field.separator=\t  \
            -jobconf stream.reduce.output.field.separator=\t \
            -input  ${char_train_table}    \
            -output ${char_train_table_filter}   \
            -mapper \"python odps_data_len_filter.py\" \
            -reducer NONE \
            -jobconf odps.stage.mapper.split.size=8 \
            -file ./odps_data_len_filter.py;
"
}

filter_data

work_dir="../corpus89"

# download char_train_table
${ODPS_BIN} -e "
tunnel download ${char_train_table_filter} ${work_dir}/char_test_table -fd '\t';
"

# split the table into train.src & train.trg
awk -F "\t" '{print $1}' ${work_dir}/char_test_table > ${work_dir}/test.src
awk -F "\t" '{print $2}' ${work_dir}/char_test_table > ${work_dir}/test.trg
