ODPS_BIN="odpscmd --config=~/tools/odps_clt/conf/odps_config_xzhren.ini"
ODPS_BIN="odpscmd"

input_table=$1
char_train_table="icbu_translate.xf207793_langident_format_${input_table}"
char_vocab_table="icbu_translate.xf207793_langident_char_vocab_${input_table}"
work_dir=$2


# download char_train_table
${ODPS_BIN} -e "
tunnel download ${char_train_table} ${work_dir}/char_test_table -fd '\t';
"

# split the table into train.src & train.trg
awk -F "\t" '{print $1}' ${work_dir}/char_test_table > ${work_dir}/test.src 
awk -F "\t" '{print $2}' ${work_dir}/char_test_table > ${work_dir}/test.trg


# download char_vocab_table
${ODPS_BIN} -e "
tunnel download ${char_vocab_table} ${work_dir}/test.vocab -fd '\t';
"
