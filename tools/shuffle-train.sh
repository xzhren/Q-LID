#!/bin/bash

# get_seeded_random()
# {
#   seed="$1";
#   openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
#     </dev/zero 2>/dev/null;
# }

#paste ${1} ${2} | shuf > ${1}.tmp
seed=${5}
#paste ${1} ${2} | shuf --random-source=<(get_seeded_random $seed) > ${3}.tmp
paste ${1} ${2} | shuf > ${1}.tmp
cut -f1 ${1}.tmp > ${1}.${3} &
cut -f2 ${1}.tmp > ${2}.${3} &
wait
rm -f ${1}.tmp
