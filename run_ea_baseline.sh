#! /usr/bin/env bash
set -x
set -e

# Usage (from main repo dir): 
#   nohup ./run_ea_baseline.sh &

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

rm /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-*.json || true

python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-QUESTION_SUGGESTION.json \
    --partitions QUESTION_SUGGESTION
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-CATEGORIZATION.json \
    --partitions CATEGORIZATION
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-FEATURE_EXTRACTION.json \
    --partitions FEATURE_EXTRACTION
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-SQL_FANOUT1.json \
    --partitions SQL_FANOUT1
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-SQL_FANOUT2.json \
    --partitions SQL_FANOUT2
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-SQL_FANOUT3.json \
    --partitions SQL_FANOUT3
python -m eagle.evaluation.gen_ea_answer_llama3chat_simplified \
    --trace-path /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b.json \
    --output-file /home/jovyan/pscratch/suffix-tree-decoding/trace/llama70b/cortex-llama3-70b-ea-SQL_COMBINE.json \
    --partitions SQL_COMBINE