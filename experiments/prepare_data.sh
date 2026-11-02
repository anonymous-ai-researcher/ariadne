#!/usr/bin/env bash
# Prepares every data set used by the experiment files.  Edit the paths at the top; each block can be run on its own.
# EMB is the embedder spec (wordllama | st:<sentence-transformers model> | openai:<model>); the prepared file is named
# data/<name>__<tag>.npz with tag = EMB with ":" -> "-" and "/" -> "_", and the experiment files pick it up through the
# environment variable EMB=<tag> (default wordllama).
#
#   EMB_SPEC=st:BAAI/bge-small-en-v1.5 bash experiments/prepare_data.sh
#   EMB=st-BAAI_bge-small-en-v1.5 python experiments/run.py experiments/C1_calls_vs_policy.yaml
#
# Sources (public; download them yourself):
#   LoCoMo        locomo10.json from the LoCoMo repository (snap-research/locomo, data/locomo10.json)
#   AG News       train.csv of the AG News topic classification corpus (class, title, description)
#   LongMemEval   longmemeval_s.json (or _m) from the LongMemEval release
#   QQP           quora_duplicate_questions.tsv (Quora Question Pairs)    
#   MS MARCO      collection.tsv of the passage ranking collection        
#   spaCy vectors the wheel of en_core_web_md (word vectors only, for the synthetic fan-out workload)
#   ANN-Benchmarks HDF5 files such as glove-100-angular.hdf5 (vectors only)
# To look at the first parsed items of a file before a long run:
#   python -m ariadne.data prep --dataset qqp --raw <file> --peek 5
set -euo pipefail
RAW=${RAW:-data/raw}
EMB_SPEC=${EMB_SPEC:-wordllama}

[ -f $RAW/locomo10.json ]  && python -m ariadne.data prep --dataset locomo      --raw $RAW/locomo10.json --embedder "$EMB_SPEC"
[ -f $RAW/train.csv ]      && python -m ariadne.data prep --dataset agnews      --raw $RAW/train.csv     --embedder "$EMB_SPEC"
[ -f $RAW/longmemeval_s.json ] && python -m ariadne.data prep --dataset longmemeval --raw $RAW/longmemeval_s.json --embedder "$EMB_SPEC"
[ -f $RAW/quora_duplicate_questions.tsv ] && python -m ariadne.data prep --dataset qqp --raw $RAW/quora_duplicate_questions.tsv --embedder "$EMB_SPEC"
[ -f $RAW/collection.tsv ] && python -m ariadne.data prep --dataset msmarco    --raw $RAW/collection.tsv --embedder "$EMB_SPEC" --limit 1200000
ls $RAW/en_core_web_md-*.whl >/dev/null 2>&1 && python -m ariadne.data prep --dataset spacy --raw $(ls $RAW/en_core_web_md-*.whl | head -1) --name spacy
[ -f $RAW/glove-100-angular.hdf5 ] && python -m ariadne.data prep --dataset hdf5 --raw $RAW/glove-100-angular.hdf5 --name glove100

# toy duplicate clusters for the smoke test of A3 (stands in for QQP)
if [ -f data/agnews.texts.jsonl ]; then
  python experiments/make_toy_pairs.py data/agnews.texts.jsonl data/toyqqp.jsonl
  python -m ariadne.data prep --dataset jsonl --raw data/toyqqp.jsonl --name toyqqp --embedder "$EMB_SPEC"
fi

# locality statistics of every prepared file (use them to choose tau_cos per embedder)
for f in data/*__*.npz; do echo "== $f"; python -m ariadne.data stats --npz $f; done
