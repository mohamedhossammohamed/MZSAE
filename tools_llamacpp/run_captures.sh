#!/bin/bash
# STEP 3 batch: 4 docs x (8k,16k) + docA 32k, 20 decode steps. 10-min cap per run.
M=work_gguf/qwen2.5-0.5b-ollama.gguf
T0=$(date +%s)
run_one() {
  doc=$1; n=$2; out=captures/q4_${doc}_${n}
  echo "=== $doc @ $n tokens -> $out ==="
  S=$(date +%s)
  timeout 600 ./tools_llamacpp/capture_qkv --model $M --text work_docs/${doc}.txt \
    --tokens $n --decode 20 --chunk 2048 --out $out >> logs/mzsae_q4_capture.log 2>&1
  rc=$?
  E=$(date +%s)
  echo "done $doc@$n rc=$rc elapsed=$((E-S))s rss_GB=$(ps -o rss= -p $$ | awk '{print $1/1e6}')"
  [ $rc -ne 0 ] && echo "FAILED $doc@$n rc=$rc"
}
for d in docA docB docC docD; do run_one $d 8192; done
for d in docA docB docC docD; do run_one $d 16384; done
run_one docA 32768
E=$(date +%s)
echo "BATCH TOTAL $((E-T0))s"
