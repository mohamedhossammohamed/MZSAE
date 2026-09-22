#!/bin/bash
# Recapture all 9 runs with the fp32 tool. Sequential (GPU-bound). 10-min cap each.
M=work_gguf/qwen2.5-0.5b-ollama.gguf
T0=$(date +%s)
run_one() {
  doc=$1; n=$2; out=captures/q4_${doc}_${n}
  rm -rf $out
  S=$(date +%s)
  timeout 600 ./tools_llamacpp/capture_qkv --model $M --text work_docs/${doc}.txt \
    --tokens $n --decode 20 --chunk 2048 --out $out >> logs/mzsae_q4_capture2.log 2>&1
  rc=$?; E=$(date +%s)
  echo "done $doc@$n rc=$rc elapsed=$((E-S))s"
  [ $rc -ne 0 ] && echo "FAILED $doc@$n rc=$rc"
}
for d in docA docB docC docD; do run_one $d 8192; done
for d in docA docB docC docD; do run_one $d 16384; done
run_one docA 32768
echo "RECAPTURE TOTAL $(( $(date +%s) - T0 ))s"
