#!/bin/bash

#set -e
#set -x
PYTHONPATH=$(readlink -e .):$PYTHONPATH
export PYTHONPATH
#PYTHON=/work/frink/deyoung.j/demo/venv/bin/python
PYTHON=/work/frink/deyoung.j/demo/rrnlp_demo/venv/bin/python
set -o nounset
set -o pipefail
MAX_JOBS=15

function skip_ckpt {
    local cmd="$1"
    local name="$2"
    echo "skipping running job $name"
}

# when failure really is an option!
# like when I want to a pile of training jobs sequentially and don't have time to monitor them.
function allow_fail_ckpt {
    local cmd="$1"
    local name="$2"
    local ckpt_file="$ARTIFACTS/logs/$name.ckpt"
    local partial_ckpt_file="$ARTIFACTS/logs/$name.partial"
    local log_file_base="$ARTIFACTS/logs/$name"
    mkdir -p "$(dirname $ckpt_file)" "$(dirname $log_file_base)"
    if [ -e "$partial_ckpt_file" ] ; then
        cat "$partial_ckpt_file" >> "$partial_ckpt_file".old
    fi
    if [ ! -e "$ckpt_file" ] ; then
        echo "running $name; $cmd"
        echo "$cmd" > "$partial_ckpt_file"
        if [ -e "${log_file_base}.e" ]; then
            mv "${log_file_base}.e" "${log_file_base}.e.old"
        fi
        if [ -e "${log_file_base}.o" ]; then
            mv "${log_file_base}.o" "${log_file_base}.o.old"
        fi
        # shellcheck disable=SC2086
        eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; echo "$cmd;" "$log_file_base."{e.o} >> $FAILED_FILE ; echo "to skip: touch $ckpt_file"; true) || true
    fi
}

function ckpt {
    local cmd="$1"
    local name="$2"
    local ckpt_file="$ARTIFACTS/logs/$name.ckpt"
    local partial_ckpt_file="$ARTIFACTS/logs/$name.partial"
    local log_file_base="$ARTIFACTS/logs/$name"
    mkdir -p "$(dirname $ckpt_file)" "$(dirname $log_file_base)"
    if [ -e "$partial_ckpt_file" ] ; then
        cat "$partial_ckpt_file" >> "$partial_ckpt_file".old
    fi
    if [ ! -e "$ckpt_file" ] ; then
        echo "running $name; $cmd"
        echo "$cmd" > "$partial_ckpt_file"
        if [ -e "${log_file_base}.e" ]; then
            mv "${log_file_base}.e" "${log_file_base}.e.old"
        fi
        if [ -e "${log_file_base}.o" ]; then
            mv "${log_file_base}.o" "${log_file_base}.o.old"
        fi
        # shellcheck disable=SC2086
        #eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; exit 1)
        eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; echo "$cmd;" "$log_file_base."{e.o} >> $FAILED_FILE ; echo "to skip: touch $ckpt_file"; exit 1)
        #else
        #echo "already ran '$name'; clear '$ckpt_file' to rerun"
        fi
}

ARTIFACTS=/work/frink/deyoung.j/demo/rrnlp_demo/expts_screening
OUTPUTS=$ARTIFACTS
mkdir -p $ARTIFACTS
FAILED_FILE=$ARTIFACTS/failed_pubmed_updates_index
rm -f $FAILED_FILE
#MAX_JOBS=2
#gpu_preamble="srun -p frink --gres gpu:1 --mem 16G --cpus-per-task 2 --time 4:00:00 "
#for csv in /scratch/deyoung.j/trialstreamer_csv_processing_splits/{1304..0}.csv ; do
gpu_preamble="srun -p 177huntington --gres gpu:1 --mem 32G --cpus-per-task 2 --time 4:00:00 "
model=facebook/contriever
mkdir -p $ARTIFACTS/contriever
json_index=$ARTIFACTS/contriever/pmids_updates.json
faiss_index=$ARTIFACTS/contriever/faiss_index_updates.bin
for nxml in /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
    echo $nxml
    #cmd="$gpu_preamble -J $(basename $nxml) \
    #$PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/process_pubmed_nxmls.py \
    #    --input_gzs $nxml \
    #    --output_csv $OUTPUTS/$(basename $nxml).parquet \
    #"
    #ckpt "$cmd" $(basename $nxml) &
    #sleep 1s
    cmd="$gpu_preamble -J $(basename $nxml) \
    $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/build_pubmed_index_embeds.py \
        --input_gzs $nxml \
        --batch_size 512 \
        --json_index $json_index \
        --faiss_index $faiss_index \
        --model $model \
    " 
    ckpt "$cmd" "$(basename $nxml)/extract_embeds"
    while [ "$(jobs | wc -l | xargs)" -ge $MAX_JOBS ];do
        echo "$(jobs | wc -l | xargs)"
        sleep $(jobs | wc -l)
    done
    if [ -e $FAILED_FILE ] ; then
        echo "found $FAILED_FILE"
        exit 1
    fi
done

wait


