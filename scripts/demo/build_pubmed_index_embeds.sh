#!/bin/bash

set -e
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

BASE_ARTIFACTS=/work/frink/deyoung.j/demo/rrnlp_demo/expts_embeds/
gpu_preamble="srun -p gpu --gres gpu:v100-sxm2:1 --mem 16G --cpus-per-task 2 --time 4:00:00 "
MAX_JOBS=8
#MAX_JOBS=2
#gpu_preamble="srun -p frink --gres gpu:1 --mem 16G --cpus-per-task 2 --time 4:00:00 "
#for csv in /scratch/deyoung.j/trialstreamer_csv_processing_splits/{1304..0}.csv ; do
#gpu_preamble="srun -p 177huntington --gres gpu:1 --mem 32G --cpus-per-task 2 --time 4:00:00 "
#model=facebook/contriever
#for model in facebook/contriever ; do
#for model in facebook/contriever facebook/contriever-msmarco ; do
for model in facebook/contriever facebook/contriever-msmarco google-bert/bert-base-uncased ; do
    ARTIFACTS=${BASE_ARTIFACTS}/$model
    OUTPUTS=$ARTIFACTS
    mkdir -p $ARTIFACTS
    FAILED_FILE=$ARTIFACTS/failed_pubmed_index
    rm -f $FAILED_FILE
    mkdir -p $ARTIFACTS/
    #json_index=$ARTIFACTS/contriever/pmids.json
    for nxml in /scratch/deyoung.j/pubmed/*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz; do
        echo $nxml
        json_index=$OUTPUTS/$(basename $nxml).json
        np_embeds=$OUTPUTS/$(basename $nxml).npy
        #cmd="$gpu_preamble -J $(basename $nxml) \
        #$PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/process_pubmed_nxmls.py \
        #    --input_gzs $nxml \
        #    --output_csv $OUTPUTS/$(basename $nxml).parquet \
        #"
        #ckpt "$cmd" $(basename $nxml) &
        #sleep 1s
        cmd="$gpu_preamble -J $(basename $nxml | sed -e 's/pubmed24n//' | sed -e 's/.xml.gz//') \
        $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/embed_title_abstracts.py \
            --input_gzs $nxml \
            --batch_size 256 \
            --json_index $json_index \
            --np_embeds $np_embeds \
            --model $model \
        " 
        ckpt "$cmd" "$(basename $nxml)/extract_embeds" &
        sleep 1s
        #allow_fail_ckpt "$cmd" "$(basename $nxml)/extract_embeds" &
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
    if [ -e $FAILED_FILE ] ; then
        echo "found $FAILED_FILE"
        exit 1
    fi
    jsons=""
    nps=""
    for nxml in /scratch/deyoung.j/pubmed/*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz; do
        echo $nxml
        json_index=$OUTPUTS/$(basename $nxml).json
        np_embeds=$OUTPUTS/$(basename $nxml).npy
        nps="${nps} $np_embeds"
        jsons="${jsons} $json_index"
    done
    for index_type in cosine dot ; do
        ckpt "srun -p frink -w d3089 --cpus-per-task 8 --mem 96G --time 96:00:00 -J embed$model \
            $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/read_np_files_to_annoy.py \
                --json_index $jsons \
                --np_embeds $nps \
                --output_index_file $OUTPUTS/annoy_${index_type}.bin \
                --output_json_index_file $OUTPUTS/annoy_${index_type}.json \
                --metric $index_type \
        " combine_embeds_annoy_${index_type} &
    done
done

wait

exit 1

#for model in facebook/contriever facebook/contriever-msmarco google-bert/bert-base-uncased ; do
for model in facebook/contriever ; do
    ARTIFACTS=${BASE_ARTIFACTS}/$model
    OUTPUTS=$ARTIFACTS
    mkdir -p $ARTIFACTS
    FAILED_FILE=$ARTIFACTS/failed_pubmed_index
    rm -f $FAILED_FILE
    mkdir -p $ARTIFACTS/
    #json_index=$ARTIFACTS/contriever/pmids.json
    jsons=""
    nps=""
    for nxml in /scratch/deyoung.j/pubmed/*xml.gz ; do
        echo $nxml
        json_index=$OUTPUTS/$(basename $nxml).json
        np_embeds=$OUTPUTS/$(basename $nxml).npy
        nps="${nps} $np_embeds"
        jsons="${jsons} $json_index"
    done
    ckpt "$srun -p frink --cpus-per-task 4 --mem 48G --time 24:00:00 -J embed$model \
        $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/read_np_files_to_annoy.py \
            --json_index $jsons \
            --np_embeds $nps \
            --output_index_file $OUTPUTS/annoy.bin \
            --output_json_index_file $OUTPUTS/annoy.json \
    " combine_embeds
done

exit 1

#cmd="srun -p 177huntington --mem 64G --cpus-per-task 4 --time 4:00:00 -J combine_indexes $PYTHON \
#    /work/frink/deyoung.j/demo/rrnlp_demo/merge_indexes.py \
#    --input_indexes /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to273 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to568 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to610   /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to793 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to1164 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin.to1219 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index_updates.bin \
#    --output_index /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/faiss_index.bin \
#    --input_json_indexes /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to273  /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to568  /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to610  /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to793 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to1164 /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json.to1219   /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids_updates.json \
#    --output_json_index /work/frink/deyoung.j/demo/rrnlp_demo/expts_screening/contriever/pmids.json \
#    "
#ckpt "$cmd" combine_indexes
#
