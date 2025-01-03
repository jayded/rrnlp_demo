#!/bin/bash

set -e
#set -x
PYTHONPATH=$(readlink -e .):$PYTHONPATH
export PYTHONPATH
PYTHON=/work/frink/deyoung.j/demo/rrnlp_demo/venv/bin/python
set -o nounset
set -o pipefail
MAX_JOBS=20

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
        echo "touch $ckpt_file" >> "$partial_ckpt_file"
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

OG_ARTIFACTS=/work/frink/deyoung.j/demo/rrnlp_demo/expts
ARTIFACTS=/work/frink/deyoung.j/demo/rrnlp_demo/expts_reprocessed
OUTPUTS=$ARTIFACTS
mkdir -p $ARTIFACTS
FAILED_FILE=$ARTIFACTS/failed_nxmls
rm -f $ARTIFACTS/failed_nxmls*
rm -f $FAILED_FILE
sqlite_db="$ARTIFACTS/pubmed_extractions.db"
echo "sqlite $sqlite_db"

#MAX_JOBS=10
MAX_JOBS=50
#MAX_JOBS=45
#gpu_preamble="srun -p short --mem 32G --cpus-per-task 8 --time 14:45:00 "
FAILED_FILE=$ARTIFACTS/failed_nxmls_short

#gpu_preamble="srun -p gpu --gres gpu:a100:1 --mem 32G --cpus-per-task 4 --time 4:00:00 "
#gpu_preamble="srun -p gpu --gres gpu:v100-pcie:1 --mem 32G --cpus-per-task 4 --time 4:00:00 "
#FAILED_FILE=$ARTIFACTS/failed_nxmls_gpu_v100
gpu_preamble="srun -p frink --gres gpu:1 --mem 16G --cpus-per-task 4 --time 24:00:00 "
#gpu_preamble="srun -p frink --mem 20G --cpus-per-task 4 --time 24:00:00 "
#gpu_preamble="srun -p gpu --gres gpu:v100-sxm2:1 --mem 16G --cpus-per-task 2 --time 7:00:00 "
#FAILED_FILE=$ARTIFACTS/failed_nxmls_gpu
#FAILED_FILE=$ARTIFACTS/failed_nxmls_frink_gpu1
#FAILED_FILE=$ARTIFACTS/failed_nxmls_frink_cpu
#MAX_JOBS=3
#gpu_preamble="srun -p 177huntington --gres gpu:1 --mem 32G --cpus-per-task 2 --time 10:00:00 "
#FAILED_FILE=$ARTIFACTS/failed_nxmls_177
#for nxml in /scratch/deyoung.j/pubmed/*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/*{1199..0800}*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/*{0000..0699}*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/*{0800..1000}*xml.gz ; do

#MAX_JOBS=2
#gpu_preamble="srun -p frink --gres gpu:1 --mem 64G --cpus-per-task 2 --time 8:00:00 "
#FAILED_FILE=$ARTIFACTS/failed_nxmls_frink_cpu
#FAILED_FILE=$ARTIFACTS/failed_nxmls_frink_2
#for nxml in /scratch/deyoung.j/pubmed/*{1199..1100}*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n12*xml.gz /scratch/deyoung.j/pubmed/pubmed24n11*xml.gz /scratch/deyoung.j/pubmed/pubmed24n10*xml.gz /scratch/deyoung.j/pubmed/pubmed24n09*xml.gz /scratch/deyoung.j/pubmed/pubmed24n08*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz /scratch/deyoung.j/pubmed/*xml.gz  ; do

#MAX_JOBS=6
#gpu_preamble="srun -p gpu --gres gpu:a100:1 --mem 32G --cpus-per-task 2 --time 7:00:00 "
#FAILED_FILE=$ARTIFACTS/failed_nxmls_gpu_a100
#FAILED_FILE=$ARTIFACTS/failed_nxmls_gpu
#gpu_preamble="srun -p gpu --gres gpu:v100-sxm2:1 --mem 16G --cpus-per-task 2 --time 7:00:00 "
#for nxml in /scratch/deyoung.j/pubmed_updates/*xml.gz /scratch/deyoung.j/pubmed/pubmed24n12*xml.gz /scratch/deyoung.j/pubmed/pubmed24n11*xml.gz /scratch/deyoung.j/pubmed/pubmed24n10*xml.gz /scratch/deyoung.j/pubmed/pubmed24n09*xml.gz /scratch/deyoung.j/pubmed/pubmed24n08*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz /scratch/deyoung.j/pubmed/*xml.gz  ; do

#for nxml in /scratch/deyoung.j/pubmed/pubmed24n05*xml.gz /scratch/deyoung.j/pubmed/pubmed24n04*xml.gz /scratch/deyoung.j/pubmed/pubmed24n08*xml.gz /scratch/deyoung.j/pubmed/*xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0350..0450}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0600..0800}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1219..0600}.xml.gz  ; do

#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0873..0850}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{0799..0669}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{1219..0001}.xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0001..1219}.xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0301..0400}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{0430..1219}.xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0350..0303}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0293..0666}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{0900..1219}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0799..0515}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0800..0850}.xml.gz  /scratch/deyoung.j/pubmed/pubmed24n{1219..0850}.xml.gz ; do
# TODO restart this with more in the morning if it needs?
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0530..0612}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0350..0530}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0659..0900}.xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0293..0347}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1050..1010}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{1070..1050}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0886..0700}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{0887..0950}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1219..0001}.xml.gz  ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0001..0717}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{1219..1100}.xml.gz /scratch/deyoung.j/pubmed/pubmed24n{0718..1219}.xml.gz //scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1043..1219}.xml.gz ; do
for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1219..1219}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0680..1219}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed_updates/*xml.gz ; do

# TODO: run these on frink
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{1210..1100}.xml.gz /scratch/deyoung.j/pubmed_updates/pubmed24n{1592..1220}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed/pubmed24n{0950..1100}.xml.gz /scratch/deyoung.j/pubmed_updates/pubmed24n{1220..1592}.xml.gz ; do
    echo $nxml
        #--previous_output_jsonls $OG_ARTIFACTS \
    cmd="$gpu_preamble -J $(basename $nxml | sed -e 's/pubmed24n//' | sed -e 's/.xml.gz//') \
    $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/process_pubmed_nxmls.py \
        --input_gzs $nxml \
        --output_csv $OUTPUTS/$(basename $nxml).parquet \
        --additional_ids /work/frink/deyoung.j/demo/cochrane_high_recall_rcts.txt \
    "
    #ckpt "$cmd" $(basename $nxml) 
    allow_fail_ckpt "$cmd" $(basename $nxml) &
    #allow_fail_ckpt "$cmd" $(basename $nxml)
    sleep 2s
    sleep $(jobs | wc -l)
    #cmd="srun -p short -J $(basename $nxml | sed -e 's/pubmed//') \
    cmd="srun -p frink -J $(basename $nxml | sed -e 's/pubmed//') \
        $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/fix_pubmed_outputs.py \
            --input_parquets $OUTPUTS/$(basename $nxml).parquet \
            --input_jsonls $OUTPUTS/$(basename $nxml).parquet.jsonl \
            --output_database $sqlite_db \
    "
    #ckpt "$cmd" "$(basename $nxml)/outputs_to_db"
    #cmd="srun -p short -J $(basename $nxml) \
    #$PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/extract_title_abstracts.py \
    #    --input_gzs $nxml \
    #    --output_json /work/frink/deyoung.j/demo/titles_abstracts.json \
    #" 
    #ckpt "$cmd" "$(basename $nxml)/extract_tiab"
    #sleep 1s
    while [ "$(jobs | wc -l | xargs)" -ge $MAX_JOBS ];do
        echo "$(jobs | wc -l | xargs)"
        sleep $(jobs | wc -l)
    done
    if [ -e $FAILED_FILE ] ; then
        echo "found $FAILED_FILE"
        #exit 1
    fi
done

if [ -e $FAILED_FILE ] ; then
    echo "found $FAILED_FILE"
    #exit 1
fi

#for nxml in /scratch/deyoung.j/pubmed_updates/pubmed24n{1400..1596}.xml.gz ; do
#for nxml in /scratch/deyoung.j/pubmed_updates/pubmed24n{1220..1607}.xml.gz ; do
for nxml in /scratch/deyoung.j/pubmed_updates/pubmed24n{1630..1640}.xml.gz ; do
    echo $nxml
        #--previous_output_jsonls $OG_ARTIFACTS \
    cmd="$gpu_preamble -J $(basename $nxml | sed -e 's/pubmed24n//' | sed -e 's/.xml.gz//') \
    $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/process_pubmed_nxmls.py \
        --input_gzs $nxml \
        --output_csv $OUTPUTS/$(basename $nxml).parquet \
        --updates \
        --additional_ids /work/frink/deyoung.j/demo/cochrane_high_recall_rcts.txt \
    "
    ckpt "$cmd" $(basename $nxml) 
    #allow_fail_ckpt "$cmd" $(basename $nxml) &
    #allow_fail_ckpt "$cmd" $(basename $nxml)
    sleep 0.5s
    #sleep $(jobs | wc -l)
    #cmd="srun -p short -J $(basename $nxml | sed -e 's/pubmed//') \
    cmd="srun -p frink -J $(basename $nxml | sed -e 's/pubmed//') \
        $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/fix_pubmed_outputs.py \
            --input_parquets $OUTPUTS/$(basename $nxml).parquet \
            --input_jsonls $OUTPUTS/$(basename $nxml).parquet.jsonl \
            --output_database $sqlite_db \
            --delete_ids $OUTPUTS/$(basename $nxml).parquet.delete_ids.json \
    "
    ckpt "$cmd" "$(basename $nxml)/outputs_to_db"
    #cmd="srun -p short -J $(basename $nxml) \
    #$PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/extract_title_abstracts.py \
    #    --input_gzs $nxml \
    #    --output_json /work/frink/deyoung.j/demo/titles_abstracts.json \
    #" 
    #ckpt "$cmd" "$(basename $nxml)/extract_tiab"
    #sleep 1s
    while [ "$(jobs | wc -l | xargs)" -ge $MAX_JOBS ];do
        echo "$(jobs | wc -l | xargs)"
        sleep $(jobs | wc -l)
    done
    if [ -e $FAILED_FILE ] ; then
        echo "found $FAILED_FILE"
        #exit 1
    fi
done

wait
exit 1

cmd="srun -p frink -J delete_pmids \
    $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/delete_pmids.py \
        --delete_jsons $OUTPUTS/*.delete_ids.json \
        --output_database $sqlite_db \
"
ckpt "$cmd" "delete_pmids"

#sqlite_db="$ARTIFACTS/expts/pubmed_extractions.db"
#for nxml in /scratch/deyoung.j/pubmed/*xml.gz /scratch/deyoung.j/pubmed_updates/*xml.gz ; do
#    echo $nxml
#    cmd="srun -p short --time 1:00:00 --mem 4G -J $(basename $nxml) \
#    $PYTHON /work/frink/deyoung.j/demo/rrnlp_demo/scripts/demo/fix_pubmed_outputs.py \
#        --input_parquets $OUTPUTS/$(basename $nxml).parquet \
#        --input_jsonls $OUTPUTS/$(basename $nxml).parquet.jsonl \
#        --output_database $sqlite_db \
#    "
#    ckpt "$cmd" $(basename $nxml)/to_sqlite
#    sleep 0.5s
#    #while [ "$(jobs | wc -l | xargs)" -ge $MAX_JOBS ];do
#    #    echo "$(jobs | wc -l | xargs)"
#    #    sleep $(jobs | wc -l)
#    #done
#    #if [ -e $FAILED_FILE ] ; then
#    #    echo "found $FAILED_FILE"
#    #    exit 1
#    #fi
#done
#
## TODO combine nxmls
#
