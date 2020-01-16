#!/bin/bash

TYPE=$1
AUDIO=$2
SPECT=$3
MIN=${4:-100}
MAX=${5:-16000}
BANDS=${6:-80}

for f in $AUDIO/$TYPE/*/*.wav
do
    b="${f##*/}"
    p="${f%/*}"
    sp="${p##*/}"

    mkdir -p $SPECT/$TYPE/$sp 2> /dev/null

    out="$SPECT/$TYPE/$sp/${b%.*}"

    if [ ! -f "$out" ]; then
        echo "Making Spectrogram $out"
        if ! ./tf_melspect.py -i "$f" -b ${BANDS} -m ${MIN} -M ${MAX} -o "$out.png"; then
            echo "Failed making ${out} - exiting now"
            exit $?
        fi
    fi
done
