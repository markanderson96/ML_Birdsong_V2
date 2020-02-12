#!/bin/bash

TYPE=$1
AUDIO=$2
SPECT=$3
MIN=${4:-100}
MAX=${5:-16000}
BANDS=${6:-80}
SPECT_TYPE=${7:-linear}

for f in $AUDIO/$TYPE/*/*.wav
do
    b="${f##*/}"
    p="${f%/*}"
    sp="${p##*/}"

    mkdir -p $SPECT/$TYPE/$SPECT_TYPE/$sp 2> /dev/null

    out="$SPECT/$TYPE/$SPECT_TYPE/$sp/${b%.*}"

    if [ ! -f "$out" ]; then
        echo "Making Spectrogram $out"
        if ! ./tf_melspect.py -i "$f" -t ${SPECT_TYPE} -o "$out.png" -d 2 -p -f 100 -O 50 -m 100 -M 11000 --filter; then
            echo "Failed making ${out} - exiting now"
            exit $?
        fi
    fi
done
