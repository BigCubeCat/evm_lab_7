#!/bin/bash

OUT=`(time ./$1 < test.input > test.output) 2>&1 | grep real`

echo "$OUT"

