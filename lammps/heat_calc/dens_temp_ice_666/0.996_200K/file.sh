#!/bin/bash

file="corrs/corr.09.data"
if [ -f "$file" ]
then    echo exist
else    echo not_exist
fi
