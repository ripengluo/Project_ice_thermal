#!/usr/bin/env bash

n_parts=5
total=`ls corrs/ | wc -l`
npparts=$(($total/$n_parts))
dump_file=corrs
for ((i=0; i<$n_parts; i++))
do
    python3 calc_divide.py $i > heat_conductivity_$i'.dat'
    #ifiles=`ls $dump_file/ |sed -n "$(($i*$npparts+1)),$((($i+1)*$npparts))p"`
    #echo $ifiles[0]
done
