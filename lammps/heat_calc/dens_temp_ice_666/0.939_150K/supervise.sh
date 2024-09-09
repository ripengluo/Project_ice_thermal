#!/usr/bin

i=0
str="0.939_15"
file="corrs/corr.99.data"
while true
do
squeue > tmp_queue
status_="stopped"
while read line
do
    #echo $line,'loop' $i
    if [[ $line == *$str* ]]
    then
        #echo 'include' 
        status_="run"
   # else
        #echo 'exclude'
    fi
    ((i+=1))
done<tmp_queue
echo $status_"0.939_150K"
if [ $status_ == "stopped" ] ;then
    if [ ! -f "$file" ] ; then
        echo 'submit'
        spylammps -np 24 -walltime 24:00:00 -mem 64000mb -j 0.939_150K_lrp prog_restart.py
    else break
    fi
elif [ $status_ == "run" ]
then echo 'idle'
fi
rm tmp_queue
sleep 20s
done
