#!/bin/bash

status_="stopped"
n=10
if n==10
then
status_="run"
fi
echo $status_
if [ $status_ = "stopped" ]
then
    echo spylammps
elif [ $status_ = "run" ]
then
    echo do_noting
fi
#echo $status_
