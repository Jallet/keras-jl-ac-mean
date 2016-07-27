#!/usr/bin/env bash
if [ $# -ne 1 ]
then
  echo "Take exact 1 param"
  exit
fi

file=$1
rm nohup.out
mkdir $file
mv acc $file/
mv val_acc $file/
mv loss $file/
mv val_loss $file/
