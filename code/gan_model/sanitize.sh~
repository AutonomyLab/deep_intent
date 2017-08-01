#!/bin/sh
filename="counter.txt"

cat $filename | while IFS='' read num
do
  folder="./generated_images/gen_$num/"
  mkdir $folder
  count=$((num+1))
  echo $count >  $filename
  mv "./generated_images/"*.png $folder
  mv "./logs/"*.json $folder
  rm "./tf_logs/"*
done

