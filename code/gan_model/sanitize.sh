#!/bin/sh
filename="counter.txt"

cat $filename | while IFS='' read num
do
  if [ -f ./generated_images/*.png ]; then
     folder="./generated_images/gen_$num/"
     mkdir $folder
     count=$((num+1))
     echo $count >  $filename
     mv "./generated_images/"*.png $folder
     mv "./logs/"*.json $folder
  else
     echo "File not found"
  fi
  rm "./tf_logs/"*
  rm "./checkpoints/"*
done

