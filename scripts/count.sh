#! /bin/bash

cat _original/**/**/vi*.txt >> all.tmp.txt

while read a; do
    echo $a | awk '{print $1}' >> all_filtered.tmp.txt
done < all.tmp.txt

cat all_filtered.tmp.txt | sort | uniq -c

rm all.tmp.txt all_filtered.tmp.txt