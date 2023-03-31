#!/bin/bash

for DATA in BZR_MD COX2_MD PTC_FR MUTAG KKI DHFR_MD  DD NCI1 PROTEINS IMDB-BINARY IMDB-MULTI COLLAB
do
  echo "The accuracy results for ${DATA} are as follows:"
  python calculate_average_accuracy.py --dataset ${DATA}
done