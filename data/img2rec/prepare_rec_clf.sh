#!/bin/bash
#dataDir=$1
dataDir='Dataset/Classification/PlantingSeeds'
#prefix=$2
prefix=seeds
mkdir -p ./RecDataSet
python ./src/im2rec.py $dataDir"/"$prefix $dataDir --recursive --train-ratio 0.7 --test-ratio 0.1 --list 
python ./src/im2rec.py  $dataDir $dataDir --num-thread 8 
mv $dataDir"/"$prefix* ./Rec_dataset
echo "[INFO] RecDataSet prepared done! "
