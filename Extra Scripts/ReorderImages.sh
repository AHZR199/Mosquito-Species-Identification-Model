#!/bin/bash
mv /work/soghigian_lab/abdullah.zubair/Resnet/model/thesis/project/2023/code/ImageBase/AWS_sync/cropped/* /work/soghigian_lab/abdullah.zubair/Resnet/model/thesis/project/2023/code/ImageBase/
rm -r /work/soghigian_lab/abdullah.zubair/Resnet/model/thesis/project/2023/code/ImageBase/AWS_sync

BASE_DIR="/work/soghigian_lab/abdullah.zubair/Resnet/model/thesis/project/2023/code/ImageBase"
cd "$BASE_DIR"
for genus in *; do
  if [ -d "$genus" ]; then
    cd "$genus"
    for species in *; do
      if [ -d "$species" ]; then
        mkdir -p "${BASE_DIR}/${genus}_${species}"
        mv "$species"/* "${BASE_DIR}/${genus}_${species}/"
        rmdir "$species"
      fi
    done
    cd "$BASE_DIR"
    rmdir "$genus"
  fi
done
