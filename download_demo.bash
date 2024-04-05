#!/usr/bin/env bash

DATA_DIR=$1
mkdir -p ${DATA_DIR}

if [ -f "demo_video_6fps.zip" ]; then
    echo "demo_video_6fps.zip exists, skip downloading!"
else
    echo "Downloading demo_video_6fps.zip."
    wget wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1mOTXPOBRGXgoAycgZ4pLZVCCKglJAyOC' -O demo_video_6fps.zip

fi

echo "Processing videos started."
unzip -q demo_video_6fps.zip -d ${DATA_DIR}
mkdir -p "${DATA_DIR}/demo/videos_6fps/" && find "${DATA_DIR}/demo_video_6fps" -name "*.mp4" -exec mv {} "${DATA_DIR}/demo/videos_6fps/" \;
echo "Processing videos completed."

rm -rf "${DATA_DIR}/demo_video_6fps"
rm demo_video_6fps.zip
echo "The preparation of the msrvtt dataset has been successfully completed."