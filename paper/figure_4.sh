#!/usr/bin/bash

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 85 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_85_prediction.png letter

DISPLACE_NEGATIVE_PROMPT='True'
export DISPLACE_NEGATIVE_PROMPT

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 85 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_85_prediction.png letter/slice_85_prediction_improved.png
