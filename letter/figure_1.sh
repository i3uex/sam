#!/usr/bin/bash

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 122 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_122_prediction.png letter

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_radiopaedia_10_85902_3.npy \
  --masks_file_path working_data/masks_radiopaedia_10_85902_3.npy \
  --slice 171 \
  --use_bounding_box \
  --debug
cp working_data/debug/image_radiopaedia_10_85902_3/slice_171_prediction.png letter
