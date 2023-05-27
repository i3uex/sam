#!/usr/bin/bash

DEBUG_DRAW_SAM_PREDICTION='False'
DEBUG_DRAW_MASKS_CONTOURS='False'
DEBUG_DRAW_BOUNDING_BOX='False'
DEBUG_DRAW_NEGATIVE_PROMPT='False'

export DEBUG_DRAW_SAM_PREDICTION
export DEBUG_DRAW_MASKS_CONTOURS
export DEBUG_DRAW_BOUNDING_BOX
export DEBUG_DRAW_NEGATIVE_PROMPT

MOVE_CENTROIDS_INSIDE_MASKS='True'
export MOVE_CENTROIDS_INSIDE_MASKS

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 98 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_98_prediction.png letter/centroid_inside.png

MOVE_CENTROIDS_INSIDE_MASKS='False'
export MOVE_CENTROIDS_INSIDE_MASKS

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 98 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_98_prediction.png letter/centroid_outside.png
