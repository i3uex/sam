#!/usr/bin/bash
for slice in {0..301}; do
  /home/andy/miniconda3/bin/conda run -n sam --no-capture-output python main.py --image_file_path working_data/image_coronacases_001.npy --masks_file_path working_data/masks_coronacases_001.npy --slice "$slice" --debug
done
