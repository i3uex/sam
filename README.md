
```shell
conda create --name sam python=3.9
conda activate sam
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

nifti_to_numpy:

--input_file_path datasets/zenodo/COVID-19-CT-Seg_20cases/coronacases_001.nii.gz --output_file_path working_data/image_coronacases_001.npy --transpose --rotate
--input_file_path datasets/zenodo/Lung_Mask/coronacases_001.nii.gz --output_file_path working_data/masks_coronacases_001.npy --transpose --rotate

main:

--image_file_path working_data/image_coronacases_001.npy --masks_file_path working_data/masks_coronacases_001.npy --slice 122 --dry_run --debug
