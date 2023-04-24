# Segment Anything Model Applied to Medical Images

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
   1. [Commands](#commands)
3. [Workspace Configuration](#workspace-configuration)
4. [Dataset](#dataset)
5. [Working Data](#working-data)
6. [Image Processing](#image-processing)

## Introduction

Test [Segment Anything Model (SAM)][sam] performance when working with medical images.

[sam]: https://segment-anything.com/ "Segment Anything Model (SAM): a new AI model from Meta AI that can \"cut out\" any object, in any image, with a single click"

## Requirements

### Commands

- `miniconda` to manage Python virtual environments.
- `wget` to download files.
- `unzip` to unpack compressed files.

## Workspace Configuration

1. Install miniconda:

    ```shell
    $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    ```

2. Create a new environment:

    ```shell
    $ conda create --name sam python=3.9
    ```

    In this command, **sam** stands for **Segment Anything Model**.

3. Deactivate any previous environment and activate the new one:

    ```shell
    $ conda deactivate
    $ conda activate sam
    ```

4. Install PyTorch, Torchvision, and Torchaudio:

    ```shell
    $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

5. Install SAM:

    ```shell
    $ pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

6. Install SAM's requirements:

    ```shell
    $ pip install opencv-python pycocotools matplotlib onnxruntime onnx
    ```

    > **Note:** I'm not totally sure these requirements are mandatory. They may be for the examples, but this project doesn't need OpenCV, for example.

7. Execute the shell script **scripts/download_checkpoints.sh** to get SAM's model checkpoints:

    ```shell
    $ sh ./scripts/download_checkpoints.sh
    ```

8. Configure PyCharm. If you are working on Windows, make sure you use WSL and that your interpreter is also based on WSL.

    > **Note:** Be careful with the EOL configuration in you IDE. Always choose LF instead of CRLF to avoid compatibility issues, as this is a multiplatform project.

> **Note:** If you no longer need the Conda environment, just delete it with `conda remove -n sam --all`.

## Dataset

- **Location:** [Zenodo][dataset].
- **Description:** 3520 slices from 20 patients.

Execute the script **scripts/download_dataset.sh** from the root folder of this project download the dataset:

```shell
$ sh ./scripts/download_dataset.sh
```

[dataset]: https://zenodo.org/record/3757476 "COVID-19 CT Lung and Infection Segmentation Dataset"

## Working Data

The project includes various run/debug configurations. In order to create the working data from the dataset, you need to execute **nifti_to_numpy** with the required arguments. As a reference, this is the command you have to execute from the command line to create the NumPy files from the NIfTI ones, for one image:

```shell
$ conda run -n sam --no-capture-output python nifti_to_numpy.py --input_file_path datasets/zenodo/COVID-19-CT-Seg_20cases/coronacases_001.nii.gz --output_file_path working_data/image_coronacases_001.npy

$ conda run -n sam --no-capture-output python nifti_to_numpy.py --input_file_path datasets/zenodo/Lung_Mask/coronacases_001.nii.gz --output_file_path working_data/masks_coronacases_001.npy
```

## Image Processing

You can process just a slide from a CT image:

```shell
$ conda run -n sam --no-capture-output python process_image.py --image_file_path working_data/image_coronacases_001.npy --masks_file_path working_data/masks_coronacases_001.npy --slice 122 --debug 
```

You can also process the whole CT image:

```shell
$ conda run -n sam --no-capture-output python process_image.py --image_file_path working_data/image_coronacases_001.npy --masks_file_path working_data/masks_coronacases_001.npy --debug 
```
