# Experiments

This folder contains different scripts that we are not specially proud of, but that accomplished a specific goal. Usually, they are rough modifications of existing scripts, but they can be almost anything.

- **process_image_moving_negative_prompt.py**: version of **process_image.py**, specifically changed to move the negative prompt along the vertical axis of the bounding box, keeping the same column (x) but changing the row (y). It is intended to work with the montgomery sample **MCUCXR_0251_1**, the one with the worst result. The idea was to see how can the results be improved just by moving that negative prompt. Unfortunately, not much.
- **process_image_multiple_negative_prompts.py**: version of **process_image.py**, specifically changed to use multiple hard-coded negative prompts. It is intended to work with the montgomery sample **MCUCXR_0002_0**.
