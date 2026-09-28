# VAJRA-CSDD Training

This directory contains the trainable ML stack. The dashboard is not the training system.

## Current model

The first learned model is a ConvLSTM radar nowcaster:

past radar sequence -> ConvLSTM -> future radar fields

Input: [batch, input_steps, channels, height, width]
Output: [batch, forecast_steps, channels, height, width]

The trainer is designed so additional modalities can later be added as channels or separate encoders without changing the checkpoint/data-split workflow.

## Data contract

Put preprocessed event arrays in data/processed/. Each .npy file should contain a chronological radar sequence with shape [time, height, width] or [time, 1, height, width] and values normalized to [0,1].

Do not mix timestamps randomly. Files should represent complete events or continuous periods. Splitting happens by file, preventing windows from the same event leaking between train and validation/test.

## Train

uv run python training/train.py

The trainer uses CUDA when a CUDA-enabled PyTorch build is installed, otherwise CPU. It saves last.pt, best.pt, and the split manifest.

## Scaling to more data

Add more event files to data/processed/ and rerun training. The dataset uses memory-mapped NumPy files and creates windows lazily, so the complete dataset is not loaded into RAM.

Later adapters will convert DWR/INSAT/ILDN/AWS/GFS data into this canonical representation. Keep raw source data untouched in data/raw/.