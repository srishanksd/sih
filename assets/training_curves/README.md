# Model evidence for SIH PPT

All five PNG files in this directory have been generated and verified on the local machine.

- `radar_convlstm.png`: three-epoch diagnostic training/validation loss on DWR data (12 train, 6 validation windows).
- `dwr_insat_convlstm.png`: three-epoch diagnostic training/validation loss (12 train, 1 validation window).
- `dwr_insat_gfs_convlstm.png`: three-epoch diagnostic training/validation loss (12 train, 1 validation window).
- `aws_validation_curve.png`: AWS ExtraTrees validation error versus tree count.
- `gfs_heldout_evidence.png`: existing production-checkpoint held-out normalized MAE against persistence.

IMPORTANT: The three neural curves are lightweight retraining diagnostics (32x32 input, hidden channels 8/8, 3 epochs). They are NOT loss histories from the original full-resolution, 20-epoch production models. The original checkpoints did not retain complete loss histories. The multimodal diagnostic validation splits contain only one sampled window, so these plots must NOT be used to claim generalization or compare model performance. Do not compare the neural loss values across different datasets as a model ranking.

JSON alongside each PNG contains the underlying numeric data. `diagnostic_run.log` records the actual diagnostic training output. Script: `training/quick_curves.py`.
