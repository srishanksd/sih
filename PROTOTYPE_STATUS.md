# VAJRA-CSDD Prototype Status

## What is ready

- MOSDAC Cherrapunji DWR NetCDF ingestion
- INSAT-3DR CTP/CTT HDF5 ingestion
- DBZ extraction and polar-to-Cartesian conversion
- paired DWR + INSAT temporal alignment
- 256x256 radar baseline and 128x128 multimodal fusion grid
- chronological event splitting
- GPU ConvLSTM training on RTX 5050
- held-out event replay
- 15/30/45/60 minute model forecasts
- storm-cell detection/tracking baseline
- hazard probability prototype
- Flask API
- live dashboard replay

## Demo

Start:

```powershell
cd D:\PS84
.\.venv\Scripts\python.exe main.py
```

Open:

```
http://127.0.0.1:5000
```

Use the slider at 0, 15, 30, 45 and 60 minutes, or press Play.

## Current honest scope

This is a **prototype demonstration**, not an operational 0–6 hour system.

The current trained multimodal prototype uses DWR plus INSAT-3DR CTP/CTT auxiliary channels. It is trained on paired Cherrapunji observations and predicts future radar fields autoregressively. ILDN, AWS and GFS are still not connected to the trained model.

The dashboard labels the result as a **historical replay** so it is not presented as live weather data. The current dashboard replay is specifically the Cherrapunji DWR domain.
