# VAJRA-CSDD Prototype Status

## What is ready

- MOSDAC Cherrapunji DWR NetCDF ingestion
- DBZ extraction and polar-to-Cartesian conversion
- 256x256 radar frames
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

The current trained model is radar-only and uses a small set of historical DWR event sequences. INSAT, ILDN, AWS and GFS inputs are represented in the architecture but are not yet connected to the trained forecast model.

The dashboard labels the result as a **DWR replay** so it is not presented as live weather data.
