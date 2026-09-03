# Magnetic Storm Detection Program

This repository contains a Streamlit application for detecting past geomagnetic storm events, filtering their main phases, and computing solar wind derived parameters over the main phases.

The data used are from NASA OMNIWeb.

The algorithm description, parameter definitions, and analysis workflow are provided in `parameters.pdf`.

## App workflow

The app follows a three-stage workflow:

1. **Storm detection**  
   Detects disturbed Dst episodes and storm minima.

2. **Main phase filtering**  
   Uses the detected storms to identify `t_start` and retain valid main phase intervals.

3. **Data-complete storms + metrics**  
   Checks main phase solar wind data completeness and computes the final storm metrics.

Additional outputs include correlation tables, plots and downloadable CSV/ZIP files.

## Required files

The following files should be placed in the same folder as `app.py`:

```text
app.py
requirements.txt
parameters.pdf
1964-may 2026.txt
background image file
```

The bundled OMNI data file `1964-may 2026.txt` contains data up to May 2026. Over time, this file will become outdated.

To update the dataset, download a new hourly averaged OMNI file from NASA OMNIWeb Data Explorer:

https://omniweb.gsfc.nasa.gov/form/dx1.html

Select **Hourly averaged** resolution and request the same variables used by the app:

```text
IMF Magnitude Avg, nT
Bz, GSM, nT
Flow Speed, km/sec
Ey - Electric Field, mV/m
Dst Index, nT
```

## Input OMNI format

The expected OMNI input format is 8 columns:

```text
Year DOY Hour IMF Bz Vsw Ey Dst
```

## Run locally

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

## Environment

Tested with Python 3.13.12.

Python package versions are specified in `requirements.txt`.

## Notes

The background image is created by the author using AI.
