# Band-Gap Engineering Explorer

An interactive Python application for exploring approximate band-gap engineering
in semiconductor materials used in quantum technology research.

## Material systems

- Si and Ge
- GaAs and AlGaAs
- InAs, InP, InSb, GaSb and InGaAs
- GaN, AlN and AlGaN
- 4H-SiC and diamond

## Features

- Temperature-dependent band-gap estimates
- Alloy composition controls
- Approximate hydrostatic strain shifts
- An n-type donor-density/Fermi-level model
- Finite and infinite square-well energy calculations
- Parabolic near-band-edge sketches and a schematic well diagram

## Run on Ubuntu or another desktop Python installation

```bash
python3 -m pip install -r requirements.txt
python3 band_gap_engineering.py
```

NumPy, SciPy, and Matplotlib are required. On Linux, the Matplotlib GUI backend
must be installed and available.

## Build a Windows executable

This repository contains a GitHub Actions workflow under
`.github/workflows/build-windows.yml`.

1. Open the repository's **Actions** tab.
2. Select **Build Windows EXE**.
3. Select **Run workflow**.
4. When the workflow finishes successfully, download the
   `BandGapEngineering-Windows` artifact from the run page.
5. Extract the artifact to obtain `BandGapEngineering.exe`.

The workflow builds on a Windows runner. A native Windows build is needed;
PyInstaller does not reliably cross-compile a Windows executable on Linux.

Alternatively, on Windows with 64-bit Python 3.11 installed, run
`build_windows.bat`.

## Scientific limitations

This is a teaching and trend-exploration application, not a calibrated
electronic-structure solver. Material parameters are representative and may
vary by source, alloy composition, crystal phase, strain, and temperature.
Band plots are local parabolic sketches rather than full Brillouin-zone band
structures. The quantum-well solver uses a user-specified well depth and a
simplified effective-mass model; it does not calculate actual heterostructure
band offsets. Doping, donor ionisation, strain, and band-gap narrowing are
approximate. Diamond and 4H-SiC are treated as host semiconductors; their
individual quantum-defect states are not modelled.

Check the code comments and cited references before using any numerical result
for quantitative research.
