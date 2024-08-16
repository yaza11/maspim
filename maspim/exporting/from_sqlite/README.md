## Usage example

### parse peaks from sqlite database

```python
from maspim.exporting.from_sqlite.parser import parse_sqlite

xy, mzs, intensities, snrs, fwhms = parse_sqlite(r'path/to/sqlite')
```

get the calibration constants

```python
from maspim.exporting.from_sqlite.parser import parse_acqumethod

acqumethod = parse_acqumethod(r'path/to/acqumethod')
A = acqumethod['ML1']
B = acqumethod['ML2']
```

calibration with known peaks

```python
from maspim.exporting.from_sqlite.calibration import Calibration

cal = Calibration(target_mz, A, B, tol, min_int, min_snr)
cal.fit(mzs, weights=intensities, snrs=snrs)
mzs = cal.fit(mzs)
```

export target peaks (e.g, alkenones)

```python
from maspim.exporting.from_sqlite.parser import extract_mzs

df = extract_mzs(biomarker_mzs, xy, mzs, intensities, snrs, tol=0.01, min_int=10000, min_snr=0)
```

## Notes

- Somehow SNR values in peaks.sqlite is much lower than those exported from DataAnalysis, not sure why, but not using it for now.
