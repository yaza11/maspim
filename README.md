## Usage example
parse peaks from sqlite database
```python
from exporting.parse import parse_sqlite

xy, mzs, intensities,snrs = parse_sqlite(r'path/to/sqlite')
```
get the calibration constants
```python
from exporting.parse import parse_acqumethod

acqumethod = parse_acqumethod(r'path/to/acqumethod')
A = acqumethod['ML1']
B = acqumethod['ML2']
```

calibration with known peaks
```python
from exporting.calibration import Calibration

cal = Calibration(target_mz,A,B, tol, min_int, min_snr)
cal.fit(mzs, weights=intensities, snrs=snrs)
mzs = cal.transform(mzs)
```

export target peaks (e.g, alkenones)
```python
from exporting.parse import export_mzs

df = extract_mzs(target_mzs, xy, mzs, intensities, snrs, tol=0.01, min_int=10000, min_snr=0)
```