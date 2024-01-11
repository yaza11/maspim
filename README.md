## Usage example
parse peaks from sqlite database
```python
from exporting.parser import parse_sqlite

xy, mzs, intensities,snrs = parse_sqlite(r'path/to/sqlite')
```
get the calibration constants
```python
from exporting.parser import parse_acqumethod

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
from exporting.parser import extract_mzs

df = extract_mzs(biomarker_mzs, xy, mzs, intensities, snrs, tol=0.01, min_int=10000, min_snr=0)
```

## Usage example for exporting from mcf file
create a reader object
```python
from exporting_mcf.rtms_communicator import ReadBrukerMCF
d_folder = 'path/to/d_folder.d'
con = ReadBrukerMCF(d_folder)
con.create_reader()
```

bin profiles based on kernels
```python
spectra = Spectra(reader=con)
spectra.add_all_spectra(con)
spectra.set_peaks()
spectra.set_kernels()
spectra.plt_kernels()
spectra.bin_spectra(con)
spectra.binned_spectra_to_df(con)
spectra.save()  # save can be used at any step, to load call "spectra = Spectra(reader=con, load=True)" instead
```
