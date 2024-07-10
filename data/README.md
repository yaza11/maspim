## Overview

This module implements objects for handling age models and measurements (MSI and XRF).
Below some usage examples are shown

### Age Model
Offer depth-age conversions for tabled data.
This object allows to read in tabulated age data, interpolate between missing depth and combine multiple objects.

#### Example Usage
Usually the depth-age information is stored in a separate file. The file reader is a wrapper around the pandas read
functions, hence, to read data in correctly, the user is referenced to the documentation of read_csv, read_excel,
depending on the file type. For a tab separated file with column names 'age' and 'depth' the initalization can look like this:
```python
from msi_workflow.data.cAgeModel import AgeModel
age_model = AgeModel(path_file='path/to/file.txt', sep='\t', index_col=False)
```
The class expects the depth data to be in cm below seaflow and the age in years. Oftentimes this is not the case.
Here, the add_depth_offset and convert_depth_scale methods are handy (call order does not matter,
but a different depth offset is required after converting the depth-scale)
```python
age_model.convert_depth_scale(1 / 10)  # converts mm to cm
age_model.add_depth_offset(500)  # age model starts add 500 cmbsf
```
Those parameters can also be provided upon initialization
```python
age_model = AgeModel(
    path_file='path/to/file.txt',
    depth_offset=5000,
    conversion_to_cm=1 / 10,
    sep='\t',
    index_col=False
)
```
In this case the depth offset is applied first.
By default, the age model will be saved in the same folder from which the data has been loaded
```python
age_model.save()  # saves the object in 'path/to'
```
This can be changed by providing a path
```python
age_model.save('path/to/desired/folder')
```
Age models can be combined, if the depths do not overlap
```python
age_model1 = AgeModel(path1, ...)
age_model2 = AgeModel(path2, ...)
age_model_combined = age_model1 + age_model2
```

### MSI Data
Class to wrap and process mass spectrometry imaging data.

This object needs a table with compounds as columns
(plus the x and y coordiantes of the data pixels). Each row corresponds to a data pixel. The
recommended way is to use the set_feature_table_from_spectra method, but it is always possible
to inject the data by
```python
msi.feature_table = ft
```
where msi is the MSI instance and ft is a pandas dataframe with data, x and y columns.

#### Example Usage
Import
```python
from data.cMSI import MSI
```
Initialize
```python
msi = MSI(path_d_folder='path/to/your/d_folder.d')
```
If there are multiple mis files in the parent folder, it is recommended to provide
the path_mis_file parameter as well
```python
msi = MSI(path_d_folder='path/to/your/d_folder.d', path_mis_file='path/to/your/mis/file.mis')
```
In that case the object will infere the data pixel resolution from the mis file.

Set the feature table (here assuming that a spectra object has been saved to disk before):
```python
from exporting.from_mcf.cSpectrum import Spectra
spec = Spectra(path_d_folder='path/to/your/d_folder.d', load=True)
msi.set_feature_table_from_spectra(spec)
```

Now we are ready to do some analysis, e.g. nonnegative matrix factorization
```python
msi.plt_NMF(k=5)
```

### XRF Data
Class to wrap and process mass spectrometry imaging data.

This object compiles a feature table from txt files in a folder (txt files were exported,
example: folder name 'S0343a_480-485cm', file name 'S0343a_Al.txt'). It is recommended to leave
the folder name consistent with the file names. Further, the original image should be exported and
contian the keyword 'Mosaic'. This will be necessary for the image classes.

The feature table also contains information about the x and y coordiantes of the data pixels.
Each row corresponds to a data pixel.

#### Example Usage
Import
```python
from data.cXRF import XRF
```
Initialize
```python
xrf = XRF(path_folder='path/to/your/folder')
```
By default the measurement name will be infered from the folder name and the distance_pixels
read from the bcf file. If multiple exports are located in the folder, the measurement name
should be changed to the desired export:
```python
xrf = XRF(path_folder='path/to/your/folder', measurement_name='D0343a')
```
Set the feature table.
```python
xrf.set_feature_table_from_txts()
```
Now we are ready to do some analysis, e.g. nonnegative matrix factorization
```python
xrf.plt_NMF(k=5)
```