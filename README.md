# Mass Spectrometry Imaging Workflow (`maspim`)

`maspim` is a python package that allows users to read, process and interpret mass spectrometry imaging (MSI) data from Bruker instruments.

This package implements methods to process mass spectra, combine them with image information or other measurements (such as micro X-Ray fluorescence, µXRF) and special functionality for laminated sediments to combine spectra in the same layers.

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->

* You have a version of `python3` installed (preferentially the up-to-date version)
* You have `R` installed
* You have the `R` package `rtms` installed

You may also want to check out [msiAlign](https://github.com/weimin-liu/msiAlign), which is not directly a prerequisite, but useful for setting the punch-holes when the automatic detection fails and [msi_feature_extraction](https://github.com/weimin-liu/msi_feature_extraction) if you want to work with exported txt files from DataAnalysis.

## Installing `maspim`

Currently `maspim` is not on PyPI. For now just run

```bash
pip install git+https://github.com/yaza11/maspim.git
```

in your console (assuming you have pip installed).

## Using `maspim`

This is just a quick overview. For more comprehensive tutorials please have a look at the Notebooks.

Generally it is advised to stick to the objects provided at the top level of `maspim`, which are

* `ProjectMSI` and `ProjectXRF` to manage MSI and µXRF measurements respectively and fetched with the `get_project` function
* `ImageSample`, `ImageROI` and `ImageClassified` to set photo properties and finding the sample area
* `Transformation` to register images
* `ReadBrukerMCF` and `hdf5handler` for reading and storing data files.
* `Spectra` to extract intensities from the mass spectra
* `MSI` and `XRF` for handling MSI and µXRF measurements.
* `XRay` for adding information from an X-Ray
* `AgeModel` to set an age model
and a few others. `ProjectMSI` and `ProjectXRF` are the core objects of this package, which manage most of the aforementioned objects. So unless you have a very specific application in mind, it is recommended to do everything with the methods provided by `ProjectMSI` and `ProjectXRF`.

Let's look at a short example of how to define the ${U\_{37}^{k}}^\\prime$ proxy, which you can use to check that your installation worked.
```python
from maspim import get_project
from maspim.res.compound_masses import mC37_2, mC37_3

p = get_project(is_MSI=True, path_folder='path/to/your/measurement.i', is_laminated=False)
# sets or loads ImageHandler, ImageSample, ImageROI and ImageClassified
p.require_images()
# perform all steps to extract intensities from alkenones
p.set_spectra(targets=[mC37_2, mC37_3], suppress_warnings=True)
p.set_data_object()
p.add_pixels_ROI()
# we did not provide any ages, so a time series does not make much sense, but 
# this is just a test, so don't try to interpret the results
p.set_time_series(average_by_col='x', is_continuous=True)
p.set_UK37(method_SST='prahl', n_successes_required=0)
# we can use the index instead of the age for now
p.uk37_proxy.feature_table.loc[:, 'age'] = p.uk37_proxy.feature_table.x
# you should see a plot after this
p.uk37_proxy.plot_comp('SST', errors=False)
```

## Contributing to `maspim`

If you find any bugs or missing features, you are welcome to contribute to `maspim`, just follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to [@weimin-liu](https://github.com/weimin-liu) who has contributed to this project.

## Contact

Feel free to reach out at me via [yzander@marum.de](mailto:yzander@marum.de).
