from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'Processing of mass spectrometry imaging and micro X-Ray fluorescence data.'

# Setting up
setup(
    name="msi_workflow",
    version=VERSION,
    author="Yannick Zander",
    author_email="yzander@marum.de",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/weimin-liu/msi_workflow",
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'pandas', 'rpy2', 'scipy', 'tqdm',
                      'cv2', 'sklearn', 'skimage', 'psutil'],
    extras_require={'dev': 'twine'},
    keywords=['python', 'mass spectrometry imaging', 'bruker', 'mcf', 'MALDI',
              'laminated', 'lamination', 'image registration'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
