from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.3.1'
DESCRIPTION = 'Processing of mass spectrometry imaging and micro X-Ray fluorescence data.'
NAME = "maspim"

# Setting up
setup(
    name=NAME,
    version=VERSION,
    author="Yannick Zander",
    author_email="yzander@marum.de",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/yaza11/maspim",
    packages=find_packages(),
    package_data={NAME: ['res/calibrants.txt', 'res/elements.txt']},
    install_requires=['matplotlib',
                      'numpy',
                      'pandas',
                      'scipy',
                      'tqdm',
                      'opencv-python',
                      'scikit-learn',
                      'scikit-image',
                      'psutil',
                      'h5py',
                      'pillow',
                      'textdistance',
                      'astropy',
                      'mfe'],
    extras_require={'dev': 'twine', 'all': ['rpy2==3.5.12']},
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
