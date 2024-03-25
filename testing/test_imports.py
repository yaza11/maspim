"""Test initialization of all submodules."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import res.constants
import res.directory_paths

import util.manage_class_imports
import util.manage_obj_saves
import util.cClass

import misc.cAgeModel

import imaging.util.coordinate_transformations
import imaging.util.Image_boxes
import imaging.util.Image_convert_types
import imaging.util.Image_geometry
import imaging.util.Image_helpers
import imaging.util.Image_processing
import imaging.util.Image_plotting

import imaging.misc.fit_distorted_rectangle

import imaging.main.cImage
import imaging.main.cTransformation

import data.annotating
import data.cDataClass
import data.cMSI
import data.cXRay
import data.cXRF

import timeSeries.cTimeSeries
import timeSeries.cProxy

import clustering.cMetaFeatures
import clustering.network
import clustering.network_from_nmf