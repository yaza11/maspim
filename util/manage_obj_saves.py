
Image_disc_attributes = {
    '_is_parent',
    '_data_type',
    '_section',
    '_window',
    'current_image_type'}

ImageProbe_disc_attributes = Image_disc_attributes | {
    'obj_color', 'xywh_ROI'}

ImageROI_disc_attributes = ImageProbe_disc_attributes | {
    'image_classification', 'params_classification'
}

ImageClassified_disc_attributes = ImageProbe_disc_attributes | {
    'params_laminae_simplified',
    'image_seeds'}

ImageTransformation_disc_attributes = {'_section, _window_from', '_window_to'}

Data_disc_attributes = {
    '_section',
    '_section_str',
    '_window',
    '_data_type',
    '_mass_window',
    'peak_th_ref_peak',
    'max_deviation_mz',
    'distance_pixels',
    'current_feature_table'
}

Data_nondata_columns = {
    'L',
    'x_ROI',
    'y_ROI',
    'classification',
    'classification_s',
    'seed'
}

MSI_disc_attributes = Data_disc_attributes
XRF_disc_attributes = Data_disc_attributes | {'default_file_type'}

TimeSeries_disc_attributes = {
    '_section',
    '_window',
    '_data_type',
    'distance_pixels',
    'feature_table_zone_averages',
    'feature_table_zone_standard_deviations',
    'feature_table_zone_successes'}

MetaFeatures_disc_attributes = {
    '_section',
    'windows',
    '_window',
    'cmpds',
    'formulas',
    'df',
    'features_clustering'
}

name_to_attrs = {
    'Image': Image_disc_attributes,
    'ImageProbe': ImageProbe_disc_attributes,
    'ImageROI': ImageROI_disc_attributes,
    'ImageClassified': ImageClassified_disc_attributes,
    'Data': Data_disc_attributes,
    'MSI': MSI_disc_attributes,
    'XRF': XRF_disc_attributes,
    'TimeSeries': TimeSeries_disc_attributes,
    'MetaFeatures': MetaFeatures_disc_attributes
}


def class_to_attributes(class_object: object) -> list[str]:
    c: str = class_object.__class__.__name__
    return name_to_attrs[c]
