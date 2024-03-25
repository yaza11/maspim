import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.util.find_XRF_ROI import find_ROI_in_image

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image


def test_rois():
    s490, s495, s500, s505 = [1, 0, 0, 0]
    
    if s490:
        file_image = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343c_490-495cm/PS343c 490-495cm Mosaic.bmp'
        file_image_roi = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343c_490-495cm/PS343c cts _Video 1.tif'
        
        find_ROI_in_image(file_image, file_image_roi, plts=True)
    
    if s495:
        file_image = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343d_495-500cm/Caricao_495-500cm_100µm_Mosaic.tif'
        file_image_roi = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343d_495-500cm/Caricao_495-500cm_100µm_ElementImages_Video 1.tif'
        
        find_ROI_in_image(file_image, file_image_roi, plts=True)
    
    if s500:
        file_image = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343e_500-505cm/S0343e 500-505cm Mosaic ROI.bmp'
        file_image_roi = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343e_500-505cm/D0343e_noLegend_Video 1.tif'
        
        find_ROI_in_image(file_image, file_image_roi, plts=True)
    
    if s505:
        file_image = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343f_505-510cm/S0434f 505-510 Mosaic.bmp'
        file_image_roi = 'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343f_505-510cm/D0343f_Video 1.tif'
        
        find_ROI_in_image(file_image, file_image_roi, plts=True)


# from data.cXRF import XRF

# xrf = XRF(
#     r'D:\Cariaco line scan Xray\uXRF slices\S0343 Cariaco_480-510cm_100µm slices\S0343c_490-495cm',
#     file_image_sample = r'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/S0343c_490-495cm/Caricao_490-495cm_100µm_Mosaic.tif',
#     measurement_name = 'S0343c'
# )

# xrf.set_feature_table_from_txts()

# xrf.plt_comp('graylevel')

# xrf.analyzing_NMF(k=3)
# xrf.plt_NMF(k=3)

# from rsciio.bruker import file_reader

# import matplotlib.pyplot as plt

# Load the Bruker microXRF file
# s = file_reader(
#     'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm slices/' +
#     'S0343c_490-495cm/Cariaco_490-495cm_100µm.bcf',
#     # cutoff_at_kV=16
# )

# s_video = s[0]
# s_data = s[1]

# data = s_data['data']

# https://hyperspy.org/hyperspy-doc/v1.1/user_guide/eds.html

## %% summed spectrum
# ax = s_data['axes'][2]
# energy = np.arange(0, ax['size']) * ax['scale'] + ax['offset']
# plt.plot(energy, data.sum(axis=0).sum(axis=0))
# plt.ylabel('Counts')
# plt.xlabel('Energy in keV')

# s_data = hs.signals.Signal2D(
#     data=s[1].data
# )
# for attr in ('_metadata', '_original_metadata', 'axes_manager'):
#     s_data.__dict__[attr] = s[1].__dict__[attr]

# s_data.change_dtype('float64')

# lines = s_data.metadata.Sample.xray_lines

# element_images = s_data.get_lines_intensity(elements=lines)



import xml.etree.ElementTree as ET
import re
import numpy as np
from struct import unpack as strct_unp
from rsciio.bruker._api import BCF_reader

file = 'D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/Cariaco_490-495cm_100µm.bcf'


r = BCF_reader(file)
    
header = r.header

def extract_all_xml_tags(binary_data):
    # Construct a regular expression pattern to match any XML tag
    xml_pattern = re.compile(rb'<[^\x00-\x2F\x3A-\x40\x5B-\x60\x7B-\x7F]+>')

    xml_tags = xml_pattern.findall(binary_data)
            
    return xml_tags

with open(file, "rb") as binary_file:
    binary_data = binary_file.read()
    
    binary_file.seek(0x124)
    version, chunksize = strct_unp("<fI", binary_file.read(8))
    sfs_version = "{0:4.2f}".format(version)
    usable_chunk = chunksize - 32
    binary_file.seek(0x140)
    # the sfs tree and number of the items / files + directories in it,
    # and the number in chunks of whole sfs:
    tree_address, n_tree_items, sfs_n_of_chunks = strct_unp(
        "<III", binary_file.read(12)
    )
    n_file_tree_chunks = np.ceil(
        (n_tree_items * 0x200) / (chunksize - 0x20)
    )
    raw_tree = binary_data[
        chunksize * tree_address + 0x138: 
        chunksize * tree_address + 0x138 + 0x200 * n_tree_items
    ]


def extract_xml_chunks(binary_data, element_name, encoding='WINDOWS-1252'):
    # Construct a regular expression pattern for the specified XML element
    pattern_str = fr'<{element_name}.*?</{element_name}>'.encode(encoding)
    xml_pattern = re.compile(pattern_str, re.DOTALL)
    
    xml_chunks = xml_pattern.findall(binary_data)

    # for chunk in xml_chunks:
    #     print(chunk[:1000])

    # return xml_chunks
    roots = []
    for chunk in xml_chunks:
        # Parse each XML chunk
        decoded_chunk = chunk.decode(encoding, errors='replace')

        # Replace non-ASCII characters with an empty string
        clean_chunk = re.sub(r'[^\x20-\x7D]', '_', decoded_chunk)
        try:
            root = ET.fromstring(clean_chunk)
            roots.append(root)
        except (UnicodeDecodeError, ET.ParseError) as e:
            print(f"Error processing XML chunk: {e}")
            return clean_chunk
    return roots
        
# tags = extract_all_xml_tags(binary_data)
# tags_ = []
# for tag in tags:
#     try:
#         tag = tag.decode('utf-8')[1:-1]
#         if tag not in tags_:
#             tags_.append(tag)
#     except UnicodeDecodeError:
#         continue
                
# roots = extract_xml_chunks(binary_data, tags_[0])

# for root in roots:
#     print(root.tag, root.attrib, root.text)
#     for child in root:
#         print()
#         print(child.tag, child.attrib, child.text)
    
    
