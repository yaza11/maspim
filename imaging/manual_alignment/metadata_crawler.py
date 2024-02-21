""" crawler for metadata """
import os
import tqdm

# create a namedtuple for the metadata
from collections import namedtuple

from imaging.manual_alignment.func import get_image_file_from_mis, get_px_rect_from_mis, get_msi_rect_from_imaginginfo

Metadata = namedtuple("Metadata", ["spec_file_name", "msi_img_file_path", "msi_img_file_name", "px_rect", "msi_rect"])


class MetadataCrawler:
    """Crawl the metadata directory and return the metadata files"""

    def __init__(self, metadata_dir):
        self.metadata_dir = metadata_dir
        self.metadata = {}
        # make sure the subdirectories are something ending with '.i'
        self.idir = [subdir for subdir in os.listdir(metadata_dir) if subdir.endswith(".i")]
        assert len(self.idir) > 0, "No .i subdirectories found"

    def crawl(self):
        # craw the metadata under .i subdirectories
        # get all the .mis files under the .i subdirectories
        for subdir in self.idir:
            for root, dirs, files in os.walk(os.path.join(self.metadata_dir, subdir)):
                for file in files:
                    if file.endswith(".mis"):
                        mis_file = os.path.join(root, file)
                        # infer the spec file name from the mis file
                        spec_file_name = mis_file.replace(".mis", ".d")
                        if os.path.exists(spec_file_name):
                            try:
                                px_rect = get_px_rect_from_mis(mis_file)
                                xml_file = os.path.join(root, spec_file_name, "ImagingInfo.xml")
                                msi_rect = get_msi_rect_from_imaginginfo(xml_file)
                                im_name = get_image_file_from_mis(mis_file)
                                im_file_path = os.path.join(root, im_name)
                                assert im_name not in self.metadata.keys(), 'duplicate entries found'
                                self.metadata[os.path.basename(spec_file_name)] = Metadata(
                                    os.path.basename(spec_file_name),
                                    im_file_path,
                                    im_name,
                                    px_rect,
                                    msi_rect)
                            except ValueError:
                                continue

    def to_sqlite(self, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS metadata (spec_file_name TEXT, msi_img_file_path TEXT, '
                  'msi_img_file_name TEXT, px_rect TEXT, msi_rect TEXT)')
        for k, v in self.metadata.items():
            c.execute('INSERT INTO metadata VALUES (?, ?, ?, ?, ?)', (v.spec_file_name, v.msi_img_file_path,
                                                                      v.msi_img_file_name, str(v.px_rect),
                                                                      str(v.msi_rect)))
        conn.commit()
        conn.close()

    def collect_msi_img(self, target_dir):
        assert len(self.metadata) > 0, "No metadata found"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for k, v in tqdm.tqdm(self.metadata.items(), desc="Copying msi images"):
            # copy the msi image to the target directory
            # if it's Windows, use copy
            if os.name == "nt":
                import shutil
                os.system(f"shutil.copy({v.msi_img_file_path} {os.path.join(target_dir, v.msi_img_file_name)})")
            else:
                os.system(f"cp {v.msi_img_file_path} {os.path.join(target_dir, v.msi_img_file_name)}")


if __name__ == "__main__":
    mc = MetadataCrawler(r"\\intranet.marum.de\storage\groups\BioGeoChem\Store2\FTICR-MS\FTICR-MS\Susanne\2021"
                         r"\SBB_MV0811-14TC")
    mc.crawl()
    mc.to_sqlite(r"./data/metadata.db")
    mc.collect_msi_img(r"./data/msi_img")
