import os
import rasterio
import numpy as np
from multiprocessing import Pool, cpu_count
from rasterio.warp import reproject, Resampling
from typing import Tuple, List


class SARUtils:
    def __init__(self, landcover_tif_path: str):
        """
        Initialize the SARUtils class with the path to the landcover GeoTIFF file.

        Args:
            landcover_tif_path (str): Path to the landcover GeoTIFF file.
        """
        self.landcover_tif_path = landcover_tif_path

        # Read the landcover GeoTIFF and store the data and metadata
        with rasterio.open(landcover_tif_path) as land_src:
            self.landcover_data = land_src.read(1)
            self.land_transform = land_src.transform
            self.land_crs = land_src.crs

    def apply_landmask(self, sar_image_path: str, output_dir: str) -> None:
        """
        Apply the landmask to a single SAR image and save the result.

        Args:
            sar_image_path (str): Path to the SAR image GeoTIFF file.
            output_dir (str): Directory to save the masked SAR image.
        """
        with rasterio.open(sar_image_path) as src:
            # Read the SAR image data
            sar_image = src.read(1)
            no_data_value = src.nodata
            sar_image = np.where(sar_image == no_data_value, np.nan, sar_image)
            sar_meta = src.meta
            sar_crs = src.crs
            sar_transform = src.transform

            # Reproject landcover data if CRS does not match
            if sar_crs != self.land_crs:
                reprojected_landcover_data = np.empty(shape=(sar_meta['height'], sar_meta['width']),
                                                      dtype=self.landcover_data.dtype)
                reproject(
                    source=self.landcover_data,
                    destination=reprojected_landcover_data,
                    src_transform=self.land_transform,
                    src_crs=self.land_crs,
                    dst_transform=sar_transform,
                    dst_crs=sar_crs,
                    resampling=Resampling.nearest
                )
                landcover_data = reprojected_landcover_data
            else:
                landcover_data = self.landcover_data

        # Define land classes to mask (e.g., water bodies)
        water_classes = [255]
        land_mask = np.isin(landcover_data, water_classes, invert=True)

        # Apply the land mask to the SAR image
        masked_sar_image = np.where(land_mask, np.nan, sar_image)

        # Generate output file path
        output_file_path = os.path.join(output_dir, os.path.basename(sar_image_path).replace('.tif', '_landmask.tif'))

        # Save the masked SAR image to a new file
        sar_meta.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(output_file_path, 'w', **sar_meta) as dst:
            dst.write(masked_sar_image, 1)

    def process_file(self, args: Tuple[str, str]) -> None:
        """
        Wrapper function to process a single SAR image file.

        Args:
            args (Tuple[str, str]): Tuple containing the SAR image path and output directory.
        """
        sar_image_path, output_dir = args
        self.apply_landmask(sar_image_path, output_dir)

    def multiprocess_apply_landmask(self, input_dir: str, output_dir: str) -> None:
        """
        Apply the landmask to all SAR images in the input directory using multiprocessing.

        Args:
            input_dir (str): Directory containing the SAR image GeoTIFF files.
            output_dir (str): Directory to save the masked SAR images.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get list of all GeoTIFF files in the input directory
        tif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

        # Create a pool of worker processes
        with Pool(cpu_count()) as pool:
            pool.map(self.process_file, [(file_path, output_dir) for file_path in tif_files])