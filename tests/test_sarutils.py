import unittest
import os
import json
import numpy as np
import rasterio
from src.sarutils import SARUtils


class TestSARUtils(unittest.TestCase):

	def setUp(self):
		"""
		Set up test environment.
		"""
		self.landcover_tif_path = 'path_to_test_landcover.tif'
		self.sar_image_path = 'path_to_test_sar_image.tif'
		self.output_dir = 'test_output'
		self.index_output_file = 'test_tile_index.json'

		# Create a SARUtils instance
		self.sar_utils = SARUtils(self.landcover_tif_path)

		# Ensure output directory exists
		os.makedirs(self.output_dir, exist_ok=True)

	def tearDown(self):
		"""
		Clean up test environment.
		"""
		if os.path.exists(self.index_output_file):
			os.remove(self.index_output_file)
		if os.path.exists(self.output_dir):
			for file in os.listdir(self.output_dir):
				os.remove(os.path.join(self.output_dir, file))
			os.rmdir(self.output_dir)

	def test_initialization(self):
		"""
		Test initialization of SARUtils.
		"""
		self.assertIsNotNone(self.sar_utils.landcover_data, "Landcover data should not be None")
		self.assertIsNotNone(self.sar_utils.land_transform, "Land transform should not be None")
		self.assertIsNotNone(self.sar_utils.land_crs, "Land CRS should not be None")

	def test_apply_landmask(self):
		"""
		Test applying the landmask to a SAR image.
		"""
		self.sar_utils.apply_landmask(self.sar_image_path, self.output_dir)

		output_file_path = os.path.join(self.output_dir,
		                                os.path.basename(self.sar_image_path).replace(".tif", "_landmask.tif"))
		self.assertTrue(os.path.exists(output_file_path), "Output file should exist after applying landmask")

		with rasterio.open(output_file_path) as src:
			masked_sar_image = src.read(1)
			self.assertTrue(np.isnan(masked_sar_image).any(),
			                "Masked SAR image should contain NaNs where land is present")

	def test_index_tiles(self):
		"""
		Test indexing tiles from GeoTIFF files.
		"""
		SARUtils.index_tiles(
			tiff_dir=self.output_dir,
			output_file=self.index_output_file,
			tile_size=64,
			land_threshold=0.25,
			random_sampling=True,
			num_random_samples=10
		)

		self.assertTrue(os.path.exists(self.index_output_file), "Index output file should exist after indexing tiles")

		with open(self.index_output_file, 'r') as f:
			tile_index = json.load(f)
			self.assertGreater(len(tile_index), 0, "Tile index should contain entries")


if __name__ == '__main__':
	unittest.main()
