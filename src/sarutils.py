import rasterio
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.warp import reproject, Resampling

import json
import logging
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple
from typing import Union
from rtree import index


class SARUtils:
    def __init__(self, landcover_tif_path: str):
        """
        Initialize the SARUtils class with the path to the landcover GeoTIFF file.

        Args:
            landcover_tif_path (str): Path to the landcover GeoTIFF file.
        """
        try:
            self.landcover_tif_path = landcover_tif_path

            # Read the landcover GeoTIFF and store the data and metadata
            with rasterio.open(landcover_tif_path) as land_src:
                self.landcover_data = land_src.read(1)
                self.land_transform = land_src.transform
                self.land_crs = land_src.crs
        except Exception as e:
            print(f"Error loading landcover data: {e}")

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
                reprojected_landcover_data = np.empty(
                    shape=(sar_meta["height"], sar_meta["width"]),
                    dtype=self.landcover_data.dtype,
                )
                reproject(
                    source=self.landcover_data,
                    destination=reprojected_landcover_data,
                    src_transform=self.land_transform,
                    src_crs=self.land_crs,
                    dst_transform=sar_transform,
                    dst_crs=sar_crs,
                    resampling=Resampling.nearest,
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
        output_file_path = os.path.join(
            output_dir,
            os.path.basename(sar_image_path).replace(".tif", "_landmask.tif"),
        )

        # Save the masked SAR image to a new file
        sar_meta.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(output_file_path, "w", **sar_meta) as dst:
            dst.write(masked_sar_image, 1)

        print(f"Finished processing {sar_image_path}")

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
        tif_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".tif")
        ]

        # Create a pool of worker processes
        with Pool(cpu_count()) as pool:
            pool.map(
                self.process_file, [(file_path, output_dir) for file_path in tif_files]
            )

    @staticmethod
    def index_tiles(
        tiff_dir: str,
        output_file: str,
        tile_size: int = 512,
        land_threshold: float = 0.25,
        stride: Union[int, float] = 1.0,
        random_sampling: bool = False,
        num_random_samples: int = 1000,
    ) -> None:
        """
        Index potential valid tiles from GeoTIFF files.

        Args:
                tiff_dir (str): Directory containing the landmasked GeoTIFFs.
                output_file (str): File to save the tile index.
                tile_size (int): Size of the square tiles to extract.
                land_threshold (float): Maximum allowable proportion of land in a tile.
                stride (Union[int, float]): Stride for sliding window. Can be an integer or a float representing a fraction of the tile size.
                random_sampling (bool): Whether to use random sampling instead of sliding window.
                num_random_samples (int): Number of random samples to generate if random_sampling is True.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        start_time = time.time()

        tile_index = []
        tiff_files = [
            os.path.join(tiff_dir, f)
            for f in os.listdir(tiff_dir)
            if f.endswith(".tif")
        ]

        if random_sampling:
            # Calculate the total possible number of non-overlapping tiles
            total_image_area = sum(
                [rasterio.open(f).width * rasterio.open(f).height for f in tiff_files]
            )
            tile_area = tile_size**2
            max_possible_tiles = total_image_area // tile_area
            logging.info(f"Theoretical maxiumum number of tiles: {max_possible_tiles}")
            if num_random_samples > max_possible_tiles:
                logging.warning(
                    "Requested number of random samples exceeds the feasible number of tiles."
                )

        # Create an R-tree index for efficient spatial querying
        idx = index.Index()

        for file_path in tqdm(tiff_files, desc="Indexing tiles"):
            with rasterio.open(file_path) as src:
                height, width = src.height, src.width

                # Calculate stride
                if isinstance(stride, float):
                    stride = int(tile_size * stride)

                if random_sampling:
                    for _ in range(num_random_samples):
                        x = random.randint(0, width - tile_size)
                        y = random.randint(0, height - tile_size)
                        # Check for duplicates using the R-tree index
                        if list(idx.intersection((x, y, x + tile_size, y + tile_size))):
                            continue
                        window = rasterio.windows.Window(x, y, tile_size, tile_size)
                        tile = src.read(1, window=window)
                        land_pixels = np.isnan(tile)
                        land_proportion = np.mean(land_pixels)
                        if land_proportion <= land_threshold:
                            tile_index.append({"file": file_path, "x": x, "y": y})
                            idx.insert(
                                len(tile_index) - 1,
                                (x, y, x + tile_size, y + tile_size),
                            )
                else:
                    for y in range(0, height - tile_size + 1, stride):
                        for x in range(0, width - tile_size + 1, stride):
                            window = rasterio.windows.Window(x, y, tile_size, tile_size)
                            tile = src.read(1, window=window)
                            land_pixels = np.isnan(tile)
                            land_proportion = np.mean(land_pixels)
                            if land_proportion <= land_threshold:
                                tile_index.append({"file": file_path, "x": x, "y": y})

        with open(output_file, "w") as f:
            json.dump(tile_index, f)

        end_time = time.time()
        logging.info(f"Indexing completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Total tiles indexed: {len(tile_index)}")

    @staticmethod
    def preview_tiles(index_file: str, tile_size: int, num_tiles: int = 9) -> None:
        """
        Preview tiles from the indexed tiles.

        Args:
            index_file (str): Path to the JSON file containing the tile index.
            tile_size (int): Size of the square tiles to extract.
            num_tiles (int): Number of tiles to preview.
        """
        with open(index_file, "r") as f:
            tile_index = json.load(f)

        # Randomly select one frame
        selected_frame = random.choice(tile_index)["file"]
        selected_tiles = [tile for tile in tile_index if tile["file"] == selected_frame]
        selected_tiles = random.sample(
            selected_tiles, min(num_tiles, len(selected_tiles))
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Visualize the entire SAR frame with boxes representing the tiles
        with rasterio.open(selected_frame) as src:
            image = src.read(1)
            ax.imshow(image, cmap="gray")
            for tile_info in selected_tiles:
                x = tile_info["x"]
                y = tile_info["y"]
                rect = Rectangle(
                    (x, y),
                    tile_size,
                    tile_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_title("SAR Frame with Selected Tiles")

        plt.show()

        # Display nxn grid of selected tiles
        n = int(np.ceil(np.sqrt(num_tiles)))
        fig, axes = plt.subplots(n, n, figsize=(15, 15))

        for i, tile_info in enumerate(selected_tiles):
            file_path = tile_info["file"]
            x = tile_info["x"]
            y = tile_info["y"]

            with rasterio.open(file_path) as src:
                window = rasterio.windows.Window(x, y, tile_size, tile_size)
                tile = src.read(1, window=window)
                tile = np.nan_to_num(tile, nan=0.0)

            ax = axes[i // n, i % n]
            ax.imshow(tile, cmap="gray")
            ax.set_title(f"Tile {i + 1}")

        # Hide unused subplots
        for j in range(i + 1, n * n):
            axes[j // n, j % n].axis("off")

        plt.tight_layout()
        plt.show()

        plt.tight_layout()
        plt.show()
