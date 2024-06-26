import rasterio as rio
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.warp import reproject, Resampling

import csv
import json
import logging
import os
import re
import shutil
import time
import zipfile
from multiprocessing import Pool, Lock, Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple
from typing import Union
from rtree import index


def extract_log_info(log_file_path):
    key_info = {
        "SAFE directory": "",
        "Output resolution": "",
        "Radiometry": "",
        "Scale": "",
        "Speckle filter": "",
        "DEM matching": "",
        "Include DEM": "",
        "Include inc. angle map": "",
        "Include scattering area": "",
        "Include RGB decomposition": "",
        "Orbit file": "",
        "Output name": "",
        "DEM name": "",
    }
    with open(log_file_path, "r") as f:
        for line in f:
            if "SAFE directory" in line:
                key_info["SAFE directory"] = line.split(": ", 1)[1].strip()
            elif "Output resolution" in line:
                key_info["Output resolution"] = line.split(": ", 1)[1].strip()
            elif "Radiometry" in line:
                key_info["Radiometry"] = line.split(": ", 1)[1].strip()
            elif "Scale" in line:
                key_info["Scale"] = line.split(": ", 1)[1].strip()
            elif "Speckle filter" in line:
                key_info["Speckle filter"] = line.split(": ", 1)[1].strip()
            elif "DEM matching" in line:
                key_info["DEM matching"] = line.split(": ", 1)[1].strip()
            elif "Include DEM" in line:
                key_info["Include DEM"] = line.split(": ", 1)[1].strip()
            elif "Include inc. angle map" in line:
                key_info["Include inc. angle map"] = line.split(": ", 1)[1].strip()
            elif "Include scattering area" in line:
                key_info["Include scattering area"] = line.split(": ", 1)[1].strip()
            elif "Include RGB decomposition" in line:
                key_info["Include RGB decomposition"] = line.split(": ", 1)[1].strip()
            elif "Orbit file" in line:
                key_info["Orbit file"] = line.split(": ", 1)[1].strip()
            elif "Output name" in line:
                key_info["Output name"] = line.split(": ", 1)[1].strip()
            elif "DEM name" in line:
                key_info["DEM name"] = line.split(": ", 1)[1].strip()
    return key_info


class SARUtils:
    # Initialize a lock and manager for multiprocessing
    lock = Lock()
    manager = Manager()
    results = manager.list()

    def __init__(self, logger: logging = None, landcover_tif_path: str = None):
        """
        Initialize the SARUtils class with the path to the landcover GeoTIFF file.

        Args:
            landcover_tif_path (str): Path to the landcover GeoTIFF file.
        """
        self.landcover_data = None
        self.logger = logger

        if self.logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

        if landcover_tif_path:
            try:
                self.landcover_tif_path = landcover_tif_path

                # Read the landcover GeoTIFF and store the data and metadata
                with rio.open(landcover_tif_path) as land_src:
                    self.landcover_data = land_src.read(1)
                    self.land_transform = land_src.transform
                    self.land_crs = land_src.crs
            except Exception as e:
                self.logger.error(f"Error loading landcover data: {e}")
        else:
            self.logger.info("Landmask not provided.")

    def apply_landmask(self, sar_image_path: str, output_dir: str) -> None:
        """
        Apply the landmask to a single SAR image and save the result.

        Args:
            sar_image_path (str): Path to the SAR image GeoTIFF file.
            output_dir (str): Directory to save the masked SAR image.
        """
        with rio.open(sar_image_path) as src:
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
        sar_meta.update(dtype=rio.float32, nodata=np.nan)
        with rio.open(output_file_path, "w", **sar_meta) as dst:
            dst.write(masked_sar_image, 1)

        self.logger.info(f"Finished processing {sar_image_path}")

    def multiprocess_apply_landmask(
        self, input_dir: str, output_dir: str, num_workers: int = None
    ) -> None:
        """
        Apply the landmask to all SAR images in the input directory using multiprocessing.

        Args:
            input_dir (str): Directory containing the SAR image GeoTIFF files.
            output_dir (str): Directory to save the masked SAR images.
            num_workers (int, optional): Number of worker processes. Defaults to the number of CPU cores.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Get a list of all GeoTIFF files in the input directory
        tif_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".tif")
        ]
        if num_workers is None:
            num_workers = cpu_count()

        # Create a pool of worker processes
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._landmask_file, (file_path, output_dir)): file_path
                for file_path in tif_files
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing {futures[future]}: {e}")

    def generate_tile_map(
        self,
        tiff_dir: str,
        output_file: str = None,
        tile_size: int = 512,
        land_threshold: float = 0.25,
        stride: Union[int, float] = 1.0,
        random_sampling: bool = False,
        num_random_samples: int = None,
        num_workers: int = None,
    ) -> None:
        """
        Create an index of tiles from the landmasked GeoTIFFs in the input directory.

        Args:
            tiff_dir (str): Directory containing the landmasked GeoTIFFs.
            output_file (str): File to save the tile index.
            tile_size (int): Size of the square tiles to extract.
            land_threshold (float): Maximum allowable proportion of land in a tile.
            stride (Union[int, float]): Stride for sliding window. Can be an integer or a float representing
                                        a fraction of the tile size.
            random_sampling (bool): Whether to use random sampling instead of sliding window.
            num_random_samples (int): Number of random samples to generate if random_sampling is True.
            num_workers (int): Number of worker processes to use. Defaults to the number of CPUs available.

        Returns:
            None
        """
        start_time = time.time()

        tile_index = []
        tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith(".tif")]

        if random_sampling:
            total_image_area = sum(
                [rio.open(os.path.join(tiff_dir, f)).width * rio.open(os.path.join(tiff_dir, f)).height for f in tiff_files]
            )
            tile_area = tile_size**2
            max_possible_tiles = total_image_area // tile_area
            self.logger.info(
                f"Theoretical maximum number of tiles: {max_possible_tiles}"
            )
            if num_random_samples > max_possible_tiles:
                self.logger.warning(
                    "Requested number of random samples exceeds the feasible number of tiles."
                )

        if num_workers is None:
            num_workers = os.cpu_count()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            args = [
                (file, tiff_dir, tile_size, land_threshold, stride, random_sampling, num_random_samples)
                for file in tiff_files
            ]
            futures = {executor.submit(SARUtils._tilemap_file, arg): arg for arg in args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing tiles"):
                try:
                    result = future.result()
                    tile_index.extend(result)
                except Exception as e:
                    self.logger.error(f"Error processing {futures[future]}: {e}")

        tile_index_data = {
            "base_dir": tiff_dir,
            "tile_size": tile_size,
            "land_threshold": land_threshold,
            "stride": stride,
            "random_sampling": random_sampling,
            "num_random_samples": num_random_samples,
            "tiles": tile_index,
        }

        if output_file is None:
            output_file = f"SAR_tile_map_ts{tile_size}_stride{stride}_land{land_threshold}_tiles{len(tile_index)}.json"

        with open(output_file, "w") as f:
            json.dump(tile_index_data, f)

        end_time = time.time()
        self.logger.info(f"Indexing completed in {end_time - start_time:.2f} seconds.")
        self.logger.info(f"Total tiles indexed: {len(tile_index)}")

    @staticmethod
    def preview_tiles(index_file: str, sar_frame: str = None, num_tiles: int = 9) -> None:
        """
        Preview tiles from the indexed tiles.

        Args:
            index_file (str): Path to the JSON file containing the tile index.
            sar_frame (str): Optional path to the specific SAR frame to preview.
            num_tiles (int): Number of tiles to preview.
        """
        with open(index_file, "r") as f:
            tile_index = json.load(f)

        base_dir = tile_index['base_dir']
        tile_size = tile_index['tile_size']
        tile_index = tile_index['tiles']

        if sar_frame is None:
            selected_frame = random.choice(tile_index)["file"]
        else:
            selected_frame = sar_frame

        total_tiles = [tile for tile in tile_index if tile["file"] == selected_frame]
        selected_tiles = random.sample(
            total_tiles, min(num_tiles, len(total_tiles))
        )

        # Visualize the entire SAR frame with boxes representing the tiles
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        with rio.open(os.path.join(base_dir, selected_frame)) as src:
            image = src.read(1)
            ax.imshow(image, cmap="gray", vmin=np.nanmin(image), vmax=np.nanmax(image))
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
            ax.set_title(f"SAR Frame: {base_dir}/{selected_frame}")
        plt.show()
        print(f"Frame {selected_frame} has {len(total_tiles)} tiles.")
        print(f"Selected frame: {selected_frame}")

        # Display nxn grid of selected tiles
        n = int(np.ceil(np.sqrt(num_tiles)))
        fig, axes = plt.subplots(n, n, figsize=(15, 15))

        for i, tile_info in enumerate(selected_tiles):
            file = tile_info["file"]
            x = tile_info["x"]
            y = tile_info["y"]

            with rio.open(os.path.join(base_dir, file)) as src:
                window = rio.windows.Window(x, y, tile_size, tile_size)
                tile = src.read(1, window=window)

            ax = axes[i // n, i % n]
            ax.imshow(tile, cmap="gray", vmin=np.nanmin(tile), vmax=np.nanmax(tile))
            ax.set_title(f"Tile {i + 1}")

        # Hide unused subplots
        for j in range(i + 1, n * n):
            axes[j // n, j % n].axis("off")

        plt.tight_layout()
        plt.show()

        plt.tight_layout()
        plt.show()

    def process_zip_files_in_directory(self, zip_dir, output_dir, num_workers=None):
        """
        Process zipped ASF hyp3 products in a directory moving and organizing GeoTIFFs and producing a CSV file
        with key information described in the log files.

        Args:
            zip_dir (str): Directory containing zipped ASF hyp3 products.
            output_dir (str): Directory to save the processed GeoTIFFs and CSV file.
            num_processes (int): Number of processes to use for multiprocessing.

        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_file = os.path.join(output_dir, "output.csv")

        if num_workers is None:
            num_workers = cpu_count()

        fieldnames = [
            "Granule Name",
            "SAFE directory",
            "Output resolution",
            "Radiometry",
            "Scale",
            "Speckle filter",
            "DEM matching",
            "Include DEM",
            "Include inc. angle map",
            "Include scattering area",
            "Include RGB decomposition",
            "Orbit file",
            "Output name",
            "DEM name",
        ]

        # Get a list of all zip files in the directory
        zip_files = [
            os.path.join(zip_dir, f) for f in os.listdir(zip_dir) if f.endswith(".zip")
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._process_zip_file, zip_file, output_dir): zip_file
                for zip_file in zip_files
            }
            for future in futures:
                future.add_done_callback(self._collect_results)
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing zip files"
            ):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing {futures[future]}: {e}")

        # Write results to CSV file
        with open(csv_file, "w", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for result in self.results:
                csv_writer.writerow(result)

    def _collect_results(self, future):
        """
        Collect results from the future object and append to the shared results list.

        Args:
            future (Future): Future object from the ProcessPoolExecutor.
        """
        try:
            result = future.result()
            with SARUtils.lock:
                SARUtils.results.append(result)
            self.logger.info(f"Processed {result['Granule Name']}")
        except Exception as e:
            self.logger.error(f"Error in future result: {e}")

    def _landmask_file(self, args: Tuple[str, str]) -> None:
        """
        Wrapper function to landmask a single SAR image file.

        Args:
            args (Tuple[str, str]): Tuple containing the SAR image path and output directory.
        """
        sar_image_path, output_dir = args
        self.apply_landmask(sar_image_path, output_dir)

    @staticmethod
    def _tilemap_file(args):
        """
        Process a single GeoTIFF file to generate tile indices.

        Args:
            args (tuple): A tuple containing the parameters needed for processing:
                          (file, tiff_dir, tile_size, land_threshold, stride, random_sampling, num_random_samples).

        Returns:
            list: A list of tile indices generated from the GeoTIFF file.
        """
        file, tiff_dir, tile_size, land_threshold, stride, random_sampling, num_random_samples = args
        tile_index = []
        file_path = os.path.join(tiff_dir, file)
        idx = index.Index()

        with rio.open(file_path) as src:
            height, width = src.height, src.width
            if isinstance(stride, float):
                stride = int(tile_size * stride)

            if random_sampling:
                for _ in range(num_random_samples):
                    x = random.randint(0, width - tile_size)
                    y = random.randint(0, height - tile_size)
                    if list(idx.intersection((x, y, x + tile_size, y + tile_size))):
                        continue
                    window = rio.windows.Window(x, y, tile_size, tile_size)
                    tile = src.read(1, window=window)
                    land_pixels = np.isnan(tile)
                    land_proportion = np.mean(land_pixels)
                    if land_proportion <= land_threshold:
                        tile_index.append({"file": file, "x": x, "y": y})
                        idx.insert(
                            len(tile_index) - 1,
                            (x, y, x + tile_size, y + tile_size),
                        )
            else:
                for y in range(0, height - tile_size + 1, stride):
                    for x in range(0, width - tile_size + 1, stride):
                        window = rio.windows.Window(x, y, tile_size, tile_size)
                        tile = src.read(1, window=window)
                        land_pixels = np.isnan(tile)
                        land_proportion = np.mean(land_pixels)
                        if land_proportion <= land_threshold:
                            tile_index.append({"file": file, "x": x, "y": y})

        return tile_index

    def _process_zip_file(self, zip_file_path, output_dir):
        """
        Process a single zipped ASF hyp3 product file.

        Args:
            zip_file_path (str): Path to the zipped ASF hyp3 product file.
            output_dir (str): Directory to save the processed GeoTIFFs.
        """
        try:
            # Create a temporary directory to extract files
            temp_dir = os.path.join(output_dir, "temp", os.path.basename(zip_file_path))
            os.makedirs(temp_dir, exist_ok=True)

            # Unzip the file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the README file, VV and VH tif files, preview images, and log file
            readme_file = None
            vv_tif_file = None
            vh_tif_file = None
            preview_files = []
            log_file = None
            granule_name = None
            granule_name_re = re.compile(r"^S1[AB]_.*")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".README.md.txt"):
                        readme_file = os.path.join(root, file)
                    elif file.endswith("_VV.tif"):
                        vv_tif_file = os.path.join(root, file)
                    elif file.endswith("_VH.tif"):
                        vh_tif_file = os.path.join(root, file)
                    elif file.endswith(".png") and "_rgb" not in file:
                        preview_files.append(os.path.join(root, file))
                    elif file.endswith(".log"):
                        log_file = os.path.join(root, file)

            # Extract the granule name from the README file
            if readme_file:
                with open(readme_file, "r") as f:
                    for line in f:
                        match = granule_name_re.match(line)
                        if match:
                            granule_name = match.group()
                            break

            # Create subdirectories for VV, VH, and preview images if they do not exist
            vv_dir = os.path.join(output_dir, "vv")
            vh_dir = os.path.join(output_dir, "vh")
            preview_dir = os.path.join(output_dir, "preview")
            os.makedirs(vv_dir, exist_ok=True)
            os.makedirs(vh_dir, exist_ok=True)
            os.makedirs(preview_dir, exist_ok=True)

            # Copy and rename the VV and VH tif files
            if granule_name:
                if vv_tif_file:
                    new_vv_tif_name = f"{granule_name}_VV.tif"
                    new_vv_tif_path = os.path.join(vv_dir, new_vv_tif_name)
                    shutil.copy(vv_tif_file, new_vv_tif_path)
                if vh_tif_file:
                    new_vh_tif_name = f"{granule_name}_VH.tif"
                    new_vh_tif_path = os.path.join(vh_dir, new_vh_tif_name)
                    shutil.copy(vh_tif_file, new_vh_tif_path)

            # Copy the preview images
            for preview_file in preview_files:
                new_preview_name = os.path.basename(preview_file)
                new_preview_path = os.path.join(preview_dir, new_preview_name)
                shutil.copy(preview_file, new_preview_path)

            # Extract key information from the log file
            log_info = None
            if log_file and granule_name:
                log_info = extract_log_info(log_file)
                log_info["Granule Name"] = granule_name

            # Remove the temporary directory
            shutil.rmtree(temp_dir)
            return log_info

        except Exception as e:
            self.logger.error(f"Error processing {zip_file_path}: {e}")
            return None
