import argparse
import sys
from asf_client import ASFClient
from sarutils import SARUtils
from config import load_config
from logger import setup_logger
from typing import Union


def int_or_float(value: Union[str, int, float]):
    """
    Utility function to convert a string to an integer or float.

    Args:
        value: The value to convert.
    """
    try:
        # Try converting to integer
        return int(value)
    except ValueError:
        # If it fails, try converting to float
        return float(value)


def merge_config_with_args(config, args):
    """
    Utility function to convert argparse namespace to dictionary and remove None values

    Args:
        config: Configuration dictionary
        args: argparse namespace object

    """
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Remove 'config' key since it's not part of the actual configuration
    args_dict.pop('config', None)

    # Update config dictionary with command line arguments
    config.update(args_dict)

    return config


def main():
    """
    Main function to parse command-line arguments
    """
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="SARSearch: a tool for processing SAR data from ASF"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of processors to use",
        default=1,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    # Sub-parser for processing zip files
    parser_zip = subparsers.add_parser("process_zip", help="Process zip files from ASF")
    parser_zip.add_argument(
        "--input_dir", type=str, help="Directory containing the zip files"
    )
    parser_zip.add_argument(
        "--output_dir", type=str, help="Destination directory for extracted files"
    )

    # Sub-parser for applying landmask to already downloaded files
    parser_apply = subparsers.add_parser(
        "apply_landmask", help="Apply landmask to downloaded files"
    )
    parser_apply.add_argument(
        "--input_dir", type=str, help="Directory containing the input GeoTIFF files"
    )
    parser_apply.add_argument(
        "--output_dir", type=str, help="Directory to save the masked GeoTIFF files"
    )
    parser_apply.add_argument(
        "--landcover_tif", type=str, help="Path to the landcover GeoTIFF file"
    )
    parser_apply.add_argument(
        "--water_class", type=int, help="Water value for landcover classification"
    )

    # Sub-parser for generating a tile map
    parser_tile = subparsers.add_parser(
        "tile_map", help="Generate a tile map from a directory of GeoTIFF files"
    )
    parser_tile.add_argument(
        "--input_dir", type=str, help="Directory containing the zip files"
    )
    parser_tile.add_argument(
        "--output_file",
        type=str,
        help="Destination file for the tile map"
    )
    parser_tile.add_argument(
        "--tile_size", type=int, help="Size of the tiles in pixels", default=500
    )
    parser_tile.add_argument(
        "--land_threshold", type=float, help="Threshold for land classification", default=0.1
    )
    parser_tile.add_argument(
        "--stride", type=int_or_float, help="Stride for the sliding window", default=0.5
    )
    parser_tile.add_argument(
        "--random_sampling", help="Randomly sample tiles", action="store_true"
    )
    parser_tile.add_argument(
        "--num_random_samples", type=int, help="Number of random samples", default=1000
    )

    # Sub-parser for interacting with ASF Hyp3
    parser_hyp3 = subparsers.add_parser(
        "asf_hyp3", help="Interface with ASF hyp3. Search and download frames"
    )
    parser_hyp3.add_argument("--config", type=str, help="Path to configuration file")
    parser_hyp3.add_argument("--hyp3_username", type=str, help="ASF username")
    parser_hyp3.add_argument("--hyp3_password", type=str, help="ASF password")
    parser_hyp3.add_argument("--job_name", type=str, help="Name of the job", default=None)
    parser_hyp3.add_argument("--granule_file", type=str, help="File containing granule IDs", default=None)

    # Add mutually exclusive group for submitting jobs, checking status, and downloading
    hyp3_group = parser_hyp3.add_mutually_exclusive_group()
    hyp3_group.add_argument(
        "--submit", help="Submit jobs to ASF HyP3", action="store_true"
    )
    hyp3_group.add_argument(
        "--download", type=str, help="Download completed jobs to this directory"
    )
    parser_hyp3.add_argument(
        "--status", action="store_true", help="Check status of a submitted job. Argument is job name"
    )

    args = parser.parse_args()

    if args.command == "process_zip":
        sar_utils = SARUtils(logger=logger)
        sar_utils.process_zip_files_in_directory(args.input_dir, args.output_dir, num_workers=args.num_workers)
        logger.info(
            f"Processed and extracted zip files from {args.input_dir} to {args.output_dir}"
        )
    elif args.command == "apply_landmask":
        sar_utils = SARUtils(logger, landcover_tif_path=args.landcover_tif)
        sar_utils.multiprocess_apply_landmask(args.input_dir, args.output_dir, args.water_class,
                                              num_workers=args.num_workers)
        logger.info(
            f"Applied landmask to files in {args.input_dir} and saved to {args.output_dir}"
        )
    elif args.command == "tile_map":
        sar_utils = SARUtils(logger)
        sar_utils.generate_tile_map(
            args.input_dir,
            args.output_file,
            tile_size=args.tile_size,
            land_threshold=args.land_threshold,
            stride=args.stride,
            random_sampling=args.random_sampling,
            num_random_samples=args.num_random_samples,
        )
    elif args.command == "asf_hyp3":
        config = load_config(args.config)
        config = merge_config_with_args(config, args)

        try:
            asf_client = ASFClient(config=config, logger=logger)
        except Exception as e:
            logger.error(f"Error initializing ASFClient: {e}")
            sys.exit(1)

        # Check if we are submitting jobs, checking status, or downloading
        if config['submit']:
            if 'granule_file' in config and config['granule_file'] is not None:
                with open(config['granule_file'], "r") as f:
                    config["granules"] = [line.strip() for line in f if line.strip()]
                asf_client.submit_jobs()
        elif config['status']:
            logger.info(f"Checking status of job: {config['job_name']}")
            asf_client.check_status(job_name=config['job_name'])
        elif config['download']:
            if 'job_name' in config and config['job_name'] is not None:
                asf_client.download(job_name=config['job_name'], output_dir=config['download'])
            else:
                logger.error("job_name not specified. Cannot download jobs without --job_name")
                sys.exit(1)
        else:
            logger.error("No action specified")
            sys.exit(1)


if __name__ == "__main__":
    main()
