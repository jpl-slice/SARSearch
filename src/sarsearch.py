import argparse
import sys
from asf_client import ASFClient
from sarutils import SARUtils
from config import load_config
from logger import setup_logger


def process_zip_files(zip_dir: str, dest_dir: str):
    """
    Process zip files from ASF and extract them to the destination directory.
    """
    sar_utils = SARUtils()
    sar_utils.process_zip_files_in_directory(zip_dir, dest_dir)
    logger.info(f"Processed and extracted zip files from {zip_dir} to {dest_dir}")


def apply_landmask_to_files(input_dir: str, output_dir: str, landcover_tif: str):
    """
    Apply the landmask to already downloaded files.
    """
    sar_utils = SARUtils(landcover_tif)
    sar_utils.multiprocess_apply_landmask(input_dir, output_dir)
    logger.info(f"Applied landmask to files in {input_dir} and saved to {output_dir}")


def asf_hyp3(config: dict, logger):
    """
    Interface with ASF HyP3.
    """
    client = ASFClient(config, logger)

    if args.granule_file:
        with open(args.granule_file, "r") as f:
            config["granules"] = [line.strip() for line in f if line.strip()]

    if args.submit:
        client.submit_jobs()
    elif args.download:
        client.download_jobs()
    elif args.status:
        client.check_status()
    else:
        logger.error("No action specified")
        sys.exit(1)
    logger.info(f"Submitted jobs to ASF HyP3")


def main():
    """
    Main function to parse command-line arguments and initiate actions based on those arguments.
    """
    parser = argparse.ArgumentParser(description="SARSearch: a tool for processing SAR data from ASF")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-parser for processing zip files
    parser_zip = subparsers.add_parser('process_zip', help="Process zip files from ASF")
    parser_zip.add_argument('--input_dir', type=str, help="Directory containing the zip files")
    parser_zip.add_argument('--output_dir', type=str, help="Destination directory for extracted files")
    parser_zip.add_argument('--num_processes', type=int, help="Number of processes to use for extraction", default=1)

    # Sub-parser for applying landmask to already downloaded files
    parser_apply = subparsers.add_parser('apply_landmask', help="Apply landmask to downloaded files")
    parser_apply.add_argument('--input_dir', type=str, help="Directory containing the input GeoTIFF files")
    parser_apply.add_argument('--output_dir', type=str, help="Directory to save the masked GeoTIFF files")
    parser_apply.add_argument('--landcover_tif', type=str, help="Path to the landcover GeoTIFF file")

    # Sub-parser for searching and downloading ASF frames
    parser_search = subparsers.add_parser('asf_hyp3', help="Interface with ASF hyp3. Search and download frames")
    parser_search.add_argument('config', type=str, help="Path to configuration file")
    parser_search.add_argument('username', type=str, help="ASF username")
    parser_search.add_argument('password', type=str, help="ASF password")
    parser_search.add_argument('granule-file', type=str, help="File path for granule list")
    parser_search.add_argument('download', type=str, help="Download completed jobs to this directory")
    parser_search.add_argument('search_params', type=str, help="JSON string of search parameters")
    parser_search.add_argument('status', help="Check status of submitted jobs", action="store_true")

    args = parser.parse_args()
    logger = setup_logger()

    if args.command == 'process_zip':
        sar_utils = SARUtils(logger)
        sar_utils.process_zip_files_in_directory(args.input_dir, args.output_dir, num_processes=args.num_processes)
        logger.info(f"Processed and extracted zip files from {args.input_dir} to {args.output_dir}")
    elif args.command == 'apply_landmask':
        sar_utils = SARUtils(args.landcover_tif)
        sar_utils.multiprocess_apply_landmask(args.input_dir, args.output_dir)
        logger.info(f"Applied landmask to files in {args.input_dir} and saved to {args.output_dir}")
    elif args.command == 'asf_hyp3':
        config = load_config(args.config)
        asf_hyp3(config, logger=logger)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
