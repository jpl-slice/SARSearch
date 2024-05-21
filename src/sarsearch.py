import argparse
import sys
from asf_client import ASFClient
from sarutils import SARUtils
from config import load_config
from logger import setup_logger


def main():
    """
    Main function to parse command-line arguments and initiate actions based on those arguments.
    """
    parser = argparse.ArgumentParser(
        description="SARSearch: A command-line tool for managing SAR data processing."
    )
    parser.add_argument(
        "-c", "--config", help="Path to configuration file", required=False
    )
    parser.add_argument(
        "-g", "--granule-file", help="File path for granules list", required=False
    )
    parser.add_argument("--submit", help="Submit jobs from list", action="store_true")
    parser.add_argument(
        "--download", help="Download completed jobs", action="store_true"
    )
    parser.add_argument(
        "--status", help="Check status of submitted jobs", action="store_true"
    )
    parser.add_argument(
        "--maskland", help="Apply mask to directory of .tifs", action="store_true"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.granule_file:
        with open(args.granule_file, "r") as f:
            config["granules"] = [line.strip() for line in f if line.strip()]

    logger = setup_logger()
    client = ASFClient(config, logger)

    if args.submit:
        client.submit_jobs()
    elif args.download:
        client.download_jobs()
    elif args.status:
        client.check_status()
    else:
        logger.error("No action specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
