import logging


def setup_logger():
    """
    Sets up and returns a logger with INFO logging level.
    :return: Configured logger.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("SARSearch")
