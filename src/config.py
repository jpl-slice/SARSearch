import yaml


def load_config(file_path):
    """
    Loads configuration from a YAML file.

    :param file_path: Path to the YAML configuration file```python
    :param file_path: Path to the YAML configuration file.
    :return: A dictionary containing configuration settings.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    granules_file = config.get('granule_filepath')
    if granules_file:
        with open(granules_file, 'r') as f:
            config['granules'] = [line.strip() for line in f if line.strip()]
    return config

