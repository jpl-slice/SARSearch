# SARSearch
SARSearch is a command-line tool designed for managing and processing Synthetic Aperture Radar (SAR) data via ASF's HyP3 service.

## Project Overview
**SARSearch** is a command-line tool designed to manage and process Synthetic Aperture Radar (SAR) data via ASF's
HyP3 service. It simplifies the process of submitting jobs, downloading results, and checking job statuses, 
facilitating efficient SAR data analysis.

## Features
- **Job Submission**: Submit SAR processing jobs using a list of granule identifiers.
- **Job Monitoring**: Check the status of submitted jobs.
- **Result Downloading**: Download completed job files to a specified directory.
- **Flexible Configuration**: Use a YAML configuration file for settings.
- **Granule Management**: Manage granule lists via text file input.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourgithub/sarsearch.git
cd sarsearch
pip install -r requirements.txt
```

## Configuration Setup

Modify the config.yaml to include your ASF HyP3 credentials and any default settings. Specify your granule list file
path if not using the command line option.

## Usage
Command Line Arguments

```bash
    -c, --config: Path to the YAML configuration file.
    -g, --granule-file: File path for a text file containing granule names.
    --submit: Submit jobs from the list of granules.
    --download: Download completed jobs.
    --status: Check the status of submitted jobs.
```

## Examples

Submit Jobs:

```bash
python asf_tool.py --submit -g path/to/granules.txt
```


Check Job Status:

```bash
python asf_tool.py --status
```

Download Results:

```bash
python asf_tool.py --download
```

## Dependencies
- Python 3.8+
- hyp3_sdk
- PyYAML

Ensure all dependencies are installed using `pip install -r requirements.txt`.
Contributing

Contributors are welcome to propose enhancements or fix bugs. For major changes, please open an issue first to discuss
what you would like to change.

1. Fork the repository.
1. Create your feature branch (git checkout -b feature/fooBar).
1. Commit your changes (git commit -am 'Add some fooBar').
1. Push to the branch (git push origin feature/fooBar).
1. Open a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
