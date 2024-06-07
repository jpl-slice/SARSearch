import hyp3_sdk as sdk


class ASFClient:
    """
    Client class to interact with ASF's HyP3 API for job submission, status checking, and file downloading.
    """

    def __init__(self, config, logger):
        """
        Initializes the ASFClient with configuration and logger.

        :param config: Configuration settings including credentials and defaults.
        :param logger: Logger instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self.hyp3 = sdk.HyP3(
            username=config["hyp3_username"], password=config["hyp3_password"]
        )
        self.succeeded_jobs = None

    def submit_jobs(self):
        """
        Submits jobs to ASF HyP3 for processing.
        """
        granules = self.config.get("granules", [])
        if not granules:
            self.logger.error("No granules provided for processing.")
            return

        batch = sdk.Batch()
        for granule in granules:
            try:
                job = self.hyp3.submit_rtc_job(
                    granule=granule,
                    resolution=self.config.get("resolution", 30),
                    scale=self.config.get("scale", "power"),
                    speckle_filter=self.config.get("speckle_filter", True),
                    name=self.config.get("job_name", "ASF_job"),
                )
                batch += job
                self.logger.info(f"Submitted job for granule {granule}")
            except Exception as e:
                self.logger.error(
                    f"Failed to submit job for granule {granule}: {str(e)}"
                )

        self.hyp3.watch(batch)
        self.logger.info(
            f"All jobs submitted and being watched. Total jobs: {len(batch)}"
        )

    def download_jobs(self):
        """
        Downloads completed jobs from ASF HyP3.
        """
        job_name = self.config.get("job_name", "ASF_job")
        jobs = self.hyp3.find_jobs(name=job_name)
        succeeded_jobs = jobs.filter_jobs(succeeded=True)

        output_dir = self.config.get("download_path", "./downloads")
        for job in succeeded_jobs:
            job.download_files(output_dir)
            self.logger.info(f"Downloaded files for job {job.job_id} to {output_dir}")

    def check_status(self, job_name=None):
        """
        Checks the status of submitted jobs.
        """
        jobs = self.hyp3.find_jobs(name=job_name)
        succeeded_jobs = jobs.filter_jobs(succeeded=True, running=False, failed=False)
        failed_jobs = jobs.filter_jobs(succeeded=False, running=False, failed=True)
        running_jobs = jobs.filter_jobs(succeeded=False, running=True, failed=True)

        self.logger.info(f"Report for job {job_name}:")
        self.logger.info(
            f"Job Status - Succeeded: {len(succeeded_jobs)}, Failed: {len(failed_jobs)}, Running: {len(running_jobs)}"
        )

    def download(self, job_name: str, output_dir: str) -> None:
        """
        Downloads the files for a specific job.
        """
        jobs = self.hyp3.find_jobs(name=job_name)
        succeeded_jobs = jobs.filter_jobs(succeeded=True, running=False, failed=False)
        self.logger.info(f"Found {len(succeeded_jobs)} succeeded jobs for {job_name}. Downloading files...")
        file_list = succeeded_jobs.download_files(output_dir)
        self.logger.info(f"Downloaded files for job {job_name} to {output_dir}")

