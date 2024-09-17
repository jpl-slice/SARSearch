import os
import argparse
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define a function to process a single image
def process_image(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Read the first band (or modify for multi-band if necessary)
        image = src.read(1)
        # Mask NaNs (land areas) for ocean statistics
        valid_pixels = image[~np.isnan(image)]
        # If no valid pixels exist, skip this image
        if valid_pixels.size == 0:
            raise ValueError(f"No valid ocean pixels found in {tiff_path}")
        # Calculate mean and std for the individual image
        img_mean = np.mean(valid_pixels)
        img_std = np.std(valid_pixels)
        # Return the sum and sum of squares for running avg, and pixel count
        return np.sum(valid_pixels), np.sum(np.square(valid_pixels)), valid_pixels.size, img_mean, img_std

# Function to gather GeoTIFF files from the given directory or file path
def gather_geotiff_paths(input_path):
    # If input_path is a directory, collect all GeoTIFF files
    if os.path.isdir(input_path):
        geo_tiff_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.tif')]
    elif os.path.isfile(input_path):
        geo_tiff_paths = [input_path] if input_path.endswith('.tif') else []
    else:
        raise ValueError(f"Invalid path: {input_path}")
    
    if not geo_tiff_paths:
        raise ValueError(f"No GeoTIFF files found at {input_path}")
    
    return geo_tiff_paths

# Main function for processing the images
def main(input_path, output_csv):
    # Gather the GeoTIFF files from the input path
    geo_tiff_paths = gather_geotiff_paths(input_path)

    # Initialize accumulators for running stats
    total_sum = 0
    total_sum_of_squares = 0
    total_pixels = 0
    error_log = []

    # Create a list to store rows of file-level statistics
    stats_list = []

    # Loop over GeoTIFF files with a progress bar
    for tiff_path in tqdm(geo_tiff_paths, desc="Processing images"):
        try:
            # Process each image, and accumulate results
            img_sum, img_sum_of_squares, img_pixel_count, img_mean, img_std = process_image(tiff_path)
            
            # Update running totals for global mean/std
            total_sum += img_sum
            total_sum_of_squares += img_sum_of_squares
            total_pixels += img_pixel_count
            
            # Append individual file stats to the list as a dictionary
            stats_list.append({
                "file_name": tiff_path,
                "mean": img_mean,
                "std": img_std
            })

            # Compute the running mean and std
            if total_pixels > 0:
                running_mean = total_sum / total_pixels
                running_std = np.sqrt((total_sum_of_squares / total_pixels) - (running_mean ** 2))
                print(f"Running Mean: {running_mean:.4f}, Running Std: {running_std:.4f}")
            
        except Exception as e:
            # Handle exceptions and log errors without stopping the loop
            error_log.append((tiff_path, str(e)))
            print(f"Error processing {tiff_path}: {e}")

    # Convert the list of statistics into a DataFrame using pd.concat
    df_stats = pd.concat([pd.DataFrame([row]) for row in stats_list], ignore_index=True)

    # Compute final running averages if we processed any pixels
    if total_pixels > 0:
        final_mean = total_sum / total_pixels
        final_std = np.sqrt((total_sum_of_squares / total_pixels) - (final_mean ** 2))
        print(f"\nFinal Running Mean: {final_mean}, Final Running Std: {final_std}")
    else:
        print("\nNo valid pixels were processed.")

    # Output error log (if any)
    if error_log:
        print("\nErrors encountered:")
        for error in error_log:
            print(f"File: {error[0]}, Error: {error[1]}")

    # Save the DataFrame to a CSV file
    df_stats.to_csv(output_csv, index=False)
    print(f"\nPer-file statistics saved to {output_csv}")

# Entry point for command-line execution
if __name__ == "__main__":
    # Setup argparse to handle input directory or file and output CSV
    parser = argparse.ArgumentParser(description="Process SAR GeoTIFF images and compute mean and std.")
    parser.add_argument('input_path', type=str, help="Path to a directory or GeoTIFF file")
    parser.add_argument('output_csv', type=str, help="Path to output CSV file")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the main function
    main(args.input_path, args.output_csv)

