import os
import argparse
import subprocess
import shutil

def check_kaggle_cli():
    """Checks if the Kaggle CLI is installed and configured."""
    if not shutil.which("kaggle"):
        raise EnvironmentError("Kaggle CLI not found. Please run 'pip install kaggle' and ensure it's in your PATH.")
    
    # Check for credentials
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_config_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"Warning: 'kaggle.json' not found in {kaggle_config_dir}. API calls might fail if not authenticated via env vars.")

def download_specific_file(kernel_slug, file_name, output_dir="kaggle_output"):
    """
    Downloads a specific file from a Kaggle kernel using the CLI.
    """
    check_kaggle_cli()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Downloading '{file_name}' from kernel: {kernel_slug} ---")
    
    # Construct command: kaggle kernels output <slug> -f <filename> -p <path>
    command = [
        "kaggle", "kernels", "output", 
        kernel_slug,
        "-f", file_name,
        "-p", output_dir
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Success! File saved to: {os.path.join(output_dir, file_name)}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file:\n{e.stderr}")
        print("Tip: Check the filename exactly matches what is in the Kaggle Output tab.")
        print("Tip: If the file is inside a folder, you might need to include the path, e.g., 'folder/file.zip'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Specific File from Kaggle Kernel")
    parser.add_argument("--kernel", type=str, required=True, help="Kaggle kernel slug (e.g. username/notebook-name)")
    parser.add_argument("--file", type=str, required=True, help="Specific file path to download (e.g. 'results.zip')")
    parser.add_argument("--output", type=str, default="downloaded_results", help="Local directory to save results")
    
    args = parser.parse_args()
    
    download_specific_file(args.kernel, args.file, args.output)
