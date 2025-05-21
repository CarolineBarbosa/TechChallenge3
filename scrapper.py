import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from typing import List


def ensure_directory(path: str) -> None:
    """Create the directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def fetch_csv_links(base_url: str, date_str: str) -> List[str]:
    """
    Scrape CSV links from the base URL that match a reference date.

    Args:
        base_url: The root URL of the CSV files.
        date_str: The date string (e.g., '20250501') to match in file names.

    Returns:
        A list of full URLs to the matching CSV files.
    """
    response = requests.get(base_url)
    response.raise_for_status()  # Raise exception if request fails

    soup = BeautifulSoup(response.content, "html.parser")
    return [
        urljoin(base_url, link.get("href"))
        for link in soup.find_all("a")
        if link.get("href", "").endswith(".csv") and date_str in link.get("href")
    ]


def download_and_save_csvs(csv_urls: List[str], save_dir: str, max_files: int = 2) -> None:
    """
    Download and save CSV files to a local directory.

    Args:
        csv_urls: List of CSV URLs to download.
        save_dir: Directory where CSV files will be saved.
        max_files: Maximum number of CSVs to download (default is 2).
    """
    for csv_url in csv_urls[:max_files]:
        print(f"Downloading: {csv_url}")
        df = pd.read_csv(csv_url, encoding="latin1", sep=",")

        filename = os.path.basename(csv_url)
        full_path = os.path.join(save_dir, filename)

        df.to_csv(full_path, index=False)
        print(f"Saved to: {full_path}")

def check_file_exists(save_path, reference_date_formatted):
    """
    Check if a file with the reference_date_formatted in its name exists in the given path.

    Args:
        save_path: The directory to check for the file.
        reference_date_formatted: The date string to look for in file names.

    Returns:
        True if a matching file exists, False otherwise.
    """
    for filename in os.listdir(save_path):
        if reference_date_formatted in filename:
            return True
    return False


def scrape_and_collect_data(reference_date_formatted: str) -> None:
    """
    Scrape and collect CSV files for a given reference date.

    Args:
        reference_date: The reference date in the format DD-MM-YYYY.
    """
    BASE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/"
    save_path = "daily_data/"

    # Convert reference_date to the required format YYYYMMDD

    ensure_directory(save_path)
    if check_file_exists(save_path, reference_date_formatted):
        print(f"File for date {reference_date_formatted} already exists in the directory.")
        return
    try:
        csv_links = fetch_csv_links(BASE_URL, reference_date_formatted)
        if not csv_links:
            print(f"No CSV files found for date {reference_date_formatted}.")
            return
        download_and_save_csvs(csv_links, save_path)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")

