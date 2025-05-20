import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from typing import List
from datetime import datetime


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


def main() -> None:
    BASE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/"
    input_date = "01-05-2025"  # Format: DD-MM-YYYY
    save_path = "daily_data/"

    # Convert input date to YYYYMMDD format
    try:
        reference_date = datetime.strptime(input_date, "%d-%m-%Y").strftime("%Y%m%d")
    except ValueError:
        print("Invalid date format. Please use 'DD-MM-YYYY'.")
        return

    ensure_directory(save_path)
    try:
        csv_links = fetch_csv_links(BASE_URL, reference_date)
        if not csv_links:
            print(f"No CSV files found for date {reference_date}.")
            return
        download_and_save_csvs(csv_links, save_path)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    main()
