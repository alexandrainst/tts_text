"""Building a list of Danish bus stops and stations."""

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import re
import io


def get_raw_html_of_bus_stops_and_stations() -> str:
    """Get the raw HTML of the page with all bus stops and stations.

    Returns:
        The raw HTML of the page with all bus stops and stations.
    """
    # Set up the driver
    options = Options()
    options.add_argument("--headless")
    gecko_driver_manager = GeckoDriverManager().install()
    service = Service(executable_path=gecko_driver_manager)
    driver = webdriver.Firefox(service=service, options=options)

    # Get the page and return the raw HTML
    driver.get("https://danskejernbaner.dk/vis.stations.oversigt.php")
    raw_html = driver.page_source
    driver.quit()
    return raw_html


def build_bus_stop_and_station_dataset(output_dir: Path | str) -> list[str]:
    """Builds a list of all bus stops and stations in Denmark.

    Args:
        output_dir: The directory to save the dataset to.

    Returns:
        A list of all bus stops and stations in Denmark.
    """
    # Extract the table with all bus stops and stations from the website
    raw_html = get_raw_html_of_bus_stops_and_stations()
    soup = BeautifulSoup(raw_html, "html.parser")
    table_elt = soup.find("table")
    assert table_elt is not None, "No table found on page."
    table = pd.read_html(io.StringIO(str(table_elt)))[0]

    # Extract the list of bus stops and stations from the table, and clean them up
    dataset = table["Stationens navn"].tolist()
    dataset = list(
        {
            re.sub(r"\(.+\)|\[.+\]", "", bus_stop_or_station).strip()
            for bus_stop_or_station in dataset
        }
    )

    # Save the dataset
    dataset_path = Path(output_dir) / "bus_stops_and_stations.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
