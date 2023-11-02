"""Scraping and preprocessing of the sundhed.dk text corpus."""

from omegaconf import DictConfig
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from .utils import get_soup


BASE_URL = "https://www.sundhed.dk"


def build_sundhed_dk_dataset(cfg: DictConfig) -> list[str]:
    """Build the sundhed.dk dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of articles from sundhed.dk.
    """
    ChromeDriverManager().install()

    soup = get_soup(url=BASE_URL + "/borger/patienthaandbogen/", dynamic=True)
    category_urls = [
        BASE_URL + url_suffix.a["href"]
        for url_suffix in soup.find_all("li", class_="list-group-item")
    ]

    all_articles: list[str] = list()
    for category_url in tqdm(category_urls, leave=True):
        all_articles.extend(extract_all_category_articles(url=category_url))

    return all_articles


def extract_all_category_articles(url: str) -> list[str]:
    """Extract all articles from a category page.

    These pages have arbitrarily nested subcategories, so this function is called
    recursively.

    Args:
        url: The URL of the category page.

    Returns:
        A list of articles from the category.
    """
    category_articles: list[str] = list()

    soup = get_soup(url=url, dynamic=True)
    subcategory_urls = [
        BASE_URL + url_suffix.a["href"]
        for url_suffix in soup.find_all("div", class_="closed")
    ]

    for subcategory_url in tqdm(subcategory_urls, leave=False):
        subcategory_articles = extract_all_category_articles(url=subcategory_url)
        category_articles.extend(subcategory_articles)

    return category_articles
