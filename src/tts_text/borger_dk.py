"""Scraping and preprocessing of the borger.dk text corpus."""

from pathlib import Path
from unicodedata import normalize
from bs4 import Tag, BeautifulSoup
from omegaconf import DictConfig
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import logging
import re
from .utils import extract_sentences, get_soup


BASE_URL = "https://www.borger.dk"
SUBSITES_TO_IGNORE = [
    # Its own special subsite, with a question-answer-type navigation, which leads
    # to articles on the main site
    "/kampagnesider/",  
]


def build_borger_dk_dataset(cfg: DictConfig) -> list[str]:
    """Build the borger.dk dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of articles from borger.dk.
    """
    # Load dataset if it already exists
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "borger_dk.txt"
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            return f.read().split("\n")

    # Install the Chrome driver, if it isn't already installed
    logging.getLogger("WDM").setLevel(logging.WARNING)
    ChromeDriverManager().install()

    # Get the overall categories from the front page
    # retry if it fails
    soup: BeautifulSoup = BeautifulSoup("")
    while not soup.contents:
        soup = get_soup(url=BASE_URL, dynamic=True)

    category_urls = [
        BASE_URL + url_suffix.contents[1].attrs["href"]
        for url_suffix in soup.find_all("li", class_="col-12")
    ]

    # Extract all articles
    all_articles: list[str] = list()
    found_urls: list[str] = list()
    desc = "Extracting articles from borger.dk"
    for category_url in tqdm(category_urls, desc=desc, leave=True):
        # Borger.dk is structured in to categories, which are structured in to
        # subcategories, which are structured in to articles. Every article contains
        # a text, and a link to another article. The links go both up, down and
        # sideways in the hierarchy, so we need to restrict ourselves to one category
        # at a time, and only follow links downwards, otherwise the recursion will
        # never end. This is why we use the category variable.
        category = category_url.split("/")[-1]
        category_articles, found_urls = extract_all_articles(
            cfg=cfg, url=category_url, category=category, found_urls=found_urls
        )
        all_articles.extend(category_articles)

    # Split the articles into sentences
    dataset = extract_sentences(
        corpus=all_articles, min_sentence_length=cfg.min_sentence_length
    )

    # Ensure sentences end with appropriate punctuation
    dataset = [
        sentence + "." if re.match(r".*[.?!]$", sentence) is None else sentence
        for sentence in dataset
    ]

    # Remove sentences not starting with a capital letter
    dataset = [sentence for sentence in dataset if sentence[0].isupper()]

    # Save the dataset
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset


def extract_all_articles(
    cfg: DictConfig, url: str, category: str, parsed_urls: list[str] = [], found_urls=[]
) -> tuple[list[str], list[str]]:
    """Extract all articles from a category page.

    These pages have arbitrarily nested subcategories, so this function is called
    recursively. Each recursion checks the sidebar for new subcategories, and if none
    are found, then the page is assumed to contain an article. The article is
    extracted, and the links in the article are followed and extracted as well.
    The recursion ends when no new subcategories are found, and the articles are
    either only text, or theirs links are not suitable for further recursion. See
    `get_suitable_links` for the criteria for suitable links.

    Args:
        cfg: 
            The Hydra configuration object.
        category: 
            The category of the page.
        parsed_urls: 
            A list of URLs that have already been parsed.
        found_urls: 
            A list of URLs that have already been found, but not necessarily parsed.

    Returns:
        A list of articles, and an updated list of found URLs.
    """
    # The accumulator of the recursion
    accumulated_articles: list[str] = list()

    # Often connection fails when we try to access a page, so we retry a few times
    # but we give up fairly fast due to time constraints.
    soup: BeautifulSoup = BeautifulSoup("")
    for _ in range(cfg.scraping_retry_connection_limit):
        soup = get_soup(url=url, dynamic=True)
        if soup.contents:
            break

    if not soup.contents:
        return [], found_urls

    # Get all good links in collapsable sidebar
    urls_to_search: list[str] = []
    for collapsed_div in soup.find_all(name="div", class_="collapse"):
        if isinstance(collapsed_div, Tag):
            urls_to_search.extend(
                get_suitable_links(collapsed_div, category, url, found_urls)
            )

    # If there are no new subcategories, then we assume that the page has an article
    # and we extract it. Any article is accompanied by a link to another
    # article, so we open these and extract them as well
    if not urls_to_search:
        scraped_text_on_current_page: list[str] = list()
        for content_div in soup.find_all(name="div", class_="content-text"):
            # Extract the article
            article_str = ""
            for paragraph in content_div.find_all(name="p"):
                article_str += paragraph.text + " "

            # Clean the final article text
            article_str = normalize("NFKC", article_str).replace("–", "-").strip()
            article_str = re.sub(
                r"[^a-zA-Z0-9æøåéÉÆØÅ.,\-?!:\n\/()\[\] ]%\"\'", "", article_str
            )

            # Some articles are only a link to another article, so we ignore them
            if article_str:
                scraped_text_on_current_page.append(article_str)

            # Find all the good links to nested articles.
            for list_link in content_div.find_all(name="ul", class_="list--links"):
                urls_to_search.extend(
                    get_suitable_links(list_link, category, url, found_urls)
                )

            accumulated_articles, found_urls = follow_links_and_extract_articles(
                cfg=cfg,
                category=category,
                accumulated_articles=accumulated_articles,
                top_url=url,
                urls_to_search=urls_to_search,
                parsed_urls=parsed_urls,
                found_urls=found_urls,
            )

        # We have now followed all the good links in the articles hence
        # accumulated_articles should now only contain text from the nested articles
        # on the current page. We add these to the text we have already found on this
        # page
        return scraped_text_on_current_page + accumulated_articles, found_urls

    # Recursively follow links and extract articles
    return follow_links_and_extract_articles(
        cfg=cfg,
        category=category,
        accumulated_articles=accumulated_articles,
        top_url=url,
        urls_to_search=urls_to_search,
        parsed_urls=parsed_urls,
        found_urls=found_urls,
    )


def follow_links_and_extract_articles(
    cfg: DictConfig,
    category: str,
    accumulated_articles: list[str],
    top_url: str,
    urls_to_search: list[str],
    parsed_urls: list[str],
    found_urls: list[str],
) -> tuple[list[str], list[str]]:
    """Follow links and extract articles.

    Args:
        cfg: The Hydra configuration object.
        category: The category of the page.
        accumulated_articles: The articles that have already been accumulated.
        top_url: The top URL.
        urls_to_search: The URLs to search.
        parsed_urls: The URLs that have already been parsed.
        found_urls: The URLs that have already been found, but not necessarily
            parsed.

    Returns:
        A tuple of the accumulated articles and the found URLs.
    """
    desc = f"Extracting articles from {top_url}"
    parsed_urls.append(top_url)
    found_urls.extend(urls_to_search + [top_url])
    for url in tqdm(urls_to_search, desc=desc, leave=False):
        if url in parsed_urls:
            continue
        extracted_articles, found_urls = extract_all_articles(
            cfg=cfg,
            url=url,
            category=category,
            parsed_urls=parsed_urls,
            found_urls=found_urls,
        )
        accumulated_articles.extend(extracted_articles)
    return accumulated_articles, found_urls


def get_suitable_links(
    list_link: Tag, category: str, url: str, found_urls: list[str]
) -> list[str]:
    """Get suitable links from a list of links.

    Suitable links are links:
        - to the same category
        - that are not going up in the hierarchy
        - that are not referencing alternative navigation hierarchies
        - that are not links to files hosted on borger.dk
        - that are not javascript-based navigation links
        - that have not already been found
        
    Args:
        list_link: The list HTML element, containing the links.
        category: The category of the links.
        url: The URL of the page where the list of links was found.
        found_urls: The URLs that have already been found, but not necessarily
            parsed.

    Returns:
        A list of suitable links.
    """
    return [
        BASE_URL + url_suffix.attrs["href"]
        for url_suffix in list_link.find_all(name="a")
        if (
            url_suffix.attrs is not None  # It's a link
            and category in url_suffix.attrs["href"]  # Only links to the same category
            and url_suffix.attrs["href"]
            not in url  # We are not going up in the hierarchy
            and not any(
                subsite in url_suffix.attrs["href"] for subsite in SUBSITES_TO_IGNORE
            )  # Alternative navigation hierarchies
            and BASE_URL not in url_suffix.attrs["href"]  # Links which contain
            # the base url are not articles, but are javascript-based
            # navigation or links to files hosted on borger.dk
            and BASE_URL + url_suffix.attrs["href"]
            not in found_urls  # We have not already found this link
        )
    ]
