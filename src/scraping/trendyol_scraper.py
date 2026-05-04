import json
import csv
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.scraping.base_scraper import build_driver, random_delay, retry


REVIEW_SELECTOR = ".comment-text"
RATING_SELECTOR = ".rating-star-count"
NEXT_PAGE_SELECTOR = ".pagination-next button"


def scrape_product_reviews(
    product_url: str,
    max_pages: int = 10,
    headless: bool = True,
) -> list[dict]:
    driver = build_driver(headless=headless)
    reviews = []

    try:
        driver.get(product_url)
        random_delay()

        for page in range(max_pages):
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, REVIEW_SELECTOR))
            )

            texts = driver.find_elements(By.CSS_SELECTOR, REVIEW_SELECTOR)
            ratings = driver.find_elements(By.CSS_SELECTOR, RATING_SELECTOR)

            for i, el in enumerate(texts):
                rating = ratings[i].text.strip() if i < len(ratings) else None
                reviews.append({
                    "text": el.text.strip(),
                    "rating": rating,
                    "page": page + 1,
                    "url": product_url,
                })

            next_btns = driver.find_elements(By.CSS_SELECTOR, NEXT_PAGE_SELECTOR)
            if not next_btns or not next_btns[0].is_enabled():
                break

            retry(lambda: next_btns[0].click())
            random_delay()

    finally:
        driver.quit()

    return reviews


def save_reviews(reviews: list[dict], output_path: str = "data/raw/scraped.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "rating", "page", "url"])
        writer.writeheader()
        writer.writerows(reviews)
    print(f"{len(reviews)} yorum kaydedildi -> {output_path}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else input("Trendyol urun URL: ")
    data = scrape_product_reviews(url)
    save_reviews(data)
