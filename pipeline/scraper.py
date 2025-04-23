import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time


def get_reviews_from_url(course_url, max_reviews=100):
    """
    Scrapes Coursera course reviews directly from the HTML of the course page.

    Args:
        course_url (str): URL of the Coursera course.
        max_reviews (int): Number of reviews to scrape.

    Returns:
        pd.DataFrame: DataFrame containing scraped reviews.
    """
    reviews = []
    print(f"🌀 Scraping reviews for course: {course_url}")

    response = requests.get(course_url)
    if response.status_code != 200:
        print(f"Failed to fetch course page: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all review sections from the page (these may vary based on site layout)
    review_elements = soup.find_all('div', class_='rc-Review')
    for review_element in review_elements:
        review_text = review_element.get_text(strip=True)
        if review_text:
            reviews.append(review_text)

        if len(reviews) >= max_reviews:
            break

    print(f"✅ Scraped {len(reviews)} reviews.")
    return pd.DataFrame({"Review": reviews})


if __name__ == "__main__":
    course_url = input("Enter Coursera course URL: ")
    df = get_reviews_from_url(course_url, max_reviews=100)
    if not df.empty:
        df.to_csv("scraped_reviews.csv", index=False)
        print("📁 Saved reviews to scraped_reviews.csv")
    else:
        print("No reviews found.")
