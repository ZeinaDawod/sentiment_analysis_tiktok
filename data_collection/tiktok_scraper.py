import asyncio
import json
import csv
import os
from playwright.async_api import async_playwright
import logging

from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "tiktok_scraper.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

KEYWORDS = [
    {"query": "CeraVe cleanser love it review", "class": "pos"},
    {"query": "CeraVe worst skincare review", "class": "neg"},
    {"query": "CeraVe before after test", "class": "neutral"}
]
MAX_VIDEOS = 25
OUTPUT_FILE = "./data/tiktok_results_alll.csv"


async def scrape_tiktok_search(keyword: str, class_label: str, max_videos: int = 25):
    results = []
    seen_urls = set()
    search_url = f"https://www.tiktok.com/search/video?q={keyword.replace(' ', '%20')}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )

        page = await context.new_page()

        intercepted = []
        api_hit_count = 0

        async def handle_response(response):
            nonlocal api_hit_count
            url = response.url
            if "api/search/general/full" in url or "api/search/item/full" in url:
                api_hit_count += 1
                try:
                    data = await response.json()
                    intercepted.append(data)
                    logger.debug(f"[API] Response #{api_hit_count} from: {url[:80]}")
                except Exception as e:
                    logger.error(f"[API] Failed to parse response: {e}")

        page.on("response", handle_response)

        logger.info(f"Opening TikTok for keyword: '{keyword}' (class={class_label})")
        await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(10)

        scroll_count = 0
        max_scrolls = 20
        no_new_results_streak = 0

        while len(results) < max_videos and scroll_count < max_scrolls:
            before_count = len(results)

            for data in intercepted:
                items = data.get("data", []) or data.get("item_list", [])

                for item in items:
                    try:
                        video_item = item.get("item") if "item" in item else item
                        video_id = video_item.get("id") or video_item.get("aweme_id", "")
                        author = video_item.get("author", {})
                        author_name = (
                                author.get("uniqueId")
                                or author.get("unique_id")
                                or author.get("nickname", "unknown")
                        )
                        video_url = f"https://www.tiktok.com/@{author_name}/video/{video_id}"
                        desc = video_item.get("desc") or video_item.get("share_info", {}).get("desc", "")

                        if video_id and video_url not in seen_urls and len(results) < max_videos:
                            seen_urls.add(video_url)
                            results.append({
                                "video_url": video_url,
                                "description": desc.strip(),
                                "class": class_label,
                            })
                            logger.info(f"Collected ({len(results)}) videos | Current: {video_url[:60]}")
                    except Exception:
                        continue

            new_this_round = len(results) - before_count
            logger.debug(f"Scroll {scroll_count + 1}/{max_scrolls} | API hits: {len(intercepted)} | New: {new_this_round} | Total: {len(results)}")

            intercepted.clear()

            if new_this_round == 0:
                no_new_results_streak += 1
                if no_new_results_streak >= 4:
                    break
            else:
                no_new_results_streak = 0

            if len(results) >= max_videos:
                break

            await page.evaluate("window.scrollBy(0, window.innerHeight * 2.5)")
            await asyncio.sleep(3.5)
            scroll_count += 1

        await browser.close()

    return results

def save_results(results, output_file, append=False):
    if not results:
        logger.warning("No results saved.")
        return

    file_exists = os.path.exists(output_file)
    mode = "a" if append else "w"

    with open(output_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_url", "description", "class"])

        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} rows to '{output_file}' (append={append})")


async def main():
    total_count = 0


    for kw in KEYWORDS:
        keyword = kw["query"]
        class_label = kw["class"]

        logger.info("=" * 60)
        logger.info(f"Searching for TikTok videos: '{keyword}' -> class: {class_label}")
        logger.info("=" * 60)

        results = await scrape_tiktok_search(keyword, class_label, MAX_VIDEOS)

        logger.info(f"Got {len(results)} results for keyword '{keyword}'")
        for i, r in enumerate(results, 1):
            logger.debug(f"[{i}] {r['video_url']}")
            logger.debug(f"    Description: {r['description'][:100]}...")

        save_results(results, OUTPUT_FILE, append=True)
        total_count += len(results)


        await asyncio.sleep(5)

    logger.info("=" * 60)
    logger.info(f"TOTAL results collected across all keywords: {total_count}")
    logger.info(f"Saved incrementally to '{OUTPUT_FILE}'")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())