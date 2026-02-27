#!/usr/bin/env python3
"""Scraper for jobsearch.az vacancies API.

Pagination works via an `ignore` mechanism:
- The server returns a batch of vacancies plus `ignore` (base64-encoded IDs)
  and `ignore_hash` to use for the next request.
- If the response doesn't include those fields, we build `ignore` ourselves
  from accumulated IDs and omit the hash.
- Stops when a page returns no new vacancy IDs.
"""

import base64
import csv
import time
import warnings
from pathlib import Path

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

BASE_URL = "https://jobsearch.az/api-az/vacancies-az"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "jobsearch.csv"
DELAY = 0.5  # seconds between requests

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8,ru;q=0.7",
    "DNT": "1",
    "Referer": "https://jobsearch.az/vacancies",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def encode_ids(ids: list) -> str:
    return base64.b64encode(",".join(str(i) for i in ids).encode()).decode()


def extract_vacancies(data) -> list[dict]:
    """Return the vacancies list from any response shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("vacancies", "data", "items", "jobs", "results"):
            if isinstance(data.get(key), list):
                return data[key]
        # fallback: first list value found
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def extract_pagination(data) -> tuple[str | None, str | None]:
    """Return (ignore_b64, ignore_hash) from a response dict, or (None, None)."""
    if not isinstance(data, dict):
        return None, None
    ignore = data.get("ignore") or data.get("next_ignore")
    ignore_hash = data.get("ignore_hash") or data.get("next_ignore_hash")
    return ignore, ignore_hash


def flatten(obj: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict using dot-notation keys."""
    result = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, key))
        else:
            result[key] = v
    return result


def fetch_all_vacancies() -> list[dict]:
    session = requests.Session()
    all_vacancies: list[dict] = []
    seen_ids: set = set()
    prev_page_ids: list = []   # only previous page's IDs go in `ignore`
    next_ignore: str | None = None
    next_hash: str | None = None
    page = 1

    while True:
        params: dict = {"hl": "az", "page": page}

        # Use server-provided ignore/hash if available; otherwise build from
        # the previous page's IDs only (keeps URL within server's size limit).
        if next_ignore and next_hash:
            params["ignore"] = next_ignore
            params["ignore_hash"] = next_hash
        elif prev_page_ids:
            params["ignore"] = encode_ids(prev_page_ids)

        print(f"Page {page:3d} | seen={len(seen_ids):4d} ...", end=" ", flush=True)

        resp = session.get(BASE_URL, headers=HEADERS, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        vacancies = extract_vacancies(data)
        next_ignore, next_hash = extract_pagination(data)

        new_count = 0
        prev_page_ids = []
        for v in vacancies:
            vid = v.get("id")
            if vid is not None:
                prev_page_ids.append(vid)
                if vid not in seen_ids:
                    all_vacancies.append(v)
                    seen_ids.add(vid)
                    new_count += 1

        print(f"fetched={len(vacancies):3d}  new={new_count:3d}  total={len(all_vacancies)}")

        if new_count == 0:
            print("No new vacancies — done.")
            break

        page += 1
        time.sleep(DELAY)

    return all_vacancies


def save_csv(vacancies: list[dict], path: Path) -> None:
    if not vacancies:
        print("Nothing to save.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    flat = [flatten(v) for v in vacancies]
    # Preserve column order from first appearance
    seen_keys: set = set()
    fieldnames: list = []
    for row in flat:
        for k in row:
            if k not in seen_keys:
                fieldnames.append(k)
                seen_keys.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)

    print(f"Saved {len(vacancies)} rows -> {path}")


def main() -> None:
    print(f"Target: {BASE_URL}")
    print(f"Output: {OUTPUT_FILE}\n")
    vacancies = fetch_all_vacancies()
    save_csv(vacancies, OUTPUT_FILE)


if __name__ == "__main__":
    main()
