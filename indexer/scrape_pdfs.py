"""
scrape_pdfs.py — Lightweight PDF scraper for the LASP website.

Crawls the LASP missions portal (lasp.colorado.edu/missions/) up to a
configurable depth and downloads every PDF it finds into a local directory.

Usage:
    python scrape_pdfs.py

Note: For a more comprehensive corpus that also captures HTML text, PDS
metadata, and GitHub documentation, use build_corpus.py instead.
"""

import os
import time
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LaspPDFScraper:
    def __init__(self, start_url, download_dir, max_depth=2):
        self.start_url = start_url
        self.download_dir = download_dir
        self.max_depth = max_depth
        self.visited_urls = set()
        self.downloaded_pdfs = set()
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Configure robust session with connection pooling and retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1, # exponential backoff: 1s, 2s, 4s, 8s...
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Use a standard User-Agent to prevent basic blocks
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def is_valid_domain(self, url):
        """Ensure we only crawl within the lasp.colorado.edu domain."""
        domain = urlparse(url).netloc
        return 'lasp.colorado.edu' in domain

    def download_pdf(self, pdf_url):
        if pdf_url in self.downloaded_pdfs:
            return
            
        filename = os.path.basename(urlparse(pdf_url).path)
        if not filename.endswith('.pdf'):
            filename = f"document_{int(time.time())}.pdf"
            
        filepath = os.path.join(self.download_dir, filename)
        
        # Skip if already downloaded from a previous run
        if os.path.exists(filepath):
            logging.info(f"Already exists, skipping: {filename}")
            self.downloaded_pdfs.add(pdf_url)
            return
            
        try:
            logging.info(f"Downloading PDF: {filename}")
            response = self.session.get(pdf_url, stream=True, timeout=15)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.downloaded_pdfs.add(pdf_url)
            logging.info(f"Successfully saved: {filepath}")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {pdf_url}: {e}")

    def crawl(self, url, depth=0):
        if depth > self.max_depth or url in self.visited_urls or not self.is_valid_domain(url):
            return
            
        self.visited_urls.add(url)
        logging.info(f"Crawling (Depth {depth}): {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to access {url}: {e}")
            return
            
        # Only parse HTML pages
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            full_url = urljoin(url, href).split('#')[0] # normalize by removing fragments
            
            if full_url.lower().endswith('.pdf'):
                self.download_pdf(full_url)
            elif self.is_valid_domain(full_url) and full_url not in self.visited_urls:
                time.sleep(0.5) # Politeness delay
                self.crawl(full_url, depth + 1)

if __name__ == "__main__":
    # Define start node and output directory
    START_URL = "https://lasp.colorado.edu/missions/"
    DOWNLOAD_DIRECTORY = "lasp_pdfs"
    
    scraper = LaspPDFScraper(
        start_url=START_URL, 
        download_dir=DOWNLOAD_DIRECTORY, 
        max_depth=2 # Adjust this to 3 or 4 for a deeper crawl
    )
    
    logging.info("Starting LASP documentation scrape...")
    scraper.crawl(START_URL)
    logging.info(f"Scraping complete. Total PDFs gathered: {len(scraper.downloaded_pdfs)}")