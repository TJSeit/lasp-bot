# LASP Corpus Builder
# pip install -r requirements.txt
# python build_corpus.py

import os
import time
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure verbose logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

class LaspCorpusBuilder:
    def __init__(self, download_dir, github_token=None):
        self.download_dir = download_dir
        self.github_token = github_token
        self.visited_urls = set()
        
        # Sub-directories for organization
        self.dirs = {
            'pdf': os.path.join(download_dir, 'pdfs'),
            'html_text': os.path.join(download_dir, 'web_text'),
            'pds_data': os.path.join(download_dir, 'pds_metadata'),
            'github': os.path.join(download_dir, 'github_docs')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        # Robust session setup
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (LASP RAG Bot Builder)'})

    def is_valid_domain(self, url):
        return 'lasp.colorado.edu' in urlparse(url).netloc

    def save_file(self, content, filename, category, is_binary=False):
        filepath = os.path.join(self.dirs[category], filename)
        if os.path.exists(filepath):
            logging.debug(f"Skipping existing file: {filename}")
            return

        mode = 'wb' if is_binary else 'w'
        encoding = None if is_binary else 'utf-8'
        
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
        logging.info(f"[{category.upper()}] Saved -> {filename}")

    def scrape_web_and_pds(self, url, depth=0, max_depth=4):
        """Pillars 1 & 2: Scrape Mission Portals and NASA Data Labels"""
        if depth > max_depth or url in self.visited_urls or not self.is_valid_domain(url):
            return
            
        self.visited_urls.add(url)
        logging.info(f"[CRAWL] Depth {depth} | {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.warning(f"[ERROR] Failed to fetch {url}: {e}")
            return

        content_type = response.headers.get('Content-Type', '').lower()

        # Handle HTML -> Convert to clean text for the LLM
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract clean text from the webpage (ignoring scripts/styles)
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            text_content = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
            
            safe_name = urlparse(url).path.strip('/').replace('/', '_') or 'index'
            self.save_file(text_content, f"{safe_name}.txt", 'html_text')

            # Find links to follow or download
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                full_url = urljoin(url, href).split('#')[0]
                
                # Check for target file types
                lower_url = full_url.lower()
                if lower_url.endswith('.pdf'):
                    self.download_binary(full_url, 'pdf')
                elif lower_url.endswith(('.lbl', '.xml')):
                    self.download_binary(full_url, 'pds_data')
                elif self.is_valid_domain(full_url) and full_url not in self.visited_urls:
                    time.sleep(0.5) # Politeness delay
                    self.scrape_web_and_pds(full_url, depth + 1, max_depth)

    def download_binary(self, url, category):
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(self.dirs[category], filename)
        
        if os.path.exists(filepath):
            return
            
        logging.info(f"[DOWNLOADING] {filename} from {url}")
        try:
            r = self.session.get(url, stream=True, timeout=15)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"[{category.upper()}] Successfully downloaded -> {filename}")
        except Exception as e:
            logging.warning(f"[ERROR] Binary download failed for {url}: {e}")

    def fetch_github_repos(self, org="lasp"):
        """Pillar 3: Pull READMEs and Docs from LASP GitHub using API"""
        logging.info(f"[GITHUB] Fetching repositories for {org}...")
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
            
        repos_url = f"https://api.github.com/users/{org}/repos?per_page=100"
        try:
            r = requests.get(repos_url, headers=headers)
            r.raise_for_status()
            repos = r.json()
            
            for repo in repos:
                repo_name = repo['name']
                logging.info(f"[GITHUB] Checking repo: {repo_name}")
                
                # Attempt to get README
                readme_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/{repo['default_branch']}/README.md"
                readme_r = requests.get(readme_url)
                if readme_r.status_code == 200:
                    self.save_file(readme_r.text, f"{repo_name}_README.md", 'github')
                    
        except Exception as e:
            logging.warning(f"[ERROR] GitHub fetch failed. API limit reached? {e}")

if __name__ == "__main__":
    # Ensure you export your GitHub token in your terminal if you want to bypass the strict 60 req/hr rate limit
    # export GITHUB_TOKEN="your_personal_access_token"
    github_pat = os.environ.get("GITHUB_TOKEN")
    
    builder = LaspCorpusBuilder(download_dir="lasp_corpus", github_token=github_pat)
    
    # 1. & 2. Crawl Mission Portals & PDS Labels
    logging.info("=== STARTING PILLAR 1 & 2: WEB AND PDS CRAWL ===")
    builder.scrape_web_and_pds("https://lasp.colorado.edu/missions/", max_depth=2)
    
    # 3. Fetch Engineering GitHub Repos
    logging.info("=== STARTING PILLAR 3: GITHUB REPOS ===")
    builder.fetch_github_repos("lasp")
    
    # 4. Academic Papers (Note on ADS API)
    logging.info("=== PILLAR 4: ACADEMIC PAPERS ===")
    logging.info("[INFO] To ingest academic papers, you must query the NASA ADS API for abstracts.")
    logging.info("[INFO] Full PDFs require institutional access. Skipping automated download to prevent IP bans.")
    
    logging.info("=== CORPUS BUILD COMPLETE ===")