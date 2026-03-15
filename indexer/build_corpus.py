# LASP Corpus Builder
# pip install -r requirements.txt
# python build_corpus.py

import os
import time
import json
import logging
import requests
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree
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
        self.state_file = os.path.join(download_dir, 'crawl_state.json')
        self.manifest_file = os.path.join(download_dir, 'source_manifest.json')
        self.visited_urls = self._load_visited_urls()
        self.source_manifest = self._load_source_manifest()
        
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
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (LASP RAG Bot Builder)'})

    def _normalize_url(self, url):
        parsed = urlparse(url)
        path = parsed.path or '/'
        if path != '/':
            path = path.rstrip('/')
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _load_visited_urls(self):
        if not os.path.exists(self.state_file):
            return set()
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            urls = state.get('visited_urls', [])
            if isinstance(urls, list):
                loaded = {self._normalize_url(u) for u in urls if isinstance(u, str)}
                logging.info(f"[STATE] Loaded {len(loaded)} previously visited URLs")
                return loaded
        except Exception as e:
            logging.warning(f"[STATE] Failed to load crawl state: {e}")
        return set()

    def _save_visited_urls(self):
        try:
            os.makedirs(self.download_dir, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({'visited_urls': sorted(self.visited_urls)}, f, indent=2)
        except Exception as e:
            logging.warning(f"[STATE] Failed to save crawl state: {e}")

    def _load_source_manifest(self):
        if not os.path.exists(self.manifest_file):
            return {}
        try:
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                logging.info(f"[STATE] Loaded {len(data)} source mappings")
                return data
        except Exception as e:
            logging.warning(f"[STATE] Failed to load source manifest: {e}")
        return {}

    def _save_source_manifest(self):
        try:
            os.makedirs(self.download_dir, exist_ok=True)
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(self.source_manifest, f, indent=2, sort_keys=True)
        except Exception as e:
            logging.warning(f"[STATE] Failed to save source manifest: {e}")

    def _record_source(self, category, filename, source_url):
        if not source_url:
            return
        relpath = f"{category}/{filename}".replace('\\\\', '/')
        normalized_url = self._normalize_url(source_url)
        if self.source_manifest.get(relpath) != normalized_url:
            self.source_manifest[relpath] = normalized_url
            self._save_source_manifest()

    def is_valid_domain(self, url):
        return 'lasp.colorado.edu' in urlparse(url).netloc

    def is_excluded_url(self, url):
        """Exclude sections we do not want in the corpus."""
        return urlparse(url).path.startswith('/people/')

    def save_file(self, content, filename, category, is_binary=False, source_url=None):
        filepath = os.path.join(self.dirs[category], filename)
        if os.path.exists(filepath):
            self._record_source(category, filename, source_url)
            logging.debug(f"Skipping existing file: {filename}")
            return

        mode = 'wb' if is_binary else 'w'
        encoding = None if is_binary else 'utf-8'
        
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
        self._record_source(category, filename, source_url)
        logging.info(f"[{category.upper()}] Saved -> {filename}")

    def scrape_web_and_pds(self, url, depth=0, max_depth=4):
        """Pillars 1 & 2: Scrape Mission Portals and NASA Data Labels"""
        normalized_url = self._normalize_url(url)
        if (
            depth > max_depth
            or normalized_url in self.visited_urls
            or not self.is_valid_domain(normalized_url)
            or self.is_excluded_url(normalized_url)
        ):
            return

        self.visited_urls.add(normalized_url)
        self._save_visited_urls()
        logging.info(f"[CRAWL] Depth {depth} | {normalized_url}")
        
        try:
            response = self.session.get(normalized_url, timeout=15, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.warning(f"[ERROR] Failed to fetch {normalized_url}: {e}")
            return

        content_type = response.headers.get('Content-Type', '').lower()

        # Catch "hidden" PDFs that were reached via a redirect or lacked a .pdf extension
        if 'application/pdf' in content_type:
            logging.info(f"[HIDDEN PDF FOUND] {normalized_url}")
            self.download_binary(normalized_url, 'pdf')
            return

        # Handle HTML -> Convert to clean text for the LLM
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract clean text from the webpage (ignoring scripts/styles)
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            text_content = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
            
            safe_name = urlparse(normalized_url).path.strip('/').replace('/', '_') or 'index'
            self.save_file(text_content, f"{safe_name}.txt", 'html_text', source_url=normalized_url)

            # Find links to follow or download
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                full_url = self._normalize_url(urljoin(normalized_url, href).split('#')[0])
                
                # Check the exact PATH, ignoring query parameters
                parsed_url = urlparse(full_url)
                lower_path = parsed_url.path.lower()
                
                if lower_path.endswith('.pdf'):
                    self.download_binary(full_url, 'pdf')
                elif lower_path.endswith(('.lbl', '.xml')):
                    self.download_binary(full_url, 'pds_data')
                elif (
                    self.is_valid_domain(full_url)
                    and not self.is_excluded_url(full_url)
                    and full_url not in self.visited_urls
                ):
                    time.sleep(0.5) # Politeness delay
                    self.scrape_web_and_pds(full_url, depth + 1, max_depth)

    def download_binary(self, url, category):
        # Safely extract the filename, ignoring query parameters like ?v=1
        parsed_path = urlparse(url).path
        filename = os.path.basename(parsed_path)

        # Fallback if the URL ends in a trailing slash or lacks a name
        if not filename or not filename.endswith(('.pdf', '.xml', '.lbl')):
            ext = 'pdf' if category == 'pdf' else 'xml'
            filename = f"document_{int(time.time())}.{ext}"

        filepath = os.path.join(self.dirs[category], filename)
        
        if os.path.exists(filepath):
            self._record_source(category, filename, url)
            return
            
        logging.info(f"[DOWNLOADING] {filename} from {url}")
        try:
            r = self.session.get(url, stream=True, timeout=15)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            self._record_source(category, filename, url)
            logging.info(f"[{category.upper()}] Successfully downloaded -> {filename}")
        except Exception as e:
            logging.warning(f"[ERROR] Binary download failed for {url}: {e}")

    def fetch_sitemap_urls(self, sitemap_url):
        """Fetch all page URLs from a sitemap or sitemap index file.

        Handles both standard sitemaps (<urlset>) and sitemap index files
        (<sitemapindex>) that reference child sitemaps.  Returns a flat list
        of every <loc> URL found.
        """
        urls = []
        logging.info(f"[SITEMAP] Fetching: {sitemap_url}")
        try:
            response = self.session.get(sitemap_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.warning(f"[SITEMAP] Failed to fetch {sitemap_url}: {e}")
            return urls

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as e:
            logging.warning(f"[SITEMAP] Failed to parse XML from {sitemap_url}: {e}")
            return urls

        # Strip XML namespace prefix for portable tag comparison.
        def _local(tag):
            return tag.split('}', 1)[-1] if '}' in tag else tag

        root_tag = _local(root.tag)

        if root_tag == 'sitemapindex':
            # Sitemap index: recurse into each child sitemap.
            for child in root:
                if _local(child.tag) == 'sitemap':
                    for elem in child:
                        if _local(elem.tag) == 'loc' and elem.text:
                            urls.extend(self.fetch_sitemap_urls(elem.text.strip()))
        elif root_tag == 'urlset':
            # Regular sitemap: collect all <url><loc> entries.
            for child in root:
                if _local(child.tag) == 'url':
                    for elem in child:
                        if _local(elem.tag) == 'loc' and elem.text:
                            urls.append(elem.text.strip())
        else:
            logging.warning(f"[SITEMAP] Unexpected root element <{root_tag}> in {sitemap_url}")

        logging.info(f"[SITEMAP] Found {len(urls)} URLs in {sitemap_url}")
        return urls

    def crawl_from_sitemap(self, sitemap_url):
        """Discover all LASP pages via sitemap then scrape each one.

        This is more efficient than pure recursive link-following because
        the sitemap already enumerates every published URL.  Only one level
        of additional link-following (max_depth=1) is performed per page so
        that any PDFs or PDS labels linked from a page are still captured.

        Returns the number of new URLs queued for scraping (0 if the sitemap
        was empty or unreachable, so callers can fall back to the old crawl).
        """
        all_urls = self.fetch_sitemap_urls(sitemap_url)
        valid_urls = [
            url for url in all_urls
            if self.is_valid_domain(url) and not self.is_excluded_url(url)
        ]
        logging.info(
            f"[SITEMAP] {len(valid_urls)} valid LASP URLs to crawl "
            f"(out of {len(all_urls)} total)"
        )
        queued = 0
        for url in valid_urls:
            normalized = self._normalize_url(url)
            if normalized not in self.visited_urls:
                time.sleep(0.5)
                self.scrape_web_and_pds(normalized, depth=0, max_depth=1)
                queued += 1
        return queued

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

                readme_filename = f"{repo_name}_README.md"
                readme_path = os.path.join(self.dirs['github'], readme_filename)
                if os.path.exists(readme_path):
                    logging.debug(f"[GITHUB] Skipping existing README: {readme_filename}")
                    continue
                
                # Attempt to get README
                readme_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/{repo['default_branch']}/README.md"
                readme_r = requests.get(readme_url)
                if readme_r.status_code == 200:
                    self.save_file(readme_r.text, readme_filename, 'github', source_url=readme_url)
                    
        except Exception as e:
            logging.warning(f"[ERROR] GitHub fetch failed. API limit reached? {e}")

if __name__ == "__main__":
    # Ensure you export your GitHub token in your terminal if you want to bypass the strict 60 req/hr rate limit
    # export GITHUB_TOKEN="your_personal_access_token"
    github_pat = os.environ.get("GITHUB_TOKEN")
    
    builder = LaspCorpusBuilder(download_dir="lasp_corpus", github_token=github_pat)
    
    # 1. & 2. Crawl Mission Portals & PDS Labels via sitemap, falling back to
    #         the recursive link-following crawl if the sitemap is unavailable.
    logging.info("=== STARTING PILLAR 1 & 2: WEB AND PDS CRAWL ===")
    queued = builder.crawl_from_sitemap("https://lasp.colorado.edu/sitemap.xml")
    if queued == 0:
        logging.warning(
            "[SITEMAP] No URLs found via sitemap — falling back to recursive crawl"
        )
        builder.scrape_web_and_pds("https://lasp.colorado.edu/missions/", max_depth=4)
    
    # 3. Fetch Engineering GitHub Repos
    logging.info("=== STARTING PILLAR 3: GITHUB REPOS ===")
    builder.fetch_github_repos("lasp")
    
    # 4. Academic Papers (Note on ADS API)
    logging.info("=== PILLAR 4: ACADEMIC PAPERS ===")
    logging.info("[INFO] To ingest academic papers, you must query the NASA ADS API for abstracts.")
    logging.info("[INFO] Full PDFs require institutional access. Skipping automated download to prevent IP bans.")
    
    logging.info("=== CORPUS BUILD COMPLETE ===")