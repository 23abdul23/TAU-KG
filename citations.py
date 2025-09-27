#!/usr/bin/env python3
"""
PubMed Citation Fetcher
Retrieves scientific citations from PubMed based on a query, using .env config
and improved query optimization.
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import time
import requests
import xml.etree.ElementTree as ET
import re
import json
import logging
import os # Import os for environment variables
import random
import httpx  # Required for OpenAI proxy configuration
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import quote_plus
from datetime import datetime

# Import constants needed for entity extraction and query building
from src.constants import (
    GENE_PATTERN, PROTEIN_PATTERN, DISEASE_PATTERN, PATHWAY_PATTERN,
    COMMON_NON_GENES, QUERY_TYPE_INDICATORS
)

# Use centralized logging configuration
from src.logger_config import get_logger
logger = get_logger(__name__)

# Import configuration
from src.utils import load_config
CONFIG = load_config("config.json")

# Don't remove proxy environment variables - let them be used if proxy is enabled

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library loaded for enhanced entity extraction")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not found. Will use regex-based extraction only.")

# Get configurable parameters from config.json
MAX_RETRIES = CONFIG["pubmed"]["max_retries"]
# Check if specific timeout values exist, otherwise fall back to generic timeout
TIMEOUT_SHORT = CONFIG["pubmed"].get("timeout_short", CONFIG["pubmed"].get("timeout", 10))
TIMEOUT_LONG = CONFIG["pubmed"].get("timeout_long", CONFIG["pubmed"].get("timeout", 15))

from dotenv import load_dotenv
load_dotenv()
import sys
import time
import requests
import xml.etree.ElementTree as ET
import re
import json
import logging
import os # Import os for environment variables
import random
import httpx  # Required for OpenAI proxy configuration
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import quote_plus
from datetime import datetime

# Import constants needed for entity extraction and query building
from src.constants import (
    GENE_PATTERN, PROTEIN_PATTERN, DISEASE_PATTERN, PATHWAY_PATTERN,
    COMMON_NON_GENES, QUERY_TYPE_INDICATORS
)

# Import configuration
from src.utils import load_config
CONFIG = load_config("config.json")


try:
    import openai
    OPENAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenAI library loaded for enhanced entity extraction")
except ImportError:
    OPENAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenAI library not found. Will use regex-based extraction only.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("citations.log"),
        logging.StreamHandler()
    ]
)

#!/usr/bin/env python3
"""
PubMed Citation Fetcher
Retrieves scientific citations from PubMed based on a query, using .env config
and improved query optimization.
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import time
import requests
import xml.etree.ElementTree as ET
import re
import json
import logging
import os # Import os for environment variables
import random
import httpx  # Required for OpenAI proxy configuration
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import quote_plus
from datetime import datetime

# Import constants needed for entity extraction and query building
from src.constants import (
    GENE_PATTERN, PROTEIN_PATTERN, DISEASE_PATTERN, PATHWAY_PATTERN,
    COMMON_NON_GENES, QUERY_TYPE_INDICATORS
)

# Import configuration
from src.utils import load_config
CONFIG = load_config("config.json")


try:
    import openai
    OPENAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenAI library loaded for enhanced entity extraction")
except ImportError:
    OPENAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenAI library not found. Will use regex-based extraction only.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("citations.log"),
        logging.StreamHandler()
    ]
)

# Get configurable parameters from config.json
MAX_RETRIES = CONFIG["pubmed"]["max_retries"]
# Check if specific timeout values exist, otherwise fall back to generic timeout
TIMEOUT_SHORT = CONFIG["pubmed"].get("timeout_short", CONFIG["pubmed"].get("timeout", 10))
TIMEOUT_LONG = CONFIG["pubmed"].get("timeout_long", CONFIG["pubmed"].get("timeout", 15))

# --- Proxy Configuration ---
# Use proxy settings from config.json if available
PROXY_CONFIG = {}
# Check if network settings exist in config
if "network" in CONFIG and CONFIG.get("network", {}).get("use_proxy", False):
    http_proxy = CONFIG["network"].get("http_proxy", "")
    https_proxy = CONFIG["network"].get("https_proxy", "")
    
    if http_proxy and http_proxy.strip():
        PROXY_CONFIG['http'] = http_proxy
        # Log proxy (but hide passwords)
        log_http_proxy = http_proxy.split('@')[-1] if '@' in http_proxy else http_proxy
        logger.info(f"Using HTTP Proxy: {log_http_proxy}")
        
    if https_proxy and https_proxy.strip():
        PROXY_CONFIG['https'] = https_proxy
        log_https_proxy = https_proxy.split('@')[-1] if '@' in https_proxy else https_proxy
        logger.info(f"Using HTTPS Proxy: {log_https_proxy}")
        
    logger.info("Proxy enabled via config.json settings")
else:
    # If proxy is disabled in config or network config doesn't exist, don't use proxy
    logger.info("Proxy not configured or disabled in config.json settings")

# --- NCBI API Key ---
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
if NCBI_API_KEY:
    logger.info("NCBI API Key found in environment.")
else:
    logger.warning("NCBI_API_KEY not found in environment. Using unauthenticated requests (rate limits apply).")

# --- OpenAI Client Setup ---
OPENAI_CLIENT = None
if OPENAI_AVAILABLE:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            # Configure proxy for OpenAI if needed
            http_proxy = None
            if "network" in CONFIG and CONFIG.get("network", {}).get("use_proxy", False):
                http_proxy = CONFIG["network"].get("http_proxy", "")
                
            # Initialize OpenAI client with proper proxy configuration
            if http_proxy and http_proxy.strip():
                try:
                    # Create a properly configured httpx client with the proxy
                    # Fix the proxy URL encoding issues with @ symbol in password
                    # Create httpx transport configuration
                    logger.info("Initializing OpenAI client with proxy configuration")
                    
                    # Create httpx client with proper proxy settings
                    transport = httpx.HTTPTransport(proxy=httpx.URL(http_proxy))
                    http_client = httpx.Client(transport=transport)
                    OPENAI_CLIENT = openai.OpenAI(
                        api_key=openai_api_key,
                        http_client=http_client
                    )
                    logger.info("Successfully initialized OpenAI client with proxy configuration")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client with proxy: {e}")
                    logger.info("Falling back to standard OpenAI client without proxy")
                    OPENAI_CLIENT = openai.OpenAI(api_key=openai_api_key)
            else:
                logger.info("Initializing OpenAI client without proxy configuration")
                OPENAI_CLIENT = openai.OpenAI(api_key=openai_api_key)
                
            logger.info("OpenAI client initialized for entity extraction.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            OPENAI_CLIENT = None
    else:
        logger.warning("OPENAI_API_KEY not set. Enhanced entity extraction disabled.")


class Citation:
    """Represents a scientific citation."""
    def __init__(self, title: str, authors: str = "", journal: str = "",
                 pmid: str = "", year: str = "", doi: str = "",
                 is_review: bool = False):
        self.title = title
        self.authors = authors
        self.journal = journal
        self.pmid = pmid
        self.year = year
        self.doi = doi
        self.is_review = is_review
        self.relevance_score = 0 # Added for sorting

    def __str__(self) -> str:
        """String representation for terminal display."""
        return f"{self.title}\n{self.authors}\n{self.journal} ({self.year})\nPMID: {self.pmid}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary format."""
        # Generate Google search URL for the paper title
        url_encoded_title = quote_plus(self.title)
        google_link = f"https://www.google.com/search?q={url_encoded_title}&btnI=I%27m%20Feeling%20Lucky"

        return {
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "pmid": self.pmid,
            "doi": self.doi,
            "is_review": self.is_review,
            "url": google_link
        }

def extract_entities_with_llm(text: str) -> Dict[str, List[str]]:
    if not OPENAI_CLIENT:
        logger.warning("OpenAI client not available. Falling back to regex extraction.")
        return extract_entities_from_text(text)
    
    try:
        logger.info("Extracting entities using OpenAI API")
        
        # Improved prompt for precise entity extraction
        system_prompt = """
        You are a biomedical NLP specialist with expertise in extracting entities from scientific text.
        
        Extract biomedical entities from the provided text, with a focus on completeness and precision.
        Identify entities in the following categories:
        
        1. Genes: Return ONLY official gene symbols using standard nomenclature (e.g., SNCA, LRRK2, PARK7)
           - Include ALL gene symbols mentioned or implied in the text
           - Use official HGNC symbols when possible
           - Include gene families when specifically mentioned (e.g., HOX genes)
        
        2. Proteins: Return protein names (e.g., alpha-synuclein, parkin, DJ-1)
           - Include protein complexes and enzymes
           - Use standard nomenclature without qualifiers
        
        3. Diseases: Return specific disease names (e.g., Parkinson's disease, Lewy body dementia)
           - Include specific disease subtypes when mentioned
           - Include related conditions that are clinically relevant
        
        4. Pathways: Return biological pathway names (e.g., ubiquitin-proteasome pathway, autophagy)
           - Include signaling cascades and cellular processes
           - Be comprehensive about involved pathways
        
        5. Keywords: Return important scientific terms not in other categories
           - Include research methods, anatomical terms, and key concepts
           - Focus on domain-specific terminology
        
        Format your response as a JSON object with these categories as keys and arrays of strings as values.
        Return EMPTY arrays for categories with no entities.
        Do NOT include explanations or notes, only the JSON.
        """
        
        user_prompt = f"Extract biomedical entities from this text: {text}"
        
        # Get model and parameters from config with fallbacks
        citation_config = CONFIG.get("citation_extraction", {})
        entity_model = citation_config.get("entity_extraction_model", "gpt-4o")
        extraction_temperature = citation_config.get("extraction_temperature", 0.1)
        extraction_max_tokens = citation_config.get("extraction_max_tokens", 500)
        
        # Check if the model supports JSON response format
        json_supported_models = ["gpt-4-1106-preview", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo-1106"]
        use_json_format = any(model in entity_model for model in json_supported_models)
        
        if use_json_format:
            response = OPENAI_CLIENT.chat.completions.create(
                model=entity_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=extraction_temperature,
                max_tokens=extraction_max_tokens
            )
        else:
            # For models that don't support JSON format, modify the prompt
            modified_system_prompt = system_prompt + "\n\nIMPORTANT: Return your response as valid JSON only, with no additional text or explanations."
            response = OPENAI_CLIENT.chat.completions.create(
                model=entity_model,
                messages=[
                    {"role": "system", "content": modified_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=extraction_temperature,
                max_tokens=extraction_max_tokens
            )
        
        # Parse the response JSON
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all expected keys exist
        expected_keys = ["genes", "proteins", "diseases", "pathways", "keywords"]
        for key in expected_keys:
            if key not in result:
                result[key] = []
                
        # Log extraction results
        total_entities = sum(len(entities) for entities in result.values())
        logger.info(f"Extracted {total_entities} entities using LLM: {', '.join(f'{k}:{len(v)}' for k, v in result.items() if v)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error using OpenAI for entity extraction: {e}", exc_info=True)
        logger.info("Falling back to regex-based entity extraction")
        return extract_entities_from_text(text)

def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extract biomedical entities from text using regex patterns.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary of entity types and their values
    """
    entities = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
        "keywords": [] # Add general keywords
    }
    text_lower = text.lower()

    # Extract genes (e.g., BRCA1, TP53)
    # Make pattern case-insensitive for matching but capture original case
    genes = set(re.findall(GENE_PATTERN, text))
    genes = {gene for gene in genes if gene.upper() not in COMMON_NON_GENES and len(gene) >= 3 and not gene.isdigit()}
    entities["genes"] = list(genes)

    # Extract proteins - capture original case
    proteins = set(re.findall(PROTEIN_PATTERN, text)) # Keep case-sensitive for proper names
    entities["proteins"] = list(proteins)

    # Extract diseases - capture original case
    diseases = set(re.findall(DISEASE_PATTERN, text))
    entities["diseases"] = list(diseases)

    # Extract pathways - capture original case
    pathways = set(re.findall(PATHWAY_PATTERN, text))
    entities["pathways"] = list(pathways)

    # Extract general keywords if specific entities are scarce
    if not any([entities["genes"], entities["diseases"], entities["proteins"], entities["pathways"]]):
         for type_key, indicators in QUERY_TYPE_INDICATORS.items():
              if any(indicator in text_lower for indicator in indicators):
                   entities["keywords"].extend(indicators) # Add the indicator words
         entities["keywords"] = list(set(entities["keywords"])) # Deduplicate

    return entities


def fetch_with_retry(url: str, params: Dict[str, Any],
                    timeout: int = TIMEOUT_SHORT,
                    max_retries: int = MAX_RETRIES) -> requests.Response:
    """
    Make a request with retry logic, using global proxy config and API key.
    """
    # Add API key if available and not already present
    if NCBI_API_KEY and 'api_key' not in params:
        params['api_key'] = NCBI_API_KEY

    # Use proxy settings from environment variables if proxy is enabled
    use_proxy = CONFIG.get("network", {}).get("use_proxy", False)
    proxies = None
    
    if use_proxy:
        # Get proxy from environment variables
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            
            # Log proxy usage (hide credentials)
            should_log = CONFIG.get("network", {}).get("log_network_requests", False)
            if should_log:
                safe_proxies = {}
                for key, proxy_url in proxies.items():
                    # Hide credentials in log
                    if '@' in proxy_url:
                        safe_proxy = proxy_url.split('@')[-1]
                        safe_proxies[key] = f"***@{safe_proxy}"
                    else:
                        safe_proxies[key] = proxy_url
                logger.info(f"Using proxy configuration: {safe_proxies}")
    
    for attempt in range(max_retries):
        try:
            # Prepare params for logging (mask api key)
            log_params = {k: v for k, v in params.items() if k != 'api_key'}
            logger.debug(f"Request attempt {attempt+1}/{max_retries} to {url.split('?')[0]} with params: {log_params}")

            # Use proxy configuration from environment variables
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                proxies=proxies
            )
            response.raise_for_status()
            # Add delay based on whether API key is used
            delay = 0.11 if NCBI_API_KEY else 0.35 # ~10/sec with key, ~3/sec without
            time.sleep(delay)
            return response
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Retrying in {backoff:.2f} seconds...")
                time.sleep(backoff)
            else:
                logger.error("Max retries reached for Timeout.")
                raise
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None
            logger.warning(f"Request error on attempt {attempt+1}/{max_retries} (Status: {status_code}): {e}")
            if attempt < max_retries - 1:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                if status_code == 429:
                    backoff = max(backoff, 5) # Wait longer for rate limit
                    logger.warning("Rate limit likely hit.")
                logger.warning(f"Retrying in {backoff:.2f} seconds...")
                time.sleep(backoff)
            else:
                logger.error("Max retries reached for RequestException.")
                raise

def parse_pubmed_xml(xml_text: str) -> List[Citation]:
    """ Parse PubMed XML response to extract citations """
    citations = []
    try:
        # Replace problematic XML entities if necessary before parsing
        # xml_text = xml_text.replace('&', '&') # Example, be careful with this
        root = ET.fromstring(xml_text)
        articles = root.findall(".//PubmedArticle")
        for article in articles:
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            title_elem = article.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else "Untitled"
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""
            if not year:
                medline_date_elem = article.find(".//MedlineDate")
                if medline_date_elem is not None and medline_date_elem.text:
                    match = re.search(r'^\d{4}', medline_date_elem.text) # Extract first 4 digits
                    if match: year = match.group(0)
            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else ""
            pub_types = article.findall(".//PublicationType")
            is_review = any("Review" in pt.text for pt in pub_types if pt is not None and pt.text is not None)
            authors = []
            author_elems = article.findall(".//Author")
            for author_elem in author_elems[:5]: # Limit authors
                last_name = author_elem.find("LastName")
                initials = author_elem.find("Initials")
                author_parts = []
                if last_name is not None and last_name.text: author_parts.append(last_name.text)
                if initials is not None and initials.text: author_parts.append(initials.text)
                if author_parts: authors.append(" ".join(author_parts))
            author_str = ", ".join(authors)
            if len(author_elems) > 5: author_str += ", et al."
            citations.append(Citation(title=title, authors=author_str, journal=journal, pmid=pmid, year=year, doi=doi, is_review=is_review))
        return citations
    except ET.ParseError as e:
        logger.error(f"Error parsing XML: {e}")
        # Optionally log problematic XML snippet
        # logger.debug(f"Problematic XML snippet: {xml_text[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing PubMed data: {e}", exc_info=True)
        return []

def json_fallback_search(query: str, max_results: int = 5) -> List[Citation]:
    """ Try to retrieve PubMed data using JSON esearch -> XML efetch """
    logger.info("Trying JSON esearch -> XML efetch...")
    try:
        params = { "db": "pubmed", "term": query, "retmax": max_results * 2, "sort": "relevance", "retmode": "json" }
        response = fetch_with_retry("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params, timeout=TIMEOUT_SHORT)
        data = response.json()
        if "esearchresult" not in data or "idlist" not in data["esearchresult"]:
            logger.warning("No results found in JSON esearch")
            return []
        id_list = data["esearchresult"]["idlist"]
        if not id_list:
            logger.warning("Empty ID list returned in JSON esearch")
            return []
        logger.info(f"Found {len(id_list)} PMIDs via JSON esearch, fetching details...")
        fetch_params = { "db": "pubmed", "id": ",".join(id_list), "retmode": "xml", "rettype": "abstract" }
        fetch_res = fetch_with_retry("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", fetch_params, timeout=TIMEOUT_LONG)
        citations = parse_pubmed_xml(fetch_res.text)
        return citations
    except Exception as e:
        logger.error(f"JSON esearch -> XML efetch failed: {e}", exc_info=True)
        return []

def extract_genes_from_titles(citations: List[Citation]) -> List[str]:
    """ Extract potential gene names from citation titles """
    extracted_genes = set() # Use set for efficiency
    for citation in citations:
        possible_genes = re.findall(GENE_PATTERN, citation.title)
        genes = { gene for gene in possible_genes if gene.upper() not in COMMON_NON_GENES and len(gene) > 2 and not gene.isdigit() }
        extracted_genes.update(genes)
    return sorted(list(extracted_genes))

def generate_improved_query_with_llm(question: str, entities: Dict[str, List[str]]) -> Tuple[str, bool]:
    """
    Use OpenAI API to generate an optimized PubMed query based on the original question and extracted entities.
    
    Args:
        question: The original user question
        entities: Dictionary of extracted biomedical entities
        
    Returns:
        Tuple of (optimized query string, success flag)
    """
    if not OPENAI_CLIENT:
        return None, False
    
    try:
        # Convert entities to a readable format for the prompt
        entities_str = ""
        for category, items in entities.items():
            if items:
                entities_str += f"{category.upper()}: {', '.join(items)}\n"
        
        system_prompt = """
        You are a biomedical search expert with deep expertise in PubMed and MEDLINE. Your task is to construct an optimal PubMed search query that will retrieve the most relevant scientific literature.

        Create a sophisticated search strategy using:

        1. MeSH Terms - Always use appropriate MeSH Terms with this format: "Term"[MeSH Terms]
           - Include MeSH explosion where appropriate
           - Use MeSH Subheadings when helpful (e.g., "Parkinson Disease/genetics"[MeSH])
        
        2. Precise Field Tags:
           - [Title/Abstract] for keyword searching in those fields
           - [Author] for author names
           - [Gene Name] or [Substance Name] for specific genes or proteins
           - [Journal] for specific journals
           - [Publication Type] for article types (review, clinical trial, etc.)
        
        3. Boolean Logic:
           - Use (parentheses) to properly nest operations
           - Connect related concepts with OR
           - Connect different concepts with AND
           - Use NOT sparingly and only when necessary
        
        4. Advanced Techniques:
           - Use wildcards (*) to capture variations (e.g., gene* for gene, genes, genetic)
           - Include appropriate synonyms for key concepts
           - Use proximity operators like NEAR or ADJ when beneficial
           - Consider recency by adding date filters if needed
        
        5. Structure your query to balance:
           - Precision: Finding specific, relevant articles
           - Recall: Capturing the breadth of relevant literature
           - Clinical relevance: Prioritizing clinically meaningful results
        
        Produce ONLY the final PubMed query string with no commentary or explanation.
        """
        
        user_prompt = f"""
        Generate an optimized PubMed search query for this question:
        
        QUESTION: {question}
        
        EXTRACTED ENTITIES:
        {entities_str}
        
        Return just the search query string using proper PubMed syntax.
        """
        
        # Get model and parameters from config
        query_model = CONFIG["citation_extraction"]["query_generation_model"]
        generation_temperature = CONFIG["citation_extraction"]["generation_temperature"]
        generation_max_tokens = CONFIG["citation_extraction"]["generation_max_tokens"]

        # Generate the query
        response = OPENAI_CLIENT.chat.completions.create(
            model=query_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=generation_temperature,
            max_tokens=generation_max_tokens
        )
        
        # Get the optimized query and clean it up
        optimized_query = response.choices[0].message.content.strip()
        # Remove any surrounding quotes or explanation that might have slipped through
        optimized_query = re.sub(r'^["\'](.*)["\']$', r'\1', optimized_query)
        optimized_query = re.sub(r'^```.*\n(.*)\n```$', r'\1', optimized_query, flags=re.DOTALL)
        
        logger.info(f"Generated LLM-optimized PubMed query: {optimized_query}")
        return optimized_query, True
        
    except Exception as e:
        logger.error(f"Error generating query with LLM: {e}", exc_info=True)
        return None, False

def optimize_query(query: str, entities: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Optimize a query for better PubMed search results using entities and MeSH terms.

    Args:
        query: Original query
        entities: Optional extracted entities (genes, diseases, etc.)

    Returns:
        Optimized query string
    """
    # Step 1: Extract or use provided entities
    if entities is None:
        # Try LLM-based extraction if available and enabled in config
        use_llm = CONFIG.get("citation_extraction", {}).get("use_llm_extraction", False)
        if OPENAI_CLIENT and use_llm:
            entities = extract_entities_with_llm(query)
        else:
            entities = extract_entities_from_text(query)
    
    # Step 2: Try to generate an optimized query using LLM if available and enabled
    use_llm_query = CONFIG.get("citation_extraction", {}).get("use_llm_query_generation", False)
    if OPENAI_CLIENT and use_llm_query:
        llm_query, success = generate_improved_query_with_llm(query, entities)
        if success and llm_query:
            return llm_query
    
    # Step 3: Fall back to rule-based optimization if LLM fails
    logger.info("Using rule-based query optimization")
    
    # --- Core Topic Extraction ---
    disease_terms = entities.get("diseases", [])
    gene_terms = entities.get("genes", [])
    protein_terms = entities.get("proteins", [])
    pathway_terms = entities.get("pathways", [])
    keyword_terms = entities.get("keywords", []) # General keywords if others fail

    # --- Intent Keywords Extraction ---
    query_lower = query.lower()
    is_gene_focused = any(kw in query_lower for kw in QUERY_TYPE_INDICATORS["gene"])
    is_pathway_focused = any(kw in query_lower for kw in QUERY_TYPE_INDICATORS["pathway"])
    is_protein_focused = any(kw in query_lower for kw in QUERY_TYPE_INDICATORS["protein"])
    is_disease_focused = any(kw in query_lower for kw in QUERY_TYPE_INDICATORS["disease"]) or disease_terms # Assume disease focus if disease mentioned

    # --- Build Structured Query ---
    query_parts = []
    topic_added = False

    # 1. Add Disease Context (Primary Focus if present)
    if disease_terms:
        mesh_disease = [f'"{d}"[MeSH Terms]' for d in disease_terms]
        tiab_disease = [f'"{d}"[Title/Abstract]' for d in disease_terms]
        query_parts.append(f"({' OR '.join(mesh_disease + tiab_disease)})")
        topic_added = True

    # 2. Add Specific Focus (Genes, Pathways, etc.) - only if NOT the primary topic already added
    if is_gene_focused:
        genetic_mesh = ["Genes, Medical[MeSH Terms]", "Genetic Predisposition to Disease[MeSH Terms]", "Mutation[MeSH Terms]"]
        genetic_tiab = ["gene", "genetic*", "mutation*", "variant*", "allele*", "polymorphism*"] # Use wildcards
        genetic_tiab_phrased = [f'"{kw}"[Title/Abstract]' for kw in genetic_tiab]
        query_parts.append(f"({' OR '.join(genetic_mesh + genetic_tiab_phrased)})")
        if gene_terms: # Also add specific genes if mentioned
             gene_names = [f'"{g}"[Gene/Protein Name]' for g in gene_terms] # Use broader tag
             query_parts.append(f"({' OR '.join(gene_names)})")
        topic_added = True # Assume if gene focused, topic is covered

    elif is_pathway_focused and not topic_added: # Avoid adding pathway if disease already added unless pathway is main focus
        pathway_mesh = ["Signal Transduction[MeSH Terms]", "Metabolic Pathways[MeSH Terms]"]
        pathway_tiab = ["pathway", "signaling", "signalling", "cascade", "metabolic process"]
        pathway_tiab_phrased = [f'"{kw}"[Title/Abstract]' for kw in pathway_tiab]
        query_parts.append(f"({' OR '.join(pathway_mesh + pathway_tiab_phrased)})")
        if pathway_terms:
             path_names = [f'"{p}"[Title/Abstract]' for p in pathway_terms]
             query_parts.append(f"({' OR '.join(path_names)})")
        topic_added = True

    elif is_protein_focused and not topic_added:
        protein_mesh = ["Proteins[MeSH Terms]"] # Very broad
        protein_tiab = ["protein", "receptor", "enzyme", "kinase", "antibody"]
        protein_tiab_phrased = [f'"{kw}"[Title/Abstract]' for kw in protein_tiab]
        query_parts.append(f"({' OR '.join(protein_mesh + protein_tiab_phrased)})")
        if protein_terms:
             prot_names = [f'"{p}"[Title/Abstract]' for p in protein_terms] # Gene/Protein Name tag less reliable here
             query_parts.append(f"({' OR '.join(prot_names)})")
        topic_added = True

    # 3. Add remaining specific entities if not covered by focus
    # Example: If query is about disease X and gene Y, disease was added in step 1, gene focus in step 2.
    # If query is about disease X and pathway Y, disease added in step 1, pathway focus might be skipped if disease was topic. Add pathway here.
    if pathway_terms and not is_pathway_focused and topic_added:
         path_names = [f'"{p}"[Title/Abstract]' for p in pathway_terms]
         query_parts.append(f"({' OR '.join(path_names)})")
    if protein_terms and not is_protein_focused and topic_added:
         prot_names = [f'"{p}"[Title/Abstract]' for p in protein_terms]
         query_parts.append(f"({' OR '.join(prot_names)})")
    # Add gene terms if disease/pathway/protein was the main focus but genes were also mentioned
    if gene_terms and not is_gene_focused and topic_added:
         gene_names = [f'"{g}"[Gene/Protein Name]' for g in gene_terms]
         query_parts.append(f"({' OR '.join(gene_names)})")


    # --- Combine Query Parts ---
    if query_parts:
        # Combine parts with AND
        optimized = " AND ".join(query_parts)
    else:
        # Fallback if no specific parts identified: quote the original query
        logger.warning(f"Could not build structured query for: '{query}'. Using original quoted query.")
        # Quote only if it contains spaces and isn't already quoted
        if ' ' in query and not (query.startswith('"') and query.endswith('"')):
             optimized = f'"{query}"'
        else:
             optimized = query # Use raw query if single word or already quoted

    return optimized


def fetch_pubmed_citations(query: str, max_citations: int = 5,
                          prioritize_reviews: bool = True,
                          max_age_years: int = 5) -> List[Citation]:
    """ Main function to fetch PubMed citations with improved query optimization """
    caller_info = ""
    try:
        import inspect
        frame = inspect.currentframe().f_back
        if frame:
            caller_module = inspect.getmodule(frame)
            if caller_module:
                caller_info = f" (called from {caller_module.__name__})"
    except Exception:
        pass
    
    logger.info(f"Searching PubMed for user query: '{query}'{caller_info}")
    logger.info(f"Citation parameters: max_citations={max_citations}, prioritize_reviews={prioritize_reviews}, max_age_years={max_age_years}")

    # Check for required environment variables and API keys
    logger.info(f"Environment check - NCBI_API_KEY: {'Available' if NCBI_API_KEY else 'Not available'}")
    logger.info(f"Environment check - OPENAI_CLIENT: {'Available' if OPENAI_CLIENT else 'Not available'}")
    
    # Extract entities using appropriate method based on config
    # Check if citation_extraction is in config, use defaults if not
    use_llm = CONFIG.get("citation_extraction", {}).get("use_llm_extraction", False)
    logger.info(f"Entity extraction method: {'LLM-based' if OPENAI_CLIENT and use_llm else 'Regex-based'}")
    
    try:
        if OPENAI_CLIENT and use_llm:
            entities = extract_entities_with_llm(query)
        else:
            entities = extract_entities_from_text(query)
        
        # Log extracted entities
        entity_summary = ", ".join([f"{k}: {len(v)}" for k, v in entities.items() if v])
        logger.info(f"Extracted entities - {entity_summary}")
        
        # Optimize query with extracted entities
        optimized_query = optimize_query(query, entities)
        logger.info(f"Optimized query: {optimized_query}")
    except Exception as e:
        logger.error(f"Error during entity extraction or query optimization: {e}", exc_info=True)
        # Fall back to a basic query if entity extraction fails
        optimized_query = f'"{query}"'
        logger.info(f"Falling back to basic query: {optimized_query}")

    # Add filters for scientific quality if requested
    query_filters = []
    if prioritize_reviews:
        query_filters.append("Review[Publication Type]")

    if max_age_years > 0:
        current_year = datetime.now().year
        min_year = current_year - max_age_years
        # Ensure correct date format YYYY/MM/DD or YYYY
        query_filters.append(f"({min_year}/01/01[PDAT] : {current_year}/12/31[PDAT])") # More precise date range

    # Combine base query and filters
    if query_filters:
        # Ensure the optimized query is properly bracketed if it contains AND/OR
        # Check needed because optimized_query might be simple quoted string
        if " AND " in optimized_query or " OR " in optimized_query or optimized_query.startswith("("):
             final_query = f"({optimized_query}) AND ({' AND '.join(query_filters)})"
        else:
             # If optimized query is simple (e.g., single term or quoted phrase), no extra parens needed
             final_query = f"{optimized_query} AND ({' AND '.join(query_filters)})"
    else:
        final_query = optimized_query

    logger.info(f"Constructed PubMed query: {final_query}")

    start_time = time.time()
    citations = []

    try:
        # Try JSON esearch -> XML efetch first
        logger.info("Attempting primary search method (JSON esearch -> XML efetch)...")
        citations = json_fallback_search(final_query, max_citations)
        
        if citations:
            logger.info(f"Primary search method successful - found {len(citations)} citations")
        else:
            logger.warning("Primary method yielded no results, trying XML esearch -> efetch...")
            # Log network information to diagnose connection issues
            proxy_info = "PROXY_CONFIG: " + str(PROXY_CONFIG)
            logger.info(f"Network settings - {proxy_info}")
            
            try:
                params = { "db": "pubmed", "term": final_query, "retmax": max_citations * 2, "sort": "relevance", "retmode": "xml" }
                search_res = fetch_with_retry("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params, timeout=TIMEOUT_SHORT)
                logger.info(f"XML esearch response status code: {search_res.status_code}")
                
                if search_res.text and len(search_res.text) > 0:
                    logger.info(f"XML esearch response length: {len(search_res.text)} bytes")
                else:
                    logger.error("XML esearch returned empty response")
                    raise Exception("Empty response from PubMed API")
                
                if '<' not in search_res.text or '>' not in search_res.text:
                     logger.error(f"Invalid XML received from esearch: {search_res.text[:200]}...")
                     raise ET.ParseError("Invalid XML structure")
                root = ET.fromstring(search_res.text)
                id_list_elems = root.findall(".//IdList/Id")
                pmids = [id_elem.text for id_elem in id_list_elems if id_elem.text]
                
                if pmids:
                    logger.info(f"Found {len(pmids)} PMIDs via XML esearch, fetching details...")
                    fetch_params = { "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract" }
                    fetch_res = fetch_with_retry("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", fetch_params, timeout=TIMEOUT_LONG)
                    logger.info(f"XML efetch response status code: {fetch_res.status_code}")
                    
                    if fetch_res.text and len(fetch_res.text) > 0:
                        logger.info(f"XML efetch response length: {len(fetch_res.text)} bytes")
                        citations = parse_pubmed_xml(fetch_res.text)
                        logger.info(f"Parsed {len(citations)} citations from XML efetch response")
                    else:
                        logger.error("XML efetch returned empty response")
                else: 
                    logger.warning("No PMIDs found via XML esearch.")
            except Exception as xml_e:
                logger.error(f"Error during XML fallback: {xml_e}", exc_info=True)

    except ET.ParseError as pe:
         logger.error(f"XML Parsing Error during PubMed fetching: {pe}")
         citations = []
    except Exception as e:
        logger.error(f"Critical error during PubMed fetching: {e}", exc_info=True)
        citations = []

    elapsed_time = time.time() - start_time
    logger.info(f"PubMed search completed in {elapsed_time:.2f} seconds.")

    # Sort articles by relevance factors and limit
    if citations:
        logger.info(f"Raw citations before sorting/filtering: {len(citations)}")
        for i, citation in enumerate(citations):
            score = 0
            if citation.is_review: score += 10
            try:
                if citation.year:
                    year = int(citation.year)
                    current_year = datetime.now().year
                    years_old = max(0, current_year - year)
                    recency_score = 5 * (0.85 ** years_old) # Exponential decay
                    score += recency_score
            except ValueError:
                logger.warning(f"Invalid year format for citation: {citation.title}")
            citation.relevance_score = score
        
        citations.sort(key=lambda x: x.relevance_score, reverse=True)
        original_count = len(citations)
        citations = citations[:max_citations] # Limit *after* sorting
        logger.info(f"Limited citations from {original_count} to {len(citations)} based on max_citations={max_citations}")
        
        extracted_genes = extract_genes_from_titles(citations)
        if extracted_genes:
            logger.info(f"Potential genes identified in top citations: {', '.join(extracted_genes)}")
        
        # Log first few citations to help with debugging
        for i, citation in enumerate(citations[:3], 1):
            logger.info(f"Top citation {i}: '{citation.title}', PMID: {citation.pmid}, Year: {citation.year}")

    # Handle case where no citations are found after all attempts
    if not citations:
        logger.warning(f"No relevant citations found for query '{query}' matching the criteria.")

    logger.info(f"Returning {len(citations)} citations")
    return citations


# Function for direct usage from command line
def main():
    """ Main CLI function to fetch citations directly from command line """
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PubMed Citation Fetcher")
    parser.add_argument("query", help="The search query for PubMed")
    # Use the config value as default if available, otherwise use 5
    default_max = CONFIG.get("pubmed", {}).get("max_citations", 5)
    parser.add_argument("--max", type=int, default=default_max, help=f"Maximum number of citations to return (default: {default_max})")
    parser.add_argument("--reviews", action="store_true", help="Prioritize review articles")
    parser.add_argument("--years", type=int, default=5, help="Maximum age of articles in years (0 for no limit)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--use-llm", action="store_true", help="Force LLM-based entity extraction and query optimization")
    args = parser.parse_args()

    # Use LLM if explicitly requested and available
    if args.use_llm and not OPENAI_CLIENT:
        logger.warning("LLM use requested but OpenAI client unavailable. Using regex-based extraction.")
    
    # Create a temporary config override if --use-llm is specified
    original_config = None
    if args.use_llm and OPENAI_CLIENT:
        logger.info("Command line --use-llm flag detected, enabling LLM-based extraction and query generation")
        
        # Ensure citation_extraction exists in CONFIG
        if "citation_extraction" not in CONFIG:
            CONFIG["citation_extraction"] = {
                "entity_extraction_model": "gpt-4o",
                "query_generation_model": "gpt-4o",
                "extraction_temperature": 0.1,
                "generation_temperature": 0.2,
                "extraction_max_tokens": 500,
                "generation_max_tokens": 500
            }
        
        # Ensure all necessary keys exist in citation_extraction
        citation_extraction = CONFIG["citation_extraction"]
        if "entity_extraction_model" not in citation_extraction:
            citation_extraction["entity_extraction_model"] = "gpt-4o"
        if "query_generation_model" not in citation_extraction:
            citation_extraction["query_generation_model"] = "gpt-4o"
        if "extraction_temperature" not in citation_extraction:
            citation_extraction["extraction_temperature"] = 0.1
        if "generation_temperature" not in citation_extraction:
            citation_extraction["generation_temperature"] = 0.2
        if "extraction_max_tokens" not in citation_extraction:
            citation_extraction["extraction_max_tokens"] = 500
        if "generation_max_tokens" not in citation_extraction:
            citation_extraction["generation_max_tokens"] = 500
            
        # Save original settings
        original_config = {
            "use_llm_extraction": CONFIG["citation_extraction"].get("use_llm_extraction", False),
            "use_llm_query_generation": CONFIG["citation_extraction"].get("use_llm_query_generation", False)
        }
        
        # Override settings
        CONFIG["citation_extraction"]["use_llm_extraction"] = True
        CONFIG["citation_extraction"]["use_llm_query_generation"] = True

    try:
        citations = fetch_pubmed_citations(
            args.query,
            max_citations=args.max,
            prioritize_reviews=args.reviews,
            max_age_years=args.years
        )
    finally:
        # Restore original config if it was modified
        if original_config:
            CONFIG["citation_extraction"]["use_llm_extraction"] = original_config["use_llm_extraction"]
            CONFIG["citation_extraction"]["use_llm_query_generation"] = original_config["use_llm_query_generation"]

    if args.format == "json":
        # Prepare the entity extraction and query data
        # Get entities using LLM if available and requested
        if OPENAI_CLIENT and args.use_llm:
            entities = extract_entities_with_llm(args.query)
        else:
            entities = extract_entities_from_text(args.query)
        
        # Generate the optimized query for reporting
        optimized_query = optimize_query(args.query, entities)
        
        json_result = {
            "query": args.query,
            "extracted_entities": entities,
            "optimized_query": optimized_query,
            "citations": [citation.to_dict() for citation in citations],
            "genes_in_titles": extract_genes_from_titles(citations)
         }
        print(json.dumps(json_result, indent=2))
    else: # Text output
        if citations:
            print(f"\nFound {len(citations)} citations for '{args.query}':")
            print("=" * 80)
            for i, citation in enumerate(citations, 1):
                review_marker = " [REVIEW]" if citation.is_review else ""
                print(f"{i}. {citation.title}{review_marker}")
                if citation.authors: print(f"   Authors: {citation.authors}")
                if citation.journal:
                    year_str = f" ({citation.year})" if citation.year else ""
                    print(f"   Journal: {citation.journal}{year_str}")
                if citation.pmid: print(f"   PMID: {citation.pmid}\n   Link: https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/")
                if citation.doi: print(f"   DOI: {citation.doi}")
                print() # Blank line between citations
            genes = extract_genes_from_titles(citations)
            if genes:
                print("-" * 80)
                print("Potential genes mentioned in citation titles:")
                print(", ".join(genes))
                print("-" * 80)
        else:
            print(f"\nNo relevant citations found for '{args.query}' matching the criteria.")


if __name__ == "__main__":
    # Ensure .env is loaded when run as script
    from src.utils import load_environment_variables
    load_environment_variables()
    main()
