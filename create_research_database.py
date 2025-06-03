#!/usr/bin/env python3

"""
MGH/BWH Psychiatry Research Database Creator
Command line tool to download publications and create searchable database
"""

import argparse
import sqlite3
import logging
import requests
import json
import time
import re
import sys
import pickle
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import numpy as np

# Import required libraries with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install scikit-learn tqdm")
    print("For OpenAI support also install: pip install openai")
    sys.exit(1)

@dataclass
class Config:
    """Configuration for the database creation tool"""
    db_path: str = "mgh_bwh_psychiatry_research.db"
    vectorizer_path: str = "./tfidf_vectorizer.pkl"
    embeddings_path: str = "./tfidf_embeddings.pkl"
    metadata_path: str = "./paper_metadata.json"
    llm_provider: str = "ollama"  # "ollama" or "openai"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str = ""
    department_years: int = 3
    download_years: int = 3
    max_papers_per_author: int = 50
    max_papers_for_tfidf: int = 3000  # Limit for TF-IDF processing
    batch_size: int = 100
    rate_limit_delay: float = 0.5

class DatabaseCreator:
    """Creates the research database from scratch"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self._setup_apis()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('database_creation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_apis(self):
        """Setup API connections"""
        if self.config.llm_provider == "openai":
            if self.config.openai_api_key:
                import openai
                openai.api_key = self.config.openai_api_key
                self.logger.info("OpenAI API configured")
            else:
                self.logger.warning("No OpenAI API key provided - themes will not be generated")
        elif self.config.llm_provider == "ollama":
            # Test Ollama connection
            try:
                response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]
                    if any(self.config.ollama_model in name for name in model_names):
                        self.logger.info(f"Ollama configured with model {self.config.ollama_model}")
                    else:
                        self.logger.warning(f"Model {self.config.ollama_model} not found in Ollama. Available: {model_names}")
                else:
                    self.logger.warning("Ollama server not responding properly")
            except Exception as e:
                self.logger.warning(f"Cannot connect to Ollama server at {self.config.ollama_url}: {e}")
        else:
            self.logger.warning("No valid LLM provider configured - themes will not be generated")

    def create_database_schema(self):
        """Create the SQLite database schema"""
        self.logger.info("Creating database schema...")
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # Create authors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS authors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        standardized_name TEXT UNIQUE,
                        department TEXT,
                        affiliation TEXT,
                        total_publications INTEGER DEFAULT 0,
                        recent_publications INTEGER DEFAULT 0,
                        first_publication_date TEXT,
                        last_publication_date TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create publications table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS publications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pmid TEXT UNIQUE,
                        title TEXT,
                        abstract TEXT,
                        journal TEXT,
                        publication_date TEXT,
                        doi TEXT,
                        mesh_terms TEXT,
                        keywords TEXT,
                        publication_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create author_publications junction table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS author_publications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        author_id INTEGER,
                        publication_id INTEGER,
                        author_position INTEGER,
                        is_corresponding BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (author_id) REFERENCES authors (id),
                        FOREIGN KEY (publication_id) REFERENCES publications (id),
                        UNIQUE(author_id, publication_id)
                    )
                ''')
                
                # Create author themes table for LLM-generated themes
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS author_themes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        author_id INTEGER,
                        themes TEXT,
                        summary TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (author_id) REFERENCES authors (id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_std_name ON authors (standardized_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_name ON authors (name)')  # Non-unique index
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_publications_pmid ON publications (pmid)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_publications_date ON publications (publication_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_author_pubs_author ON author_publications (author_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_author_pubs_pub ON author_publications (publication_id)')
                
                # Add missing columns if they don't exist (for schema migration)
                
                # Migrate authors table
                try:
                    cursor.execute('ALTER TABLE authors ADD COLUMN recent_publications INTEGER DEFAULT 0')
                    self.logger.info("Added recent_publications column to authors table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add recent_publications column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE authors ADD COLUMN first_publication_date TEXT')
                    self.logger.info("Added first_publication_date column to authors table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add first_publication_date column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE authors ADD COLUMN last_publication_date TEXT')
                    self.logger.info("Added last_publication_date column to authors table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add last_publication_date column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE authors ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                    self.logger.info("Added created_at column to authors table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add created_at column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE authors ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                    self.logger.info("Added updated_at column to authors table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add updated_at column: {e}")
                
                # Migrate publications table
                try:
                    cursor.execute('ALTER TABLE publications ADD COLUMN doi TEXT')
                    self.logger.info("Added doi column to publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add doi column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE publications ADD COLUMN mesh_terms TEXT')
                    self.logger.info("Added mesh_terms column to publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add mesh_terms column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE publications ADD COLUMN keywords TEXT')
                    self.logger.info("Added keywords column to publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add keywords column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE publications ADD COLUMN publication_type TEXT')
                    self.logger.info("Added publication_type column to publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add publication_type column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE publications ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                    self.logger.info("Added created_at column to publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add created_at column: {e}")
                
                # Migrate author_publications table
                try:
                    cursor.execute('ALTER TABLE author_publications ADD COLUMN author_position INTEGER')
                    self.logger.info("Added author_position column to author_publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add author_position column: {e}")
                
                try:
                    cursor.execute('ALTER TABLE author_publications ADD COLUMN is_corresponding BOOLEAN DEFAULT FALSE')
                    self.logger.info("Added is_corresponding column to author_publications table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Column already exists
                    else:
                        self.logger.warning(f"Could not add is_corresponding column: {e}")
                
                conn.commit()
                self.logger.info("Database schema created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create database schema: {e}")
            return False

    def download_publications(self, custom_affiliations: Optional[List[str]] = None) -> bool:
        """Download publications using proven MGH/BWH/MGB psychiatry query or custom affiliations"""
        self.logger.info(f"Downloading publications for the past {self.config.download_years} years...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.config.download_years)
        start_date_str = start_date.strftime("%Y/%m/%d")
        end_date_str = end_date.strftime("%Y/%m/%d")
        
        if custom_affiliations:
            # Use custom affiliations approach
            return self._download_with_custom_affiliations(custom_affiliations, start_date_str, end_date_str)
        else:
            # Use proven MGH/BWH/MGB psychiatry query pattern
            return self._download_with_proven_query(start_date_str)

    def _download_with_proven_query(self, start_date_str: str) -> bool:
        """Download using the proven MGH/BWH/MGB psychiatry query pattern"""
        self.logger.info("Using proven MGH/BWH/MGB psychiatry query pattern...")
        
        # Exact query pattern from the working system
        query = (
            '("Massachusetts General Hospital"[Affiliation] OR "Mass General"[Affiliation] OR "MGH"[Affiliation] OR '
            '"Brigham and Women\'s Hospital"[Affiliation] OR "Brigham and Womens Hospital"[Affiliation] OR '
            '"Brigham & Women\'s"[Affiliation] OR "Brigham & Womens"[Affiliation] OR "BWH"[Affiliation] OR '
            '"Mass General Brigham"[Affiliation] OR "MGB"[Affiliation]) '
            'AND ("Department of Psychiatry"[Affiliation] OR "Dept of Psychiatry"[Affiliation] OR "Psychiatry Department"[Affiliation]) '
            f'AND ("{start_date_str}"[Date - Publication] : "3000"[Date - Publication])'
        )
        
        self.logger.info(f"PubMed query: {query}")
        
        try:
            # Search for publications
            pmids = self._search_pubmed(query)
            
            if not pmids:
                self.logger.warning("No publications found with MGH/BWH/MGB psychiatry query")
                return False
            
            self.logger.info(f"Found {len(pmids)} publications with MGH/BWH/MGB psychiatry query")
            
            # Download publication details in batches
            total_downloaded = 0
            self.logger.info(f"Processing {len(pmids)} publications in batches of {self.config.batch_size}...")
            
            with tqdm(total=len(pmids), desc="Fetching publication details") as pbar:
                for i in range(0, len(pmids), self.config.batch_size):
                    batch_pmids = pmids[i:i + self.config.batch_size]
                    batch_num = i // self.config.batch_size + 1
                    
                    try:
                        publications = self._fetch_publication_details(batch_pmids)
                        # Filter for valid affiliations
                        valid_publications = self._filter_publications_by_affiliation(publications)
                        stored_count = self._store_publications(valid_publications, "MGH/BWH/MGB Psychiatry")
                        total_downloaded += stored_count
                        
                        pbar.set_postfix({
                            'batch': batch_num,
                            'stored': total_downloaded,
                            'valid': len(valid_publications),
                            'fetched': len(publications)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Batch {batch_num} failed: {e}")
                        
                    pbar.update(len(batch_pmids))
                    
                    # Rate limiting
                    time.sleep(self.config.rate_limit_delay)
            
            self.logger.info(f"Total publications processed: {total_downloaded} (may include existing publications)")
            return total_downloaded > 0
            
        except Exception as e:
            self.logger.error(f"Failed to download publications with proven query: {e}")
            return False

    def _download_with_custom_affiliations(self, affiliations: List[str], start_date_str: str, end_date_str: str) -> bool:
        """Download using custom affiliation list"""
        self.logger.info("Using custom affiliations...")
        
        total_downloaded = 0
        
        for affiliation in affiliations:
            self.logger.info(f"Processing affiliation: {affiliation}")
            
            try:
                # Search for publications
                search_query = f'("{affiliation}"[Affiliation]) AND ("{start_date_str}"[Date - Publication] : "{end_date_str}"[Date - Publication])'
                pmids = self._search_pubmed(search_query)
                
                if not pmids:
                    self.logger.warning(f"No publications found for {affiliation}")
                    continue
                
                self.logger.info(f"Found {len(pmids)} publications for {affiliation}")
                
                # Download publication details in batches
                self.logger.info(f"Processing {len(pmids)} publications in batches of {self.config.batch_size}...")
                
                with tqdm(total=len(pmids), desc=f"Fetching {affiliation}") as pbar:
                    for i in range(0, len(pmids), self.config.batch_size):
                        batch_pmids = pmids[i:i + self.config.batch_size]
                        batch_num = i // self.config.batch_size + 1
                        
                        try:
                            publications = self._fetch_publication_details(batch_pmids)
                            stored_count = self._store_publications(publications, affiliation)
                            total_downloaded += stored_count
                            
                            pbar.set_postfix({
                                'batch': batch_num,
                                'stored': stored_count,
                                'total_stored': total_downloaded
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Batch {batch_num} failed: {e}")
                            
                        pbar.update(len(batch_pmids))
                        
                        # Rate limiting
                        time.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                self.logger.error(f"Failed to download publications for {affiliation}: {e}")
                continue
        
        self.logger.info(f"Total publications processed: {total_downloaded} (may include existing publications)")
        return total_downloaded > 0

    def _search_pubmed(self, query: str) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        try:
            # Use ESearch to get PMIDs
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 10000,  # Maximum results
                'retmode': 'xml',
                'sort': 'pub_date'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = []
            
            for id_element in root.findall('.//Id'):
                pmids.append(id_element.text)
            
            return pmids
            
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []

    def _fetch_publication_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed publication information from PubMed"""
        try:
            # Use EFetch to get publication details
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            publications = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    pub_data = self._parse_publication_xml(article)
                    if pub_data:
                        publications.append(pub_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse article: {e}")
                    continue
            
            return publications
            
        except Exception as e:
            self.logger.error(f"Failed to fetch publication details: {e}")
            return []

    def _parse_publication_xml(self, article_element) -> Optional[Dict]:
        """Parse individual publication XML"""
        try:
            # Extract PMID
            pmid_elem = article_element.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            if not pmid:
                return None
            
            # Extract basic information
            title_elem = article_element.find('.//ArticleTitle')
            title = ""
            if title_elem is not None:
                title = title_elem.text or ""
            
            # Provide default title if missing
            if not title.strip():
                title = f"[No Title Available - PMID: {pmid}]"
            
            abstract_elem = article_element.find('.//Abstract/AbstractText')
            abstract = ""
            if abstract_elem is not None:
                # Handle structured abstracts
                abstract_parts = article_element.findall('.//Abstract/AbstractText')
                abstract_texts = []
                for part in abstract_parts:
                    label = part.get('Label', '')
                    text = part.text or ''
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
                abstract = ' '.join(abstract_texts)
            
            # Extract journal
            journal_elem = article_element.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date = self._extract_publication_date(article_element)
            
            # Extract DOI
            doi_elem = article_element.find('.//ELocationID[@EIdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else ""
            
            # Extract authors
            authors = self._extract_authors(article_element)
            
            # Extract MeSH terms
            mesh_terms = self._extract_mesh_terms(article_element)
            
            # Extract keywords
            keywords = self._extract_keywords(article_element)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'publication_date': pub_date,
                'doi': doi,
                'authors': authors,
                'mesh_terms': mesh_terms,
                'keywords': keywords
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse publication XML: {e}")
            return None

    def _extract_publication_date(self, article_element) -> str:
        """Extract publication date from XML"""
        try:
            # Try different date formats
            date_elem = article_element.find('.//PubDate')
            if date_elem is None:
                return ""
            
            year_elem = date_elem.find('Year')
            month_elem = date_elem.find('Month')
            day_elem = date_elem.find('Day')
            
            year = year_elem.text if year_elem is not None else ""
            month = month_elem.text if month_elem is not None else "01"
            day = day_elem.text if day_elem is not None else "01"
            
            # Convert month name to number if needed
            month_map = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            
            if month in month_map:
                month = month_map[month]
            elif not month.isdigit():
                month = "01"
            
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
        except Exception:
            return ""

    def _extract_authors(self, article_element) -> List[Dict]:
        """Extract author information from XML with improved affiliation handling"""
        authors = []
        
        try:
            author_list = article_element.find('.//AuthorList')
            if author_list is None:
                return authors
            
            for i, author_elem in enumerate(author_list.findall('Author')):
                try:
                    # Extract name components
                    last_name_elem = author_elem.find('LastName')
                    first_name_elem = author_elem.find('ForeName')
                    initials_elem = author_elem.find('Initials')
                    
                    last_name = last_name_elem.text if last_name_elem is not None else ""
                    first_name = first_name_elem.text if first_name_elem is not None else ""
                    initials = initials_elem.text if initials_elem is not None else ""
                    
                    if not last_name:
                        continue
                    
                    # Create full name
                    if first_name:
                        full_name = f"{last_name}, {first_name}"
                    elif initials:
                        full_name = f"{last_name}, {initials}"
                    else:
                        full_name = last_name
                    
                    # Extract all affiliations for this author
                    affiliations = []
                    affiliation_info_list = author_elem.findall('.//AffiliationInfo')
                    for aff_info in affiliation_info_list:
                        aff_elem = aff_info.find('Affiliation')
                        if aff_elem is not None and aff_elem.text:
                            affiliations.append(aff_elem.text.strip())
                    
                    # Combine all affiliations
                    affiliation_text = "; ".join(affiliations) if affiliations else ""
                    
                    authors.append({
                        'name': full_name,
                        'last_name': last_name,
                        'first_name': first_name,
                        'initials': initials,
                        'affiliation': affiliation_text,
                        'position': i + 1
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse author: {e}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Failed to extract authors: {e}")
        
        return authors

    def _extract_mesh_terms(self, article_element) -> List[str]:
        """Extract MeSH terms from XML"""
        mesh_terms = []
        
        try:
            mesh_list = article_element.find('.//MeshHeadingList')
            if mesh_list is not None:
                for mesh_heading in mesh_list.findall('MeshHeading'):
                    descriptor = mesh_heading.find('DescriptorName')
                    if descriptor is not None:
                        mesh_terms.append(descriptor.text)
        
        except Exception as e:
            self.logger.warning(f"Failed to extract MeSH terms: {e}")
        
        return mesh_terms

    def _extract_keywords(self, article_element) -> List[str]:
        """Extract keywords from XML"""
        keywords = []
        
        try:
            keyword_list = article_element.find('.//KeywordList')
            if keyword_list is not None:
                for keyword in keyword_list.findall('Keyword'):
                    if keyword.text:
                        keywords.append(keyword.text)
        
        except Exception as e:
            self.logger.warning(f"Failed to extract keywords: {e}")
        
        return keywords

    def _filter_publications_by_affiliation(self, publications: List[Dict]) -> List[Dict]:
        """Filter publications to ensure authors have valid MGH/BWH/MGB psychiatry affiliations"""
        # Hospital patterns from the proven system
        hospital_patterns = [
            r"Massachusetts General\s+(?:Hospital)?",
            r"Mass\.?\s+General\s+(?:Hospital)?",
            r"MGH",
            r"Brigham\s+(?:and|&)\s+Wom(?:e|a)n(?:')?s\s+(?:Hospital)?", 
            r"BWH",
            r"Mass\.?\s+General\s+Brigham",
            r"MGB"
        ]
        
        # Psychiatry department patterns
        psychiatry_patterns = [
            r"Department\s+of\s+Psychiatry",
            r"Dept\.?\s+of\s+Psychiatry",
            r"Psychiatry\s+Department"
        ]
        
        valid_publications = []
        
        for pub in publications:
            # Check if any author has valid affiliation
            has_valid_author = False
            valid_authors = []
            
            for author_data in pub.get('authors', []):
                affiliation = author_data.get('affiliation', '')
                
                # Check for hospital affiliation
                has_hospital = any(re.search(pattern, affiliation, re.IGNORECASE) 
                                 for pattern in hospital_patterns)
                
                # Check for psychiatry department
                has_psych_dept = any(re.search(pattern, affiliation, re.IGNORECASE) 
                                   for pattern in psychiatry_patterns)
                
                if has_hospital and has_psych_dept:
                    has_valid_author = True
                    # Extract department name for storage
                    department = "Psychiatry"
                    for pattern in psychiatry_patterns:
                        match = re.search(pattern, affiliation, re.IGNORECASE)
                        if match:
                            department = match.group(0).strip()
                            department = re.sub(r'\s+', ' ', department)
                            break
                    
                    # Update author with department info
                    author_data['department'] = department
                    valid_authors.append(author_data)
            
            if has_valid_author:
                # Update publication with only valid authors
                pub['authors'] = valid_authors
                valid_publications.append(pub)
            else:
                # Log publications that don't have valid authors for debugging
                self.logger.debug(f"Publication {pub.get('pmid', 'unknown')} has no valid MGH/BWH/MGB psychiatry authors")
        
        self.logger.info(f"Filtered to {len(valid_publications)} publications with valid MGH/BWH/MGB psychiatry authors")
        
        if not valid_publications:
            self.logger.warning("No publications passed the MGH/BWH/MGB psychiatry affiliation filter")
        
        return valid_publications

    def _store_publications(self, publications: List[Dict], affiliation: str) -> int:
        """Store publications and authors in database with progress tracking"""
        if not publications:
            self.logger.warning(f"No publications to store for {affiliation}")
            return 0
            
        stored_count = 0
        existing_count = 0
        processed_count = 0
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                self.logger.info(f"Attempting to store {len(publications)} publications...")
                
                for pub_data in tqdm(publications, desc="Storing publications", leave=False):
                    try:
                        # Validate required fields
                        if not pub_data.get('pmid'):
                            self.logger.warning("Skipping publication without PMID")
                            continue
                        
                        if not pub_data.get('title'):
                            pub_data['title'] = f"[No Title Available - PMID: {pub_data.get('pmid', 'unknown')}]"
                        
                        # Check if publication already exists
                        cursor.execute('SELECT id FROM publications WHERE pmid = ?', (pub_data['pmid'],))
                        pub_result = cursor.fetchone()
                        
                        if pub_result:
                            publication_id = pub_result[0]
                            existing_count += 1
                            self.logger.debug(f"Publication {pub_data['pmid']} already exists")
                        else:
                            # Insert publication with robust column handling
                            try:
                                # Try full insert with all columns first
                                cursor.execute('''
                                    INSERT INTO publications 
                                    (pmid, title, abstract, journal, publication_date, doi, 
                                     mesh_terms, keywords, publication_type)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    pub_data['pmid'],
                                    pub_data['title'],
                                    pub_data.get('abstract', ''),
                                    pub_data.get('journal', ''),
                                    pub_data.get('publication_date', ''),
                                    pub_data.get('doi', ''),
                                    '|'.join(pub_data.get('mesh_terms', [])),
                                    '|'.join(pub_data.get('keywords', [])),
                                    'Journal Article'
                                ))
                                publication_id = cursor.lastrowid
                                stored_count += 1
                                self.logger.debug(f"Inserted new publication {pub_data['pmid']}")
                            except sqlite3.OperationalError as e:
                                if "no column named" in str(e):
                                    # Fallback to basic insert with only core columns
                                    self.logger.debug(f"Using fallback insert for publication {pub_data['pmid']}: {e}")
                                    cursor.execute('''
                                        INSERT INTO publications 
                                        (pmid, title, abstract, journal, publication_date)
                                        VALUES (?, ?, ?, ?, ?)
                                    ''', (
                                        pub_data['pmid'],
                                        pub_data['title'],
                                        pub_data.get('abstract', ''),
                                        pub_data.get('journal', ''),
                                        pub_data.get('publication_date', '')
                                    ))
                                    publication_id = cursor.lastrowid
                                    stored_count += 1
                                    self.logger.debug(f"Inserted new publication {pub_data['pmid']} with fallback")
                                else:
                                    raise e
                        
                        # Process authors (for both new and existing publications)
                        authors_processed = 0
                        for author_data in pub_data.get('authors', []):
                            author_id = self._store_author(cursor, author_data, affiliation)
                            if author_id:
                                # Link author to publication with robust column handling
                                try:
                                    # Try with author_position first
                                    cursor.execute('''
                                        INSERT OR IGNORE INTO author_publications 
                                        (author_id, publication_id, author_position)
                                        VALUES (?, ?, ?)
                                    ''', (author_id, publication_id, author_data.get('position', 1)))
                                    authors_processed += 1
                                except sqlite3.OperationalError as e:
                                    if "no column named author_position" in str(e):
                                        # Fallback to basic insert without author_position
                                        cursor.execute('''
                                            INSERT OR IGNORE INTO author_publications 
                                            (author_id, publication_id)
                                            VALUES (?, ?)
                                        ''', (author_id, publication_id))
                                        authors_processed += 1
                                    else:
                                        raise e
                                except sqlite3.IntegrityError:
                                    pass  # Relationship already exists
                        
                        processed_count += 1
                        self.logger.debug(f"Processed {authors_processed} authors for publication {pub_data['pmid']}")
                        
                        # Commit every 50 publications for better performance
                        if processed_count % 50 == 0:
                            conn.commit()
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to store publication {pub_data.get('pmid', 'unknown')}: {e}")
                        continue
                
                conn.commit()
                self.logger.info(f"Publication processing complete: {stored_count} new, {existing_count} existing, {processed_count} total processed")
        
        except Exception as e:
            self.logger.error(f"Failed to store publications: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return total processed (new + existing) so the pipeline doesn't think it failed
        return processed_count

    def _store_author(self, cursor, author_data: Dict, affiliation: str) -> Optional[int]:
        """Store author in database and return author ID - with improved name matching"""
        try:
            # Create standardized name
            std_name = self._standardize_author_name(author_data['name'])
            
            # First check if author exists by exact name match
            cursor.execute('SELECT id FROM authors WHERE name = ?', (author_data['name'],))
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # Then check if author exists using standardized name variations
            name_variations = self._get_author_name_variations(std_name)
            
            for variation in name_variations:
                cursor.execute('SELECT id FROM authors WHERE standardized_name = ?', (variation,))
                result = cursor.fetchone()
                if result:
                    # Update with the more complete name if needed
                    cursor.execute(
                        'UPDATE authors SET name = ?, standardized_name = ? WHERE id = ?', 
                        (author_data['name'], std_name, result[0])
                    )
                    return result[0]
            
            # Determine department from affiliation
            department = self._extract_department(author_data.get('affiliation', affiliation))
            
            # Insert new author
            cursor.execute('''
                INSERT INTO authors 
                (name, standardized_name, department, affiliation)
                VALUES (?, ?, ?, ?)
            ''', (
                author_data['name'],
                std_name,
                department,
                author_data.get('affiliation', affiliation)
            ))
            
            return cursor.lastrowid
            
        except Exception as e:
            self.logger.warning(f"Failed to store author {author_data.get('name', 'unknown')}: {e}")
            return None

    def _standardize_author_name(self, name: str) -> str:
        """Standardize author name for consistent matching - handles middle names properly"""
        # Remove extra whitespace and normalize format
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Handle various name formats
        if ',' in name:
            # Format: "Last, First Middle" or "Last, First M" or "Last, F M"
            parts = name.split(',', 1)
            last_name = parts[0].strip()
            first_part = parts[1].strip() if len(parts) > 1 else ""
            
            if first_part:
                # Split first part into components
                first_components = first_part.split()
                if first_components:
                    # Take first component as first name, get initial
                    first_name = first_components[0].strip()
                    if first_name:
                        first_initial = first_name[0].upper()
                        return f"{last_name}, {first_initial}"
            
            return f"{last_name}, "
        else:
            # Format: "First Middle Last" or "First M Last" or "F M Last"
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1].strip()
                first_name = parts[0].strip()
                if first_name:
                    first_initial = first_name[0].upper()
                    return f"{last_name}, {first_initial}"
                else:
                    return f"{last_name}, "
            elif len(parts) == 1:
                return f"{parts[0]}, "
        
        return name

    def _get_author_name_variations(self, standardized_name: str) -> List[str]:
        """Generate possible name variations for matching"""
        variations = [standardized_name]
        
        if ', ' in standardized_name:
            last_name, first_part = standardized_name.split(', ', 1)
            
            # Add variation without first initial
            variations.append(f"{last_name}, ")
            variations.append(last_name)
            
            # Add variations with different spacing
            if first_part:
                variations.append(f"{last_name},{first_part}")
                variations.append(f"{last_name}, {first_part}")
        
        return list(set(variations))  # Remove duplicates

    def _extract_department(self, affiliation: str) -> str:
        """Extract department from affiliation string - optimized for psychiatry research"""
        if not affiliation:
            return "Unknown"
        
        affiliation_lower = affiliation.lower()
        
        # Common department patterns
        dept_patterns = [
            r'department of ([^,;]+)',
            r'dept\.?\s+of ([^,;]+)',
            r'division of ([^,;]+)',
            r'section of ([^,;]+)',
            r'center for ([^,;]+)',
            r'centre for ([^,;]+)',
            r'institute for ([^,;]+)',
            r'program in ([^,;]+)',
            r'laboratory of ([^,;]+)'
        ]
        
        for pattern in dept_patterns:
            match = re.search(pattern, affiliation_lower)
            if match:
                dept = match.group(1).strip()
                # Clean up common department name variations
                dept = re.sub(r'\s+', ' ', dept)
                return dept.title()
        
        # Specific institution and department mappings for psychiatry research
        if 'mclean' in affiliation_lower:
            return "Psychiatry"
        elif any(term in affiliation_lower for term in ['psychiatry', 'psychiatric']):
            return "Psychiatry"
        elif any(term in affiliation_lower for term in ['psychology', 'behavioral']):
            return "Psychology"
        elif any(term in affiliation_lower for term in ['neurology', 'neurological', 'neuroscience']):
            return "Neurology"
        elif any(term in affiliation_lower for term in ['mental health', 'behavioral health']):
            return "Mental Health"
        elif any(term in affiliation_lower for term in ['medicine', 'medical']):
            return "Medicine"
        elif any(term in affiliation_lower for term in ['biostatistics', 'epidemiology']):
            return "Biostatistics"
        elif any(term in affiliation_lower for term in ['public health']):
            return "Public Health"
        
        return "Unknown"

    def update_author_statistics(self):
        """Update publication counts and date ranges for authors"""
        self.logger.info("Updating author statistics...")
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # Check which columns exist in the authors table
                cursor.execute("PRAGMA table_info(authors)")
                columns = [row[1] for row in cursor.fetchall()]
                
                has_recent_pubs = 'recent_publications' in columns
                has_first_date = 'first_publication_date' in columns
                has_last_date = 'last_publication_date' in columns
                
                # Build update query based on available columns
                update_parts = []
                
                # Always update total_publications (should always exist)
                update_parts.append('''
                    total_publications = (
                        SELECT COUNT(*)
                        FROM author_publications ap
                        WHERE ap.author_id = authors.id
                    )
                ''')
                
                if has_recent_pubs:
                    update_parts.append('''
                        recent_publications = (
                            SELECT COUNT(*)
                            FROM author_publications ap
                            JOIN publications p ON ap.publication_id = p.id
                            WHERE ap.author_id = authors.id
                            AND p.publication_date >= date('now', '-3 years')
                        )
                    ''')
                
                if has_first_date:
                    update_parts.append('''
                        first_publication_date = (
                            SELECT MIN(p.publication_date)
                            FROM author_publications ap
                            JOIN publications p ON ap.publication_id = p.id
                            WHERE ap.author_id = authors.id
                        )
                    ''')
                
                if has_last_date:
                    update_parts.append('''
                        last_publication_date = (
                            SELECT MAX(p.publication_date)
                            FROM author_publications ap
                            JOIN publications p ON ap.publication_id = p.id
                            WHERE ap.author_id = authors.id
                        )
                    ''')
                
                # Execute the update with available columns
                if update_parts:
                    update_query = f"UPDATE authors SET {', '.join(update_parts)}"
                    cursor.execute(update_query)
                    
                    self.logger.info(f"Updated author statistics for available columns: {[col for col in ['total_publications', 'recent_publications', 'first_publication_date', 'last_publication_date'] if col == 'total_publications' or (col == 'recent_publications' and has_recent_pubs) or (col == 'first_publication_date' and has_first_date) or (col == 'last_publication_date' and has_last_date)]}")
                else:
                    self.logger.warning("No updatable columns found in authors table")
                
                conn.commit()
                self.logger.info("Author statistics updated successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update author statistics: {e}")
            return False

    def generate_author_themes(self, regenerate_all=False, min_publications=2):
        """Generate research themes for authors using LLM with improved error handling"""
        if self.config.llm_provider == "openai" and not self.config.openai_api_key:
            self.logger.warning("No OpenAI API key - skipping theme generation")
            return False
        elif self.config.llm_provider == "ollama":
            # Test Ollama connection
            try:
                response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    self.logger.warning("Ollama server not accessible - skipping theme generation")
                    return False
            except Exception as e:
                self.logger.warning(f"Cannot connect to Ollama - skipping theme generation: {e}")
                return False
        else:
            self.logger.warning("No valid LLM configuration - skipping theme generation")
            return False
            
        self.logger.info(f"Generating author research themes using {self.config.llm_provider}...")
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing themes if regenerating all
                if regenerate_all:
                    self.logger.info("Clearing existing themes for regeneration...")
                    cursor.execute('DELETE FROM author_themes')
                    conn.commit()
                
                # Check author publication statistics first
                cursor.execute('SELECT COUNT(*) FROM authors WHERE total_publications >= ?', (min_publications,))
                eligible_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM authors WHERE total_publications >= 1')
                total_with_pubs = cursor.fetchone()[0]
                
                self.logger.info(f"Authors with >= {min_publications} publications: {eligible_count}")
                self.logger.info(f"Total authors with >= 1 publication: {total_with_pubs}")
                
                # Get authors with sufficient publications
                if regenerate_all:
                    condition = 'WHERE a.total_publications >= ?'
                    params = (min_publications,)
                else:
                    condition = '''WHERE a.total_publications >= ? 
                                   AND a.id NOT IN (SELECT author_id FROM author_themes)'''
                    params = (min_publications,)
                
                query = f'''
                    SELECT a.id, a.name, a.standardized_name, a.total_publications
                    FROM authors a
                    {condition}
                    ORDER BY a.total_publications DESC
                '''
                
                cursor.execute(query, params)
                authors = cursor.fetchall()
                
                if not authors:
                    self.logger.warning(f"No authors found with >= {min_publications} publications")
                    # Try with lower threshold
                    self.logger.info("Trying with minimum 1 publication...")
                    cursor.execute('''
                        SELECT a.id, a.name, a.standardized_name, a.total_publications
                        FROM authors a
                        WHERE a.total_publications >= 1
                        ORDER BY a.total_publications DESC
                        LIMIT 10
                    ''')
                    sample_authors = cursor.fetchall()
                    self.logger.info(f"Sample authors: {sample_authors}")
                    return False
                
                self.logger.info(f"Generating themes for {len(authors)} authors...")
                
                processed = 0
                successful = 0
                
                for author_id, name, std_name, total_pubs in tqdm(authors, desc="Generating themes"):
                    try:
                        self.logger.info(f"Processing {name} ({processed + 1}/{len(authors)}) - {total_pubs} publications...")
                        
                        # Get author's publications
                        publications = self._get_author_publications(cursor, author_id)
                        
                        if len(publications) < 1:
                            self.logger.warning(f"Skipping {name} - no publications found in database")
                            continue
                        
                        self.logger.info(f"Found {len(publications)} publications for {name}")
                        
                        # Generate themes using LLM
                        themes, summary = self._generate_themes_with_llm(name, publications)
                        
                        if themes and len(themes) > 0:
                            # Store themes in database
                            cursor.execute('''
                                INSERT OR REPLACE INTO author_themes (author_id, themes, summary)
                                VALUES (?, ?, ?)
                            ''', (author_id, '|'.join(themes), summary))
                            
                            conn.commit()
                            successful += 1
                            self.logger.info(f"✅ Generated {len(themes)} themes for {name}: {', '.join(themes)}")
                        else:
                            self.logger.warning(f"❌ No valid themes generated for {name}")
                        
                        processed += 1
                        
                        # Rate limiting for API calls
                        time.sleep(1)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to generate themes for {name}: {e}")
                        continue
                
                self.logger.info(f"Theme generation completed: {successful}/{processed} authors successfully processed")
                
                # Show final statistics
                cursor.execute('SELECT COUNT(*) FROM author_themes')
                total_themes = cursor.fetchone()[0]
                self.logger.info(f"Total authors with themes in database: {total_themes}")
                
                return successful > 0
                
        except Exception as e:
            self.logger.error(f"Failed to generate author themes: {e}")
            return False

    def _get_author_publications(self, cursor, author_id: int) -> List[Dict]:
        """Get publications for an author"""
        cursor.execute('''
            SELECT p.title, p.abstract, p.journal, p.publication_date, p.mesh_terms, p.keywords
            FROM publications p
            JOIN author_publications ap ON p.id = ap.publication_id
            WHERE ap.author_id = ?
            ORDER BY p.publication_date DESC
            LIMIT ?
        ''', (author_id, self.config.max_papers_per_author))
        
        publications = []
        for row in cursor.fetchall():
            title, abstract, journal, pub_date, mesh_terms, keywords = row
            publications.append({
                'title': title or '',
                'abstract': abstract or '',
                'journal': journal or '',
                'publication_date': pub_date or '',
                'mesh_terms': mesh_terms.split('|') if mesh_terms else [],
                'keywords': keywords.split('|') if keywords else []
            })
        
        return publications

    def _generate_themes_with_llm(self, author_name: str, publications: List[Dict]) -> Tuple[List[str], str]:
        """Generate research themes using LLM (Ollama or OpenAI) with improved prompt and validation"""
        try:
            # Prepare publication data for analysis
            pub_texts = []
            for pub in publications[:10]:  # Limit to recent 10 publications
                text = f"Title: {pub['title']}\nAbstract: {pub['abstract'][:500]}"
                if pub['mesh_terms']:
                    text += f"\nMeSH Terms: {', '.join(pub['mesh_terms'][:5])}"
                pub_texts.append(text)
            
            publications_text = "\n\n---\n\n".join(pub_texts)
            
            # Create improved prompt for LLM with specific constraints
            prompt = f"""Analyze the following research publications by {author_name} and identify their main research themes.

Publications:
{publications_text}

Based on these publications, please provide:
1. A list of 1-3 SHORT research themes (each theme should be 3-8 words maximum, ideally 3-4 words)
2. A brief summary of their research expertise (2-3 sentences)

IMPORTANT GUIDELINES:
- Each theme should be very concise (3-8 words maximum)
- Examples of good themes: "Depression Treatment", "Neuroimaging Studies", "Adolescent Mental Health"
- Examples of bad themes: "Mental health interventions particularly digital health approaches", "Comprehensive analysis of psychiatric disorders"
- If all their work fits into 1-2 themes, that's perfectly fine - don't force 3 themes
- Focus on the most prominent research areas only
- If there are only 1-2 publications, extract the key research focus from those

Format your response exactly like this:
THEMES:
[short theme 1]
[short theme 2]
[short theme 3 if needed]

SUMMARY:
[2-3 sentence summary of their research expertise]
"""
            
            # Call appropriate LLM
            if self.config.llm_provider == "ollama":
                content = self._call_ollama(prompt)
            elif self.config.llm_provider == "openai":
                content = self._call_openai(prompt)
            else:
                return [], ""
            
            if not content:
                return [], ""
            
            # Parse response with improved validation
            themes, summary = self._parse_llm_response(content)
            
            # Additional validation and cleaning
            themes = self._validate_and_clean_themes(themes)
            
            return themes, summary
            
        except Exception as e:
            self.logger.warning(f"LLM theme generation failed for {author_name}: {e}")
            return [], ""

    def _validate_and_clean_themes(self, themes: List[str]) -> List[str]:
        """Validate and clean themes to ensure they meet quality standards"""
        cleaned_themes = []
        
        for theme in themes:
            if not theme:
                continue
                
            # Remove common prefixes and clean up
            theme = theme.strip()
            theme = re.sub(r'^[\d\.\-\*\•\s]*', '', theme)  # Remove numbering/bullets
            theme = re.sub(r'\s+', ' ', theme)  # Normalize whitespace
            
            # Remove common unwanted prefixes
            unwanted_prefixes = [
                'research in', 'studies on', 'analysis of', 'investigation of',
                'work on', 'focus on', 'expertise in', 'research on'
            ]
            theme_lower = theme.lower()
            for prefix in unwanted_prefixes:
                if theme_lower.startswith(prefix):
                    theme = theme[len(prefix):].strip()
                    break
            
            # Validate theme length (3-8 words, 5-50 characters)
            word_count = len(theme.split())
            char_count = len(theme)
            
            if word_count < 1 or word_count > 8:
                self.logger.debug(f"Skipping theme with {word_count} words: {theme}")
                continue
                
            if char_count < 5 or char_count > 50:
                self.logger.debug(f"Skipping theme with {char_count} characters: {theme}")
                continue
            
            # Remove themes that are too generic
            generic_themes = [
                'research', 'studies', 'analysis', 'investigation', 'work',
                'mental health', 'psychiatry', 'psychology', 'neuroscience'
            ]
            if theme.lower().strip() in generic_themes:
                self.logger.debug(f"Skipping generic theme: {theme}")
                continue
            
            # Capitalize properly
            theme = self._capitalize_theme(theme)
            
            cleaned_themes.append(theme)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_themes = []
        for theme in cleaned_themes:
            theme_lower = theme.lower()
            if theme_lower not in seen:
                seen.add(theme_lower)
                unique_themes.append(theme)
        
        return unique_themes[:3]  # Maximum 3 themes

    def _capitalize_theme(self, theme: str) -> str:
        """Properly capitalize theme text"""
        # Split into words and capitalize appropriately
        words = theme.split()
        capitalized_words = []
        
        # Words that should typically stay lowercase (unless first word)
        lowercase_words = {'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'the', 'a', 'an'}
        
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in lowercase_words:
                # Capitalize first letter, keep rest as is (preserving acronyms)
                if word.isupper() and len(word) > 1:
                    # Likely an acronym, keep as is
                    capitalized_words.append(word)
                else:
                    capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word.lower())
        
        return ' '.join(capitalized_words)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                self.logger.warning(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.warning(f"Ollama API call failed: {e}")
            return ""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research analyst who identifies research themes from academic publications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.warning(f"OpenAI API call failed: {e}")
            return ""

    def _parse_llm_response(self, content: str) -> Tuple[List[str], str]:
        """Robustly parse LLM response to extract themes and summary with improved validation"""
        themes = []
        summary = ""
        
        try:
            # Split content into sections
            lines = content.split('\n')
            current_section = None
            theme_lines = []
            summary_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Identify sections
                if 'THEMES:' in line.upper() or 'THEME:' in line.upper():
                    current_section = 'themes'
                    continue
                elif 'SUMMARY:' in line.upper():
                    current_section = 'summary'
                    continue
                
                # Collect content
                if current_section == 'themes' and line:
                    # Clean theme text
                    theme = re.sub(r'^[\d\.\-\*\•\s]+', '', line)  # Remove numbering/bullets
                    theme = theme.strip()
                    
                    # Basic validation - not too short, not too long
                    word_count = len(theme.split())
                    if theme and word_count >= 1 and word_count <= 8 and len(theme) <= 50:
                        theme_lines.append(theme)
                    elif theme:
                        self.logger.debug(f"Rejected theme (length): {theme} ({word_count} words)")
                        
                elif current_section == 'summary' and line:
                    summary_lines.append(line)
            
            # Process themes
            themes = theme_lines[:3]  # Maximum 3 themes
            
            # Process summary
            summary = ' '.join(summary_lines).strip()
            
            # Fallback parsing if structured format failed
            if not themes and not summary:
                themes, summary = self._fallback_parse(content)
            
            # Final validation - ensure we have quality themes
            if not themes:
                self.logger.warning("No valid themes extracted from LLM response")
            
            # Ensure summary exists
            if not summary:
                summary = "Research expertise analysis not available"
            
            return themes, summary
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return [], "Research expertise analysis not available"

    def _fallback_parse(self, content: str) -> Tuple[List[str], str]:
        """Improved fallback parsing when structured format is not followed"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        themes = []
        summary_parts = []
        
        # Look for theme-like lines (shorter, focused statements)
        # and summary-like lines (longer, descriptive sentences)
        for line in lines:
            # Clean the line
            clean_line = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
            
            word_count = len(clean_line.split())
            char_count = len(clean_line)
            
            if word_count < 1:  # Too short
                continue
            elif word_count <= 8 and char_count <= 50 and len(themes) < 3:  # Theme-like
                # Additional check - avoid sentences (no periods in middle)
                if '.' not in clean_line.rstrip('.') and '?' not in clean_line and '!' not in clean_line:
                    themes.append(clean_line)
            elif word_count > 8:  # Summary-like
                summary_parts.append(clean_line)
        
        summary = ' '.join(summary_parts) if summary_parts else ""
        
        return themes, summary

    def create_tfidf_search_database(self):
        """Create TF-IDF searchable database matching app.py requirements"""
        try:
            if not Path(self.config.db_path).exists():
                self.logger.error(f"SQLite database not found at {self.config.db_path}")
                return False
            
            self.logger.info("Creating TF-IDF search database...")
            
            # Get publications exactly as app.py does
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT p.id, p.pmid, p.title, p.abstract, p.journal, p.publication_date,
                           GROUP_CONCAT(a.name, '; ') as authors,
                           GROUP_CONCAT(a.standardized_name, '; ') as std_authors,
                           GROUP_CONCAT(a.department, '; ') as departments
                    FROM publications p
                    JOIN author_publications ap ON p.id = ap.publication_id
                    JOIN authors a ON ap.author_id = a.id
                    WHERE p.abstract IS NOT NULL AND LENGTH(p.abstract) > 100
                    GROUP BY p.id
                    ORDER BY p.publication_date DESC
                    LIMIT ?
                ''', (self.config.max_papers_for_tfidf,))
                publications = cursor.fetchall()
            
            if not publications:
                self.logger.error("No publications found in database")
                return False
            
            self.logger.info(f"Processing {len(publications)} publications for TF-IDF...")
            
            # Clean text function matching app.py
            def clean_text(text):
                """Clean text for TF-IDF processing - matches app.py"""
                if not text:
                    return ""
                # Simple but effective cleaning
                text = str(text).replace('\x00', ' ').replace('\r', ' ').replace('\n', ' ')
                text = ' '.join(text.split())
                # Keep reasonable length for TF-IDF
                if len(text) > 3000:
                    text = text[:3000]
                return text.strip()
            
            # Prepare documents and metadata exactly as app.py
            documents = []
            metadata = []
            
            valid_count = 0
            for i, pub in enumerate(tqdm(publications, desc="Processing publications")):
                try:
                    pub_id, pmid, title, abstract, journal, pub_date, authors, std_authors, departments = pub
                    
                    # Clean text fields
                    title_clean = clean_text(title)
                    abstract_clean = clean_text(abstract)
                    
                    if not title_clean and not abstract_clean:
                        continue
                    
                    # Create searchable text - exactly as app.py does
                    if title_clean and abstract_clean:
                        # Include title multiple times for higher weight in TF-IDF
                        search_text = f"{title_clean} {title_clean} {abstract_clean}"
                    else:
                        search_text = title_clean or abstract_clean
                    
                    if len(search_text.strip()) < 30:
                        continue
                    
                    documents.append(search_text)
                    metadata.append({
                        'pub_id': str(pub_id),
                        'pmid': str(pmid) if pmid else '',
                        'title': title_clean,
                        'journal': clean_text(journal),
                        'publication_date': str(pub_date) if pub_date else '',
                        'authors': clean_text(authors),
                        'std_authors': clean_text(std_authors),
                        'departments': clean_text(departments)
                    })
                    valid_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing publication {i}: {e}")
                    continue
            
            if not documents:
                self.logger.error("No valid documents to process")
                return False
            
            if len(documents) < 10:
                self.logger.error(f"Too few documents ({len(documents)}) for reliable TF-IDF")
                return False
            
            self.logger.info(f"Creating TF-IDF vectors for {len(documents)} documents...")
            
            # Create TF-IDF vectorizer with same settings as app.py
            vectorizer = TfidfVectorizer(
                max_features=8000,      # Good balance of features
                stop_words='english',   # Remove common words
                ngram_range=(1, 2),     # Include single words and bigrams
                min_df=2,               # Ignore rare terms
                max_df=0.95,            # Ignore too common terms
                sublinear_tf=True,      # Use log scaling
                norm='l2'               # L2 normalization
            )
            
            # Fit and transform documents
            try:
                paper_embeddings = vectorizer.fit_transform(documents)
                
                # Verify vectorizer is properly fitted
                if not hasattr(vectorizer, 'idf_'):
                    self.logger.error("Vectorizer fitting failed - no IDF computed")
                    return False
                
                self.logger.info(f"TF-IDF vectorizer fitted with {len(vectorizer.get_feature_names_out())} features")
                
            except Exception as e:
                self.logger.error(f"Failed to fit TF-IDF vectorizer: {e}")
                return False
            
            # Save the vectorizer, embeddings, and metadata for app.py
            try:
                self.logger.info("Saving TF-IDF search data to disk...")
                
                with open(self.config.vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer, f)
                self.logger.info(f"✅ Saved TF-IDF vectorizer to {self.config.vectorizer_path}")
                
                with open(self.config.embeddings_path, 'wb') as f:
                    pickle.dump(paper_embeddings, f)
                self.logger.info(f"✅ Saved TF-IDF embeddings to {self.config.embeddings_path}")
                
                with open(self.config.metadata_path, 'w') as f:
                    json.dump(metadata, f)
                self.logger.info(f"✅ Saved paper metadata to {self.config.metadata_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to save TF-IDF data: {e}")
                return False
            
            # Test the vectorizer
            try:
                test_query = "machine learning"
                test_vector = vectorizer.transform([test_query])
                self.logger.info(f"✅ TF-IDF vectorizer test successful (query: '{test_query}', output shape: {test_vector.shape})")
            except Exception as e:
                self.logger.error(f"TF-IDF vectorizer test failed: {e}")
                return False
            
            self.logger.info(f"✅ TF-IDF search database created successfully with {len(documents)} papers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create TF-IDF search database: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def debug_database_stats(self):
        """Print database statistics for debugging"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                print("\n" + "="*50)
                print("DATABASE STATISTICS")
                print("="*50)
                
                # Authors statistics
                cursor.execute('SELECT COUNT(*) FROM authors')
                total_authors = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM authors WHERE total_publications >= 1')
                authors_with_pubs = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM authors WHERE total_publications >= 2')
                authors_with_2plus = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM authors WHERE total_publications >= 3')
                authors_with_3plus = cursor.fetchone()[0]
                
                print(f"Total Authors: {total_authors}")
                print(f"Authors with ≥1 publication: {authors_with_pubs}")
                print(f"Authors with ≥2 publications: {authors_with_2plus}")
                print(f"Authors with ≥3 publications: {authors_with_3plus}")
                
                # Publications statistics
                cursor.execute('SELECT COUNT(*) FROM publications')
                total_pubs = cursor.fetchone()[0]
                print(f"\nTotal Publications: {total_pubs}")
                
                # Themes statistics
                cursor.execute('SELECT COUNT(*) FROM author_themes')
                total_themes = cursor.fetchone()[0]
                print(f"Authors with themes: {total_themes}")
                
                # Sample authors with most publications
                cursor.execute('''
                    SELECT name, total_publications 
                    FROM authors 
                    WHERE total_publications > 0 
                    ORDER BY total_publications DESC 
                    LIMIT 10
                ''')
                top_authors = cursor.fetchall()
                
                print(f"\nTop 10 Authors by Publication Count:")
                for name, pub_count in top_authors:
                    cursor.execute('SELECT COUNT(*) FROM author_themes WHERE author_id = (SELECT id FROM authors WHERE name = ?)', (name,))
                    has_themes = cursor.fetchone()[0] > 0
                    theme_status = "✅" if has_themes else "❌"
                    print(f"  {name}: {pub_count} pubs {theme_status}")
                
                # Sample themes
                if total_themes > 0:
                    cursor.execute('''
                        SELECT a.name, at.themes 
                        FROM author_themes at
                        JOIN authors a ON at.author_id = a.id
                        LIMIT 5
                    ''')
                    sample_themes = cursor.fetchall()
                    
                    print(f"\nSample Themes:")
                    for name, themes in sample_themes:
                        theme_list = themes.split('|') if themes else []
                        print(f"  {name}: {', '.join(theme_list)}")
                
                print("="*50)
                
        except Exception as e:
            print(f"Error getting database stats: {e}")

def main():
    """Main function to run the database creation tool"""
    parser = argparse.ArgumentParser(
        description="Create MGH/BWH Psychiatry Research Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with proven MGH/BWH/MGB psychiatry query (RECOMMENDED)
  python database_creator.py
  
  # Regenerate all themes with lower publication threshold
  python database_creator.py --regenerate-themes --min-publications 1
  
  # Custom parameters with Ollama
  python database_creator.py --dept-years 5 --download-years 2 --ollama-model llama3.1
  
  # Use OpenAI instead of Ollama
  python database_creator.py --llm-provider openai --openai-key YOUR_KEY
  
  # Skip theme generation
  python database_creator.py --llm-provider none
  
  # Use custom affiliations instead of proven query
  python database_creator.py --affiliations "Johns Hopkins" "Mayo Clinic"
  
  # Just show database statistics
  python database_creator.py --debug-only
  
  # Note: Creates TF-IDF search database compatible with app.py
  # Default uses the proven MGH/BWH/MGB psychiatry query that includes:
  # Massachusetts General Hospital, Brigham and Women's Hospital, Mass General Brigham
  # with Department of Psychiatry filtering
        """
    )
    
    parser.add_argument(
        '--affiliations', 
        nargs='+',
        default=None,  # Will use the proven MGH/BWH/MGB query pattern
        help='Custom affiliations to search for (default: uses proven MGH/BWH/MGB psychiatry query)'
    )
    
    parser.add_argument(
        '--dept-years',
        type=int,
        default=3,
        help='Years to consider for department membership (default: 3)'
    )
    
    parser.add_argument(
        '--download-years',
        type=int,
        default=3,
        help='Years of publications to download (default: 3)'
    )
    
    parser.add_argument(
        '--db-path',
        default="mgh_bwh_psychiatry_research.db",
        help='Database file path (default: mgh_bwh_psychiatry_research.db)'
    )
    
    parser.add_argument(
        '--llm-provider',
        choices=['ollama', 'openai', 'none'],
        default='ollama',
        help='LLM provider for theme generation (default: ollama)'
    )
    
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--ollama-model',
        default='llama3.2',
        help='Ollama model name (default: llama3.2)'
    )
    
    parser.add_argument(
        '--openai-key',
        help='OpenAI API key (for OpenAI provider)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for API calls (default: 100)'
    )
    
    parser.add_argument(
        '--max-papers',
        type=int,
        default=3000,
        help='Maximum papers to process for TF-IDF (default: 3000)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Rate limit delay in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--regenerate-themes',
        action='store_true',
        help='Regenerate all themes (clear existing and recreate)'
    )
    
    parser.add_argument(
        '--min-publications',
        type=int,
        default=2,
        help='Minimum publications required for theme generation (default: 2)'
    )
    
    parser.add_argument(
        '--debug-only',
        action='store_true',
        help='Only show database statistics and exit'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        db_path=args.db_path,
        llm_provider=args.llm_provider,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        openai_api_key=args.openai_key or "",
        department_years=args.dept_years,
        download_years=args.download_years,
        max_papers_for_tfidf=args.max_papers,
        batch_size=args.batch_size,
        rate_limit_delay=args.rate_limit
    )
    
    # Create database creator
    creator = DatabaseCreator(config)
    
    # Handle debug-only mode
    if args.debug_only:
        creator.debug_database_stats()
        return 0
    
    try:
        print("=" * 60)
        print("MGH/BWH Psychiatry Research Database Creator")
        print("=" * 60)
        print(f"Affiliations: {'Proven MGH/BWH/MGB psychiatry query' if not args.affiliations else ', '.join(args.affiliations)}")
        print(f"Download years: {args.download_years}")
        print(f"Department years: {args.dept_years}")
        print(f"Database path: {args.db_path}")
        print(f"LLM provider: {args.llm_provider}")
        if args.llm_provider == "ollama":
            print(f"Ollama URL: {args.ollama_url}")
            print(f"Ollama model: {args.ollama_model}")
        elif args.llm_provider == "openai":
            print(f"OpenAI API: {'Configured' if args.openai_key else 'Not configured'}")
        print("=" * 60)
        
        # Step 1: Create database schema
        print("\n1. Creating database schema...")
        if not creator.create_database_schema():
            print("❌ Failed to create database schema")
            return 1
        print("✅ Database schema created")
        
        # Step 2: Download publications
        print(f"\n2. Downloading publications...")
        if args.affiliations:
            print(f"Using custom affiliations: {', '.join(args.affiliations)}")
            if not creator.download_publications(args.affiliations):
                print("❌ Failed to download publications")
                return 1
        else:
            print("Using proven MGH/BWH/MGB psychiatry query pattern")
            if not creator.download_publications():
                print("❌ Failed to download publications")
                return 1
        print("✅ Publications downloaded")
        
        # Step 3: Update author statistics
        print("\n3. Updating author statistics...")
        if not creator.update_author_statistics():
            print("❌ Failed to update author statistics")
            return 1
        print("✅ Author statistics updated")
        
        # Step 4: Generate themes (if enabled)
        if args.llm_provider != 'none':
            print(f"\n4. Generating research themes with {args.llm_provider.upper()}...")
            if args.regenerate_themes:
                print("   ⚠️  Regenerating ALL themes (clearing existing)")
            print(f"   Minimum publications threshold: {args.min_publications}")
            
            if not creator.generate_author_themes(
                regenerate_all=args.regenerate_themes, 
                min_publications=args.min_publications
            ):
                print("⚠️  Theme generation failed, continuing without themes")
            else:
                print("✅ Research themes generated")
        else:
            print("\n4. Skipping theme generation (disabled)")
        
        # Step 5: Create TF-IDF search database
        print("\n5. Creating TF-IDF search database...")
        if not creator.create_tfidf_search_database():
            print("❌ Failed to create TF-IDF search database")
            return 1
        print("✅ TF-IDF search database created")
        
        print("\n" + "=" * 60)
        print("✅ Database creation completed successfully!")
        print("=" * 60)
        print(f"SQLite Database: {args.db_path}")
        print(f"TF-IDF Vectorizer: {creator.config.vectorizer_path}")
        print(f"TF-IDF Embeddings: {creator.config.embeddings_path}")
        print(f"Paper Metadata: {creator.config.metadata_path}")
        print("You can now use the TF-IDF search application (app.py) to query the database.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())