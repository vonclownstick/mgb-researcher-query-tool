#!/usr/bin/env python3

"""
Smart Paper Search with TF-IDF - Reliable FastAPI Version
No ChromaDB dependencies, uses scikit-learn for fast keyword search
"""

import sqlite3
import logging
import os
import pickle
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str

class InitializeResponse(BaseModel):
    success: bool
    message: str = None
    error: str = None

@dataclass
class QueryConfig:
    """Configuration for the query tool"""
    db_path: str = "mgh_bwh_psychiatry_research.db"
    vectorizer_path: str = "./tfidf_vectorizer.pkl"
    embeddings_path: str = "./tfidf_embeddings.pkl"
    metadata_path: str = "./paper_metadata.json"
    max_results: int = 5
    similarity_threshold: float = 0.15  # Lower threshold for TF-IDF
    min_relevant_papers: int = 2

class TFIDFPaperSearch:
    """TF-IDF based paper search - reliable and fast"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._setup_logging()
        self.vectorizer = None
        self.paper_embeddings = None
        self.paper_metadata = None
        self._collection_ready = False
        self._load_existing_data()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_existing_data(self):
        """Load existing TF-IDF vectorizer and embeddings if available"""
        try:
            if (Path(self.config.vectorizer_path).exists() and 
                Path(self.config.embeddings_path).exists() and 
                Path(self.config.metadata_path).exists()):
                
                self.logger.info("Loading existing TF-IDF search data...")
                
                with open(self.config.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.config.embeddings_path, 'rb') as f:
                    self.paper_embeddings = pickle.load(f)
                
                with open(self.config.metadata_path, 'r') as f:
                    self.paper_metadata = json.load(f)
                
                self._collection_ready = True
                self.logger.info(f"Loaded TF-IDF search data with {len(self.paper_metadata)} papers")
                return True
        except Exception as e:
            self.logger.info(f"No existing TF-IDF search data found: {e}")
        
        return False
    
    def create_paper_database(self):
        """Create TF-IDF searchable database"""
        try:
            if not Path(self.config.db_path).exists():
                self.logger.error(f"SQLite database not found at {self.config.db_path}")
                return False
            
            self.logger.info("Creating TF-IDF search database...")
            
            # Get all publications
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
                    LIMIT 3000
                ''')  # Limit for faster processing and reasonable size
                publications = cursor.fetchall()
            
            if not publications:
                self.logger.error("No publications found in database")
                return False
            
            self.logger.info(f"Processing {len(publications)} publications...")
            
            # Prepare documents and metadata
            documents = []
            metadata = []
            
            def clean_text(text):
                """Clean text for TF-IDF processing"""
                if not text:
                    return ""
                # Simple but effective cleaning
                text = str(text).replace('\x00', ' ').replace('\r', ' ').replace('\n', ' ')
                text = ' '.join(text.split())
                # Keep reasonable length for TF-IDF
                if len(text) > 3000:
                    text = text[:3000]
                return text.strip()
            
            valid_count = 0
            for i, pub in enumerate(publications):
                try:
                    pub_id, pmid, title, abstract, journal, pub_date, authors, std_authors, departments = pub
                    
                    # Clean text fields
                    title_clean = clean_text(title)
                    abstract_clean = clean_text(abstract)
                    
                    if not title_clean and not abstract_clean:
                        continue
                    
                    # Create searchable text - optimized for TF-IDF
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
                    
                    if valid_count % 100 == 0:
                        self.logger.info(f"Processed {valid_count} publications...")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing publication {i}: {e}")
                    continue
            
            if not documents:
                self.logger.error("No valid documents to process")
                return False
            
            self.logger.info(f"Creating TF-IDF vectors for {len(documents)} documents...")
            
            # Create TF-IDF vectorizer with optimized settings
            self.vectorizer = TfidfVectorizer(
                max_features=8000,      # Good balance of features
                stop_words='english',   # Remove common words
                ngram_range=(1, 2),     # Include single words and bigrams
                min_df=2,               # Ignore rare terms
                max_df=0.95,            # Ignore too common terms
                sublinear_tf=True,      # Use log scaling
                norm='l2'               # L2 normalization
            )
            
            # Fit and transform documents
            self.paper_embeddings = self.vectorizer.fit_transform(documents)
            self.paper_metadata = metadata
            
            # Save the vectorizer and embeddings
            self.logger.info("Saving TF-IDF search data to disk...")
            
            with open(self.config.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(self.config.embeddings_path, 'wb') as f:
                pickle.dump(self.paper_embeddings, f)
            
            with open(self.config.metadata_path, 'w') as f:
                json.dump(self.paper_metadata, f)
            
            self._collection_ready = True
            self.logger.info(f"Created TF-IDF search database with {len(documents)} papers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create TF-IDF search database: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def search_with_smart_filtering(self, query: str) -> List[Dict]:
        """Search using TF-IDF similarity with auto-initialization"""
        # Auto-initialize if needed
        if not self._collection_ready:
            if os.environ.get("PORT"):  # In deployment
                self.logger.error("No pre-built search database found in deployment")
                self.logger.error("Please build database locally and include in repository")
                return []
            else:
                self.logger.info("Auto-initializing TF-IDF search database...")
                if self.create_paper_database():
                    self._collection_ready = True
                    self.logger.info("Auto-initialization successful")
                else:
                    self.logger.error("Auto-initialization failed")
                    return []
        
        if not self.vectorizer or self.paper_embeddings is None:
            self.logger.error("TF-IDF search system not initialized")
            return []
        
        try:
            self.logger.info(f"Searching papers for: '{query}'")
            
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.paper_embeddings).flatten()
            
            # Get relevant papers
            relevant_papers = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.config.similarity_threshold:
                    metadata = self.paper_metadata[i]
                    relevant_papers.append({
                        'similarity': float(similarity),
                        'title': metadata['title'],
                        'journal': metadata['journal'],
                        'publication_date': metadata['publication_date'],
                        'authors': metadata['std_authors'].split('; ') if metadata['std_authors'] else [],
                        'pmid': metadata['pmid']
                    })
            
            # Sort by similarity
            relevant_papers.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.logger.info(f"Found {len(relevant_papers)} relevant papers")
            
            if not relevant_papers:
                return []
            
            # Group by author and analyze
            return self._analyze_author_relevance(relevant_papers, query)
            
        except Exception as e:
            self.logger.error(f"TF-IDF search failed: {e}")
            return []
    
    def _analyze_author_relevance(self, relevant_papers: List[Dict], query: str) -> List[Dict]:
        """Analyze author expertise from relevant papers"""
        # Group papers by author
        author_papers = defaultdict(list)
        
        for paper in relevant_papers:
            for author in paper['authors']:
                if author.strip():
                    author_papers[author.strip()].append(paper)
        
        # Analyze each author
        author_analysis = []
        
        for author, papers in author_papers.items():
            relevant_count = len(papers)
            
            if relevant_count < self.config.min_relevant_papers:
                continue
            
            # Calculate metrics
            similarities = [p['similarity'] for p in papers]
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            
            # TF-IDF expertise score (adjusted for TF-IDF characteristics)
            # TF-IDF similarities are generally lower than neural embeddings
            volume_score = min(relevant_count / 8, 1.0)  # Scale for TF-IDF
            quality_score = min(avg_similarity * 2, 1.0)  # Boost TF-IDF scores
            high_quality_count = sum(1 for s in similarities if s > 0.3)
            excellence_bonus = high_quality_count * 0.15
            
            expertise_score = (
                volume_score * 0.4 +
                quality_score * 0.5 +
                excellence_bonus
            )
            
            # Get total papers
            total_papers = self._get_author_total_papers(author)
            focus_percentage = (relevant_count / max(total_papers, 1)) * 100
            
            # Sort papers by similarity
            papers.sort(key=lambda p: p['similarity'], reverse=True)
            
            author_analysis.append({
                'author': author,
                'expertise_score': float(expertise_score),
                'relevant_papers_count': int(relevant_count),
                'total_papers': int(total_papers),
                'focus_percentage': float(min(focus_percentage, 100.0)),
                'avg_similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'high_quality_count': int(high_quality_count),
                'best_papers': papers[:5]
            })
        
        # Sort by expertise score
        author_analysis.sort(key=lambda x: x['expertise_score'], reverse=True)
        
        # Get detailed author info
        final_results = []
        for author_data in author_analysis[:self.config.max_results]:
            author_info = self._get_author_details(author_data['author'])
            if author_info:
                result_data = {
                    'author': author_info,
                    'expertise_score': author_data['expertise_score'],
                    'relevant_papers_count': author_data['relevant_papers_count'],
                    'total_papers': author_data['total_papers'],
                    'focus_percentage': author_data['focus_percentage'],
                    'avg_similarity': author_data['avg_similarity'],
                    'max_similarity': author_data['max_similarity'],
                    'high_quality_count': author_data['high_quality_count'],
                    'best_papers': author_data['best_papers'][:3]
                }
                final_results.append(result_data)
        
        self.logger.info(f"Returning {len(final_results)} qualified researchers")
        return final_results
    
    def _get_author_total_papers(self, standardized_name: str) -> int:
        """Get total papers for author"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT total_publications 
                    FROM authors 
                    WHERE standardized_name = ?
                ''', (standardized_name,))
                result = cursor.fetchone()
                return result[0] if result and result[0] else 0
        except:
            return 0
    
    def _get_author_details(self, standardized_name: str) -> Optional[Dict]:
        """Get author details"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name, standardized_name, department, total_publications
                    FROM authors 
                    WHERE standardized_name = ?
                    LIMIT 1
                ''', (standardized_name,))
                
                result = cursor.fetchone()
                if result:
                    name, std_name, department, total_pubs = result
                    return {
                        'name': str(name or ''),
                        'standardized_name': str(std_name or ''),
                        'department': str(department or ''),
                        'total_publications': int(total_pubs or 0),
                        'llm_themes': {
                            'themes': [],
                            'summary': 'TF-IDF keyword search - fast and reliable',
                            'llm_analyzed': False,
                            'available': False
                        }
                    }
        except Exception as e:
            self.logger.error(f"Failed to get author details: {e}")
        return None

# Initialize FastAPI
app = FastAPI(title="MGH/BWH/MGB Researcher Query Tool (TF-IDF)", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Initialize search system
try:
    paper_search = TFIDFPaperSearch(QueryConfig())
    print("✓ TF-IDF Paper Search initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize TF-IDF Paper Search: {e}")
    paper_search = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/initialize")
async def initialize_paper_db():
    """Initialize the TF-IDF database (optional - for local development)"""
    try:
        if not paper_search:
            return InitializeResponse(success=False, error="Search system not initialized")
        
        # Check if we're in deployment
        if os.environ.get("PORT"):
            return InitializeResponse(
                success=False,
                error="Database initialization disabled in deployment. Please build locally and include in repository."
            )
        
        success = paper_search.create_paper_database()
        if success:
            return InitializeResponse(success=True, message="TF-IDF search database created successfully")
        else:
            return InitializeResponse(success=False, error="Failed to create search database")
    except Exception as e:
        return InitializeResponse(success=False, error=str(e))

@app.post("/search")
async def search(search_request: SearchRequest):
    """Search for researchers based on query using TF-IDF"""
    try:
        if not paper_search:
            raise HTTPException(status_code=503, detail="Search system not initialized")
        
        query = search_request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        results = paper_search.search_with_smart_filtering(query)
        
        return {
            'success': True,
            'query': query,
            'results': results,
            'total_found': len(results),
            'search_type': 'tfidf_keyword_search'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    collection_ready = paper_search._collection_ready if paper_search else False
    is_deployment = bool(os.environ.get("PORT"))
    
    # Check if we have pre-built TF-IDF files
    has_prebuilt = (
        Path(QueryConfig().vectorizer_path).exists() and
        Path(QueryConfig().embeddings_path).exists() and
        Path(QueryConfig().metadata_path).exists()
    )
    
    return {
        "status": "healthy",
        "search_system_available": paper_search is not None,
        "search_type": "tfidf",
        "database_path": QueryConfig().db_path,
        "database_exists": os.path.exists(QueryConfig().db_path),
        "collection_ready": collection_ready,
        "is_deployment": is_deployment,
        "has_prebuilt_database": has_prebuilt,
        "auto_initialization": "disabled_in_deployment" if is_deployment else "enabled_locally"
    }