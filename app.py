#!/usr/bin/env python3

"""
Smart Paper Search with Author Filtering - FastAPI Version for Render
Searches actual papers but filters authors intelligently
"""

import sqlite3
import logging
import os
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from pathlib import Path
import re
from collections import defaultdict, Counter

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
    chroma_path: str = "./paper_search_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 5
    similarity_threshold: float = 0.4  # Minimum paper similarity
    min_relevant_papers: int = 2  # Minimum number of relevant papers
    high_similarity_threshold: float = 0.7  # Bonus threshold for excellent matches

class SmartPaperSearch:
    """Search papers but filter authors intelligently"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._setup_logging()
        self._initialize_models()
        self._setup_vector_db()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_models(self):
        try:
            self.logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.logger.info("Embedding model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _setup_vector_db(self):
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_path)
            try:
                self.collection = self.chroma_client.get_collection("paper_abstracts")
                self.logger.info("Connected to existing paper collection")
            except:
                self.collection = None
                self.logger.info("Paper collection not found, will create when needed")
        except Exception as e:
            self.logger.error(f"Failed to setup ChromaDB: {e}")
            self.collection = None
    
    def create_paper_database(self):
        """Create searchable database of paper abstracts"""
        try:
            self.logger.info("Creating paper search database...")
            
            # Delete existing collection
            try:
                self.chroma_client.delete_collection("paper_abstracts")
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="paper_abstracts",
                metadata={"description": "Individual paper abstracts for precise search"}
            )
            
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
                ''')
                publications = cursor.fetchall()
            
            self.logger.info(f"Processing {len(publications)} publications...")
            
            # Prepare documents for embedding
            documents = []
            metadatas = []
            ids = []
            
            for pub in publications:
                pub_id, pmid, title, abstract, journal, pub_date, authors, std_authors, departments = pub
                
                # Create rich searchable text
                title_clean = (title or '').strip()
                abstract_clean = (abstract or '').strip()
                
                # Weight title more heavily by including it multiple times
                search_text = f"Title: {title_clean}\n{title_clean}\nAbstract: {abstract_clean}\nJournal: {journal or ''}"
                
                documents.append(search_text)
                metadatas.append({
                    'pub_id': str(pub_id),
                    'pmid': pmid or '',
                    'title': title_clean,
                    'journal': journal or '',
                    'publication_date': pub_date or '',
                    'authors': authors or '',
                    'std_authors': std_authors or '',
                    'departments': departments or ''
                })
                ids.append(f"paper_{pub_id}")
            
            # Add to ChromaDB in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metadata = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            self.logger.info(f"Added {len(documents)} papers to search database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create paper database: {e}")
            return False
    
    def search_with_smart_filtering(self, query: str) -> List[Dict]:
        """Search papers with intelligent author filtering"""
        if not self.collection:
            return []
        
        try:
            self.logger.info(f"Searching papers for: '{query}'")
            
            # Search paper abstracts directly
            results = self.collection.query(
                query_texts=[query],
                n_results=100,  # Get many papers to filter properly
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Convert distances to similarities
            similarities = []
            for distance in results['distances'][0]:
                similarity = max(0, min(1, 1 - (distance / 2)))
                similarities.append(similarity)
            
            # Filter papers by similarity threshold
            relevant_papers = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.config.similarity_threshold:
                    metadata = results['metadatas'][0][i]
                    relevant_papers.append({
                        'similarity': similarity,
                        'title': metadata['title'],
                        'journal': metadata['journal'],
                        'publication_date': metadata['publication_date'],
                        'authors': metadata['std_authors'].split('; ') if metadata['std_authors'] else [],
                        'pmid': metadata['pmid']
                    })
            
            self.logger.info(f"Found {len(relevant_papers)} relevant papers")
            
            if not relevant_papers:
                return []
            
            # Group papers by author and apply smart filtering
            author_analysis = self._analyze_author_relevance(relevant_papers, query)
            
            # Get detailed author info and rank
            final_results = []
            for author_data in author_analysis:
                author_info = self._get_author_details(author_data['author'])
                if author_info:
                    # Ensure all numeric values are JSON-safe
                    result_data = {
                        'author': author_info,
                        'expertise_score': float(author_data.get('expertise_score', 0.0)),
                        'relevant_papers_count': int(author_data.get('relevant_papers_count', 0)),
                        'total_papers': int(author_data.get('total_papers', 0)),
                        'focus_percentage': float(author_data.get('focus_percentage', 0.0)),
                        'avg_similarity': float(author_data.get('avg_similarity', 0.0)),
                        'max_similarity': float(author_data.get('max_similarity', 0.0)),
                        'high_quality_count': int(author_data.get('high_quality_count', 0)),
                        'best_papers': author_data.get('best_papers', [])[:3]
                    }
                    
                    # Debug logging to see what we're sending
                    self.logger.info(f"Final result for {author_info['name']}: "
                                   f"focus_percentage={result_data['focus_percentage']}, "
                                   f"llm_themes_available={bool(author_info.get('llm_themes', {}).get('available', False))}")
                    
                    final_results.append(result_data)
            
            # Sort by expertise score
            final_results.sort(key=lambda x: x['expertise_score'], reverse=True)
            
            self.logger.info(f"Returning {len(final_results)} qualified researchers")
            return final_results[:self.config.max_results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _analyze_author_relevance(self, relevant_papers: List[Dict], query: str) -> List[Dict]:
        """Analyze which authors have the most expertise (absolute, not relative)"""
        
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
            
            # Only consider authors with minimum relevant papers
            if relevant_count < self.config.min_relevant_papers:
                continue
            
            # Calculate quality metrics
            similarities = [p['similarity'] for p in papers]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            max_similarity = max(similarities) if similarities else 0.0
            
            # Count high-quality papers (excellent matches)
            high_quality_papers = [p for p in papers if p['similarity'] >= self.config.high_similarity_threshold]
            high_quality_count = len(high_quality_papers)
            
            # Calculate expertise score based on volume and quality
            # This rewards both many relevant papers AND high-quality matches
            volume_score = min(relevant_count / 10, 1.0)  # Scale 0-1, max at 10 papers
            quality_score = avg_similarity  # 0-1 based on average similarity
            excellence_bonus = high_quality_count * 0.1  # Bonus for excellent papers
            
            # Combined expertise score (no focus percentage penalty!)
            expertise_score = (
                volume_score * 0.4 +      # Reward many relevant papers
                quality_score * 0.5 +     # Reward high similarity
                excellence_bonus          # Bonus for excellent matches
            )
            
            # FIX: Get total papers for display with error handling
            total_papers = self._get_author_total_papers(author)
            
            # FIX: Ultra-safe focus percentage calculation
            try:
                if total_papers > 0 and relevant_count > 0:
                    focus_percentage = (float(relevant_count) / float(total_papers)) * 100.0
                    # Clamp between 0 and 100
                    focus_percentage = max(0.0, min(100.0, focus_percentage))
                else:
                    focus_percentage = 0.0
                
                # Final safety check
                if not isinstance(focus_percentage, (int, float)) or np.isnan(focus_percentage) or np.isinf(focus_percentage):
                    focus_percentage = 0.0
                    
                # Round to 1 decimal place
                focus_percentage = round(float(focus_percentage), 1)
                
            except Exception as e:
                self.logger.warning(f"Focus percentage calculation failed for {author}: {e}")
                focus_percentage = 0.0
            
            # Sort papers by similarity
            papers.sort(key=lambda p: p['similarity'], reverse=True)
            
            author_analysis.append({
                'author': author,
                'expertise_score': float(expertise_score),
                'relevant_papers_count': int(relevant_count),
                'total_papers': int(total_papers),
                'focus_percentage': focus_percentage,  # Already safe float
                'avg_similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'high_quality_count': int(high_quality_count),
                'best_papers': papers[:5]
            })
        
        # Sort by expertise score (absolute expertise, not focus)
        author_analysis.sort(key=lambda x: x['expertise_score'], reverse=True)
        
        self.logger.info(f"Qualified authors: {len(author_analysis)}")
        for i, author_data in enumerate(author_analysis[:5]):
            self.logger.info(f"  {i+1}. {author_data['author']}: {author_data['expertise_score']:.3f} "
                           f"({author_data['relevant_papers_count']} papers, "
                           f"avg {author_data['avg_similarity']:.2f} similarity, "
                           f"{author_data['high_quality_count']} excellent)")
        
        return author_analysis
    
    def _get_author_total_papers(self, standardized_name: str) -> int:
        """Get total number of papers for an author with error handling"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT total_publications 
                    FROM authors 
                    WHERE standardized_name = ?
                ''', (standardized_name,))
                result = cursor.fetchone()
                total_pubs = result[0] if result and result[0] is not None else 0
                
                # FIX: Ensure we return a valid integer
                if isinstance(total_pubs, (int, float)) and not np.isnan(total_pubs):
                    return max(0, int(total_pubs))
                else:
                    return 0
                    
        except Exception as e:
            self.logger.warning(f"Failed to get total papers for {standardized_name}: {e}")
            return 0
    
    def _get_author_details(self, standardized_name: str) -> Optional[Dict]:
        """Get author details from database including LLM themes if available"""
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
                    last_name = name.split(',')[0].strip() if ',' in name else name.split()[-1]
                    
                    # Ensure total_publications is JSON-safe
                    total_publications = int(total_pubs) if total_pubs is not None else 0
                    
                    # Get LLM themes - this always returns a valid dict
                    themes = self._get_llm_themes_if_available(standardized_name)
                    
                    author_details = {
                        'name': str(name or ''),
                        'last_name': str(last_name or ''),
                        'standardized_name': str(std_name or ''),
                        'department': str(department or ''),
                        'total_publications': total_publications,
                        'llm_themes': themes  # Always a complete dict now
                    }
                    
                    # Debug log to verify structure
                    self.logger.debug(f"Author details for {name}: {author_details['llm_themes']}")
                    
                    return author_details
            
        except Exception as e:
            self.logger.error(f"Failed to get author details for {standardized_name}: {e}")
        
        return None
    
    def _get_llm_themes_if_available(self, standardized_name: str) -> Dict:
        """Check if we have LLM-generated themes from previous analysis - FIXED VERSION"""
        try:
            # Check if the author expertise database exists
            expertise_db_path = "./author_expertise_db"
            if not Path(expertise_db_path).exists():
                self.logger.debug(f"Expertise database not found at {expertise_db_path}")
                return self._get_default_themes()
            
            # Connect to the expertise ChromaDB
            try:
                expertise_client = chromadb.PersistentClient(path=expertise_db_path)
                expertise_collection = expertise_client.get_collection("author_expertise")
                
                # FIX: Try multiple query approaches to find the author
                queries_to_try = [
                    standardized_name,
                    f"Dr. {standardized_name}",
                    standardized_name.replace(',', '').strip(),
                    # Try with just last name if comma-separated
                    standardized_name.split(',')[0].strip() if ',' in standardized_name else standardized_name
                ]
                
                for query in queries_to_try:
                    try:
                        # Search by text similarity first
                        results = expertise_collection.query(
                            query_texts=[query],
                            n_results=5,
                            include=["metadatas", "distances"]
                        )
                        
                        if results['metadatas'] and results['metadatas'][0]:
                            # Look for exact match in standardized names
                            for i, metadata in enumerate(results['metadatas'][0]):
                                if metadata.get('standardized_name') == standardized_name:
                                    return self._extract_themes_from_metadata(metadata)
                            
                            # If no exact match, try the closest one if distance is reasonable
                            if results['distances'] and results['distances'][0]:
                                if results['distances'][0][0] < 0.3:  # Close match
                                    metadata = results['metadatas'][0][0]
                                    return self._extract_themes_from_metadata(metadata)
                    
                    except Exception as e:
                        self.logger.debug(f"Query failed for '{query}': {e}")
                        continue
                
                # FIX: Try direct metadata search if available
                try:
                    all_results = expertise_collection.get(
                        include=["metadatas"],
                        where={"standardized_name": standardized_name}
                    )
                    
                    if all_results['metadatas']:
                        metadata = all_results['metadatas'][0]
                        return self._extract_themes_from_metadata(metadata)
                        
                except Exception as e:
                    self.logger.debug(f"Direct metadata search failed: {e}")
                
                self.logger.debug(f"No LLM themes found for {standardized_name}")
                return self._get_default_themes()
                
            except Exception as e:
                self.logger.debug(f"Failed to connect to expertise collection: {e}")
                return self._get_default_themes()
                
        except Exception as e:
            self.logger.debug(f"Failed to get LLM themes: {e}")
            return self._get_default_themes()
    
    def _extract_themes_from_metadata(self, metadata: Dict) -> Dict:
        """Extract themes from metadata with proper error handling"""
        try:
            # FIX: Handle different possible field names and formats
            themes_text = (
                metadata.get('research_themes', '') or 
                metadata.get('themes', '') or 
                metadata.get('expertise_themes', '') or
                ''
            )
            
            summary = (
                metadata.get('expertise_summary', '') or
                metadata.get('summary', '') or
                metadata.get('research_summary', '') or
                'Research expertise summary not available'
            )
            
            # Parse themes if available
            themes = []
            if themes_text:
                if '|' in themes_text:
                    themes = [t.strip() for t in themes_text.split('|') if t.strip()]
                elif ',' in themes_text:
                    themes = [t.strip() for t in themes_text.split(',') if t.strip()]
                elif ';' in themes_text:
                    themes = [t.strip() for t in themes_text.split(';') if t.strip()]
                else:
                    themes = [themes_text.strip()] if themes_text.strip() else []
            
            return {
                'themes': themes[:5],  # Limit to top 5 themes
                'summary': summary,
                'llm_analyzed': bool(metadata.get('llm_analyzed', False)),
                'available': True
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to extract themes from metadata: {e}")
            return self._get_default_themes()
    
    def _get_default_themes(self) -> Dict:
        """Return default theme structure when LLM analysis is not available"""
        return {
            'themes': [],
            'summary': 'Expertise analysis not yet available for this researcher',
            'llm_analyzed': False,
            'available': False
        }

# Initialize FastAPI Application
app = FastAPI(title="MGH/BWH/MGB Researcher Query Tool", version="1.0.0")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Initialize the search system
paper_search = SmartPaperSearch(QueryConfig())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/initialize")
async def initialize_paper_db():
    """Initialize the paper database"""
    try:
        success = paper_search.create_paper_database()
        if success:
            return InitializeResponse(
                success=True, 
                message="Paper search database created successfully"
            )
        else:
            return InitializeResponse(
                success=False, 
                error="Failed to create paper database"
            )
    except Exception as e:
        return InitializeResponse(success=False, error=str(e))

@app.post("/search")
async def search(search_request: SearchRequest):
    """Search for researchers based on query"""
    try:
        query = search_request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        results = paper_search.search_with_smart_filtering(query)
        
        # Clean results for JSON serialization
        cleaned_results = []
        for result in results:
            cleaned_result = {}
            for key, value in result.items():
                if key == 'author' and isinstance(value, dict):
                    # Ensure author dict is clean
                    cleaned_author = {}
                    for author_key, author_value in value.items():
                        if author_key == 'llm_themes':
                            # Ensure llm_themes is always a proper dict
                            if isinstance(author_value, dict):
                                cleaned_author[author_key] = {
                                    'themes': author_value.get('themes', []),
                                    'summary': str(author_value.get('summary', 'Expertise analysis not yet available')),
                                    'llm_analyzed': bool(author_value.get('llm_analyzed', False)),
                                    'available': bool(author_value.get('available', False))
                                }
                            else:
                                cleaned_author[author_key] = {
                                    'themes': [],
                                    'summary': 'Expertise analysis not yet available',
                                    'llm_analyzed': False,
                                    'available': False
                                }
                        else:
                            cleaned_author[author_key] = author_value
                    cleaned_result[key] = cleaned_author
                elif isinstance(value, float):
                    # Ensure floats are JSON-safe
                    if np.isnan(value) or np.isinf(value):
                        cleaned_result[key] = 0.0
                    else:
                        cleaned_result[key] = float(value)
                else:
                    cleaned_result[key] = value
            
            cleaned_results.append(cleaned_result)
        
        response_data = {
            'success': True,
            'query': query,
            'results': cleaned_results,
            'total_found': len(cleaned_results),
            'search_type': 'smart_paper_search'
        }
        
        # Debug log the first result to see what we're sending
        if cleaned_results:
            first_result = cleaned_results[0]
            print(f"DEBUG - First result focus_percentage: {first_result.get('focus_percentage', 'MISSING')}")
            print(f"DEBUG - First result llm_themes: {first_result.get('author', {}).get('llm_themes', 'MISSING')}")
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# For local development
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    db_path = Path(QueryConfig().db_path)
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Note: Database will be created when you click 'Initialize Paper Database'")
    
    print("Starting Smart Paper Search System...")
    print("This searches actual papers but filters authors intelligently")
    print(f"Access: http://localhost:{port}")
    
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)