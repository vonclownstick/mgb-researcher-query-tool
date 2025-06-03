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

# New imports for SentenceTransformers
from sentence_transformers import SentenceTransformer
# Removed: import pynndescent # No longer needed
from sklearn.metrics.pairwise import cosine_similarity # Still used for calculating similarity from embeddings

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
    # Updated paths for SentenceTransformer model and embeddings
    model_path: str = "./all-MiniLM-L6-v2_model"
    embeddings_path: str = "./all-MiniLM-L6-v2_embeddings.pkl"
    metadata_path: str = "./paper_metadata.json"
    max_results: int = 5
    similarity_threshold: float = 0.55  # Adjusted for SentenceTransformer embeddings (typically higher than TF-IDF)
    min_relevant_papers: int = 1  # Allow single-paper authors

class EmbeddingPaperSearch: # Class name remains the same
    """Sentence-Transformer embedding based paper search - reliable and fast"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._setup_logging()
        self.embedding_model = None
        self.paper_embeddings = None
        self.paper_metadata = None
        # Removed: self.pynndescent_index = None # PyNNDescent index no longer needed
        self._collection_ready = False
        self._load_existing_data()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_existing_data(self):
        """Load existing SentenceTransformer model and embeddings"""
        try:
            self.logger.info(f"Looking for embedding files:")
            self.logger.info(f"  Model directory: {self.config.model_path} - {'EXISTS' if Path(self.config.model_path).exists() else 'MISSING'}")
            self.logger.info(f"  Embeddings: {self.config.embeddings_path} - {'EXISTS' if Path(self.config.embeddings_path).exists() else 'MISSING'}")
            self.logger.info(f"  Metadata: {self.config.metadata_path} - {'EXISTS' if Path(self.config.metadata_path).exists() else 'MISSING'}")
            
            # List all files in current directory for debugging
            current_files = list(Path('.').glob('*'))
            self.logger.info(f"Files in current directory: {[f.name for f in current_files if f.is_file()]}")
            
            if (Path(self.config.model_path).exists() and 
                Path(self.config.embeddings_path).exists() and 
                Path(self.config.metadata_path).exists()):
                
                self.logger.info("All embedding files found - loading...")
                
                # Load SentenceTransformer model
                try:
                    self.embedding_model = SentenceTransformer(self.config.model_path)
                    self.logger.info("✅ SentenceTransformer model loaded successfully")
                except Exception as e:
                    self.logger.error(f"Failed to load SentenceTransformer model from {self.config.model_path}: {e}")
                    return False
                
                # Load embeddings
                try:
                    with open(self.config.embeddings_path, 'rb') as f:
                        self.paper_embeddings = pickle.load(f)
                    self.logger.info(f"✅ Embeddings loaded - shape: {self.paper_embeddings.shape}")
                except Exception as e:
                    self.logger.error(f"Failed to load embeddings from {self.config.embeddings_path}: {e}")
                    return False
                
                # Load metadata
                try:
                    with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                        self.paper_metadata = json.load(f)
                    self.logger.info(f"✅ Metadata loaded - {len(self.paper_metadata)} papers")
                except Exception as e:
                    self.logger.error(f"Failed to load metadata from {self.config.metadata_path}: {e}")
                    return False
                
                # Verify data consistency
                if (len(self.paper_metadata) != self.paper_embeddings.shape[0]):
                    self.logger.error(f"Data size mismatch: metadata={len(self.paper_metadata)}, embeddings={self.paper_embeddings.shape[0]}")
                    return False
                
                # Removed: Building PyNNDescent index
                # self.logger.info("Building PyNNDescent index...")
                # try:
                #     self.pynndescent_index = pynndescent.NNDescent(...)
                #     self.pynndescent_index.prepare()
                #     self.logger.info("✅ PyNNDescent index built successfully")
                # except Exception as e:
                #     self.logger.error(f"Failed to build PyNNDescent index: {e}")
                #     return False

                self._collection_ready = True
                self.logger.info(f"🚀 Embedding search system ready with {len(self.paper_metadata)} papers")
                return True
            else:
                self.logger.info("❌ Embedding files not found - please run create_embeddings.py first.")
                
        except Exception as e:
            self.logger.error(f"Failed to load existing embedding data: {e}") # Removed "or build index"
            import traceback
            self.logger.error(f"Load traceback: {traceback.format_exc()}")
        
        return False
    
    def search_with_smart_filtering(self, query: str) -> List[Dict]:
        """Search using embedding similarity with auto-initialization"""
        # Auto-initialize if needed (though typically pre-built in deployment)
        if not self._collection_ready:
            self.logger.error("Search system not initialized. Please ensure embedding files are present.")
            return []
        
        if self.embedding_model is None or self.paper_embeddings is None: # Changed condition
            self.logger.error("Embedding model or paper embeddings not available.")
            return []
        
        try:
            self.logger.info(f"Searching papers for: '{query}'")
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            # Normalize query embedding (important for cosine similarity)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # --- Brute-force Cosine Similarity Calculation ---
            # Calculate cosine similarity between the query embedding and all paper embeddings
            # The result is a 1D array of similarities
            similarities = cosine_similarity(query_embedding, self.paper_embeddings).flatten()
            
            # Get the indices of the top N results based on similarity
            # Use argpartition for efficiency if only top_N is needed, then sort those N
            # For 2799, full argsort is also fine, but argpartition is more scalable
            top_indices_unsorted = np.argpartition(similarities, -self.config.max_results * 10)[-self.config.max_results * 10:] # Get more than max_results
            
            # Sort these top indices by their similarity values
            top_indices = top_indices_unsorted[np.argsort(similarities[top_indices_unsorted])][::-1]
            # ----------------------------------------------------
            
            relevant_papers = []
            for i, idx in enumerate(top_indices):
                similarity = similarities[idx]
                if similarity >= self.config.similarity_threshold:
                    metadata = self.paper_metadata[idx]
                    relevant_papers.append({
                        'similarity': float(similarity),
                        'title': metadata['title'],
                        'journal': metadata['journal'],
                        'publication_date': metadata['publication_date'],
                        'authors': metadata['std_authors'].split('; ') if metadata['std_authors'] else [],
                        'pmid': metadata['pmid']
                    })
            
            # Sort by similarity (important as argpartition doesn't guarantee sorted order of the top N)
            relevant_papers.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.logger.info(f"Found {len(relevant_papers)} relevant papers (threshold: {self.config.similarity_threshold})")
            
            # Show top papers for debugging
            for i, paper in enumerate(relevant_papers[:5]):
                self.logger.info(f"  {i+1}. {paper['title'][:60]}... (sim: {paper['similarity']:.3f})")
            
            if not relevant_papers:
                self.logger.info("No papers above similarity threshold - try lowering threshold or different keywords")
                return []
            
            # Group by author and analyze
            return self._analyze_author_relevance(relevant_papers, query)
            
        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            import traceback
            self.logger.error(f"Search traceback: {traceback.format_exc()}")
            return []
    
    def _analyze_author_relevance(self, relevant_papers: List[Dict], query: str) -> List[Dict]:
        """Analyze author expertise from relevant papers"""
        # Group papers by author
        author_papers = defaultdict(list)
        
        for paper in relevant_papers:
            for author in paper['authors']:
                if author.strip():
                    author_papers[author.strip()].append(paper)
        
        self.logger.info(f"Authors with papers: {len(author_papers)}")
        for author, papers in list(author_papers.items())[:10]:  # Show first 10
            self.logger.info(f"  {author}: {len(papers)} papers (avg sim: {sum(p['similarity'] for p in papers)/len(papers):.3f})")
        
        # Analyze each author
        author_analysis = []
        
        for author, papers in author_papers.items():
            relevant_count = len(papers)
            
            if relevant_count < self.config.min_relevant_papers:
                self.logger.debug(f"Skipping {author}: only {relevant_count} papers (need {self.config.min_relevant_papers})")
                continue
            
            # Calculate metrics
            similarities = [p['similarity'] for p in papers]
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            
            # Expertise score for SentenceTransformer embeddings
            # These similarities are typically higher than TF-IDF
            volume_score = min(relevant_count / 8, 1.0)  # Scale based on number of relevant papers
            quality_score = min(avg_similarity * 2, 1.0)  # Weight by average similarity
            high_quality_count = sum(1 for s in similarities if s > 0.7) # Count papers with very high similarity
            excellence_bonus = high_quality_count * 0.15 # Bonus for highly relevant papers
            
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
            
            self.logger.info(f"Qualified: {author} - {expertise_score:.3f} score ({relevant_count} papers, avg sim: {avg_similarity:.3f})")
        
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
        except Exception as e:
            self.logger.error(f"Error getting total papers for {standardized_name}: {e}")
            return 0
    
    def _get_author_details(self, standardized_name: str) -> Optional[Dict]:
        """Get author details including LLM-generated themes"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic author info
                cursor.execute('''
                    SELECT name, standardized_name, department, total_publications
                    FROM authors 
                    WHERE standardized_name = ?
                    LIMIT 1
                ''', (standardized_name,))
                
                result = cursor.fetchone()
                if result:
                    name, std_name, department, total_pubs = result
                    
                    # Get LLM-generated themes if available
                    cursor.execute('''
                        SELECT themes, summary
                        FROM author_themes at
                        JOIN authors a ON at.author_id = a.id
                        WHERE a.standardized_name = ?
                        LIMIT 1
                    ''', (standardized_name,))
                    
                    theme_result = cursor.fetchone()
                    
                    if theme_result and theme_result[0]:
                        # Parse themes from pipe-separated string
                        themes_text, summary = theme_result
                        theme_list = themes_text.split('|') if themes_text else []
                        
                        # Clean and validate themes
                        clean_themes = []
                        for theme in theme_list:
                            theme = theme.strip()
                            if theme and len(theme) > 5:  # Basic validation
                                clean_themes.append(theme)
                        
                        llm_themes = {
                            'themes': clean_themes,
                            'summary': summary or 'LLM-generated research themes available',
                            'llm_analyzed': True,
                            'available': True
                        }
                    else:
                        # No themes generated yet
                        llm_themes = {
                            'themes': [],
                            'summary': 'Research themes not yet generated for this investigator',
                            'llm_analyzed': False,
                            'available': False
                        }
                    
                    return {
                        'name': str(name or ''),
                        'standardized_name': str(std_name or ''),
                        'department': str(department or ''),
                        'total_publications': int(total_pubs or 0),
                        'llm_themes': llm_themes
                    }
        except Exception as e:
            self.logger.error(f"Failed to get author details for {standardized_name}: {e}")
        return None

# Initialize FastAPI
app = FastAPI(title="MGH/BWH/MGB Researcher Query Tool (Embeddings)", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Initialize search system
try:
    paper_search = EmbeddingPaperSearch(QueryConfig())
    print("✓ Embedding Paper Search initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Embedding Paper Search: {e}")
    paper_search = None

# For local development
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting Embedding Paper Search System...")
    print(f"📍 Access: http://localhost:{port}")
    
    if paper_search and paper_search._collection_ready:
        print(f"✅ Search ready with {len(paper_search.paper_metadata)} papers")
    else:
        print("⚠️  Search database needs pre-built embedding files. Run create_embeddings.py first.")
    
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/initialize")
async def initialize_paper_db():
    """Attempt to initialize the embedding database (will only load pre-built files)"""
    try:
        if not paper_search:
            return InitializeResponse(success=False, error="Search system not initialized")
        
        # This endpoint will now just try to load the pre-built embeddings
        success = paper_search._load_existing_data()
        if success:
            return InitializeResponse(success=True, message="Embedding search database loaded successfully")
        else:
            return InitializeResponse(success=False, error="Failed to load search database. Ensure pre-built files exist.")
    except Exception as e:
        return InitializeResponse(success=False, error=str(e))

@app.post("/search")
async def search(search_request: SearchRequest):
    """Search for researchers based on query using embeddings"""
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
            'search_type': 'embedding_semantic_search'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adjust_thresholds")
async def adjust_thresholds(
    similarity_threshold: float = 0.55, # Default adjusted for embeddings
    min_relevant_papers: int = 1
):
    """Temporarily adjust search thresholds for testing"""
    if paper_search:
        old_sim = paper_search.config.similarity_threshold
        old_min = paper_search.config.min_relevant_papers
        
        paper_search.config.similarity_threshold = similarity_threshold
        paper_search.config.min_relevant_papers = min_relevant_papers
        
        return {
            "success": True,
            "message": f"Thresholds updated",
            "old_values": {
                "similarity_threshold": old_sim,
                "min_relevant_papers": old_min
            },
            "new_values": {
                "similarity_threshold": similarity_threshold,
                "min_relevant_papers": min_relevant_papers
            }
        }
    else:
        return {"success": False, "error": "Search system not available"}

@app.get("/debug")
async def debug_files():
    """Debug endpoint to check file status"""
    config = QueryConfig()
    
    def get_file_info(path):
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            return {"exists": True, "size_mb": round(size_mb, 2)}
        else:
            return {"exists": False, "size_mb": 0}
            
    def get_dir_info(path):
        p = Path(path)
        if p.is_dir():
            size_bytes = sum(f.stat().st_size for f in p.glob('**/*') if f.is_file())
            return {"exists": True, "size_mb": round(size_bytes / (1024 * 1024), 2)}
        else:
            return {"exists": False, "size_mb": 0}
    
    # List all files in current directory
    current_files = []
    for f in Path('.').iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            current_files.append({
                "name": f.name,
                "size_mb": round(size_mb, 2)
            })
    
    return {
        "embedding_files": {
            "model_directory": get_dir_info(config.model_path),
            "embeddings_pkl": get_file_info(config.embeddings_path),
            "metadata_json": get_file_info(config.metadata_path)
        },
        "sqlite_database": get_file_info(config.db_path),
        "all_files": sorted(current_files, key=lambda x: x['name']),
        "search_system_status": {
            "initialized": paper_search is not None,
            "collection_ready": paper_search._collection_ready if paper_search else False,
            "has_embedding_model": paper_search.embedding_model is not None if paper_search else False,
            "has_embeddings": paper_search.paper_embeddings is not None if paper_search else False,
            "has_metadata": paper_search.paper_metadata is not None if paper_search else False,
            # Removed: "has_pynndescent_index"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    collection_ready = paper_search._collection_ready if paper_search else False
    is_deployment = bool(os.environ.get("PORT"))
    
    # Check if we have pre-built embedding files
    has_prebuilt = (
        Path(QueryConfig().model_path).exists() and
        Path(QueryConfig().embeddings_path).exists() and
        Path(QueryConfig().metadata_path).exists()
    )
    
    return {
        "status": "healthy",
        "search_system_available": paper_search is not None,
        "search_type": "embedding_semantic_search_brute_force", # Updated search type
        "database_path": QueryConfig().db_path,
        "database_exists": os.path.exists(QueryConfig().db_path),
        "collection_ready": collection_ready,
        "is_deployment": is_deployment,
        "has_prebuilt_database": has_prebuilt,
        "auto_initialization": "disabled_in_deployment" # Embeddings are always pre-built
    }