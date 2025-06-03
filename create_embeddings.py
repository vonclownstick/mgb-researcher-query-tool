import sqlite3
import logging
import os
import pickle
import json
import numpy as np # Added for embedding normalization
from pathlib import Path
from sentence_transformers import SentenceTransformer # Changed from TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Kept for similarity calculation
import sys
from tqdm import tqdm # Added for progress bar in embedding generation

def build_embedding_database(): # Renamed function from build_tfidf_database
    """Build Sentence-Transformer embedding database locally for deployment"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Configuration
    db_path = "mgh_bwh_psychiatry_research.db"
    model_save_path = "./all-MiniLM-L6-v2_model" # Changed path for SentenceTransformer model directory
    embeddings_path = "./all-MiniLM-L6-v2_embeddings.pkl" # Changed path for embeddings
    metadata_path = "./paper_metadata.json"
    
    # Check if SQLite database exists
    if not Path(db_path).exists():
        logger.error(f"SQLite database not found at {db_path}")
        logger.error("Please ensure your mgh_bwh_psychiatry_research.db file is in the current directory")
        return False
    
    logger.info("🔨 Building embedding search database using all-MiniLM-L6-v2...")
    
    try:
        # Get publications from SQLite
        logger.info("Reading publications from SQLite database...")
        with sqlite3.connect(db_path) as conn:
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
            ''')
            publications = cursor.fetchall()
        
        if not publications:
            logger.error("No publications found in database")
            return False
        
        logger.info(f"Processing {len(publications)} publications...")
        
        # Process documents
        documents = []
        metadata = []
        
        def clean_text(text):
            if not text:
                return ""
            text = str(text).replace('\x00', ' ').replace('\r', ' ').replace('\n', ' ')
            text = ' '.join(text.split())
            if len(text) > 3000:
                text = text[:3000]
            return text.strip()
        
        valid_count = 0
        for pub in publications:
            try:
                pub_id, pmid, title, abstract, journal, pub_date, authors, std_authors, departments = pub
                
                title_clean = clean_text(title)
                abstract_clean = clean_text(abstract)
                
                if not title_clean and not abstract_clean:
                    continue
                
                # Create searchable text from title and abstract
                # Giving more weight to title by repeating it
                if title_clean and abstract_clean:
                    search_text = f"{title_clean} {title_clean} {abstract_clean}"
                else:
                    search_text = title_clean or abstract_clean
                
                if len(search_text.strip()) < 30: # Ensure meaningful text
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
                    logger.info(f"Processed {valid_count} publications...")
            except Exception as e:
                logger.warning(f"Error processing publication: {e}")
                continue
        
        if not documents:
            logger.error("No valid documents to process for embeddings")
            return False
            
        logger.info(f"Generating embeddings for {len(documents)} documents using all-MiniLM-L6-v2...")
        
        # Load SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity (dot product of normalized vectors)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        logger.info(f"Embeddings generated with shape: {embeddings.shape}")
        
        # Save the model, embeddings, and metadata
        logger.info(f"Saving embedding model to {model_save_path}...")
        model.save(model_save_path) # Save the whole model directory
        
        logger.info(f"Saving embeddings to {embeddings_path}...")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Embedding database creation complete. Running a quick test...")
        
        # Quick test
        test_query = "machine learning applications in medicine"
        logger.info(f"Loading SentenceTransformer model from {model_save_path} for testing...")
        test_model = SentenceTransformer(model_save_path) # Load the saved model for testing
        
        logger.info(f"Loading embeddings from {embeddings_path} for testing...")
        with open(embeddings_path, 'rb') as f:
            test_embeddings = pickle.load(f)
            
        logger.info(f"Loading metadata from {metadata_path} for testing...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            test_metadata = json.load(f)

        query_embedding = test_model.encode([test_query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True) # Normalize query embedding
        
        similarities = cosine_similarity(query_embedding, test_embeddings).flatten()
        
        # Find top results
        top_indices = similarities.argsort()[-3:][::-1]
        logger.info(f"Test query: '{test_query}'")
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0.1: # Only show results with a reasonable similarity score
                logger.info(f"   {i+1}. {test_metadata[idx]['title'][:70]}... (score: {similarities[idx]:.3f})")
        
        logger.info("\n🚀 Ready for deployment!")
        logger.info("Files created:")
        logger.info(f"  - {model_save_path} (directory)")
        logger.info(f"  - {embeddings_path}")
        logger.info(f"  - {metadata_path}")
        logger.info("\nNext steps:")
        logger.info(f"1. Ensure '{model_save_path}' directory is deployed along with '{embeddings_path}' and '{metadata_path}'")
        logger.info("2. You can now use the embedding search application to query the database.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build embedding database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔨 Building Sentence-Transformer embedding search database for fast deployment...\n")
    
    success = build_embedding_database() # Renamed function call
    
    if success:
        print("\n✅ Embedding search database creation completed successfully!")
    else:
        print("\n❌ Embedding search database creation failed.")