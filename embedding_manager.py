import chromadb
from sentence_transformers import SentenceTransformer
import os

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_dir="./chroma_db"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Sentence transformer model (default: all-MiniLM-L6-v2)
            persist_dir: Directory to persist ChromaDB (default: ./chroma_db)
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB with persistence
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        
    def get_or_create_collection(self, name="pdf_embeddings"):
        """Get or create a collection in ChromaDB."""
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embeddings
        """
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def store_chunks(self, chunks, collection_name="pdf_embeddings", metadata=None):
        """
        Store text chunks with their embeddings in ChromaDB.
        
        Args:
            chunks: List of text chunks
            collection_name: Name of the collection to store in
            metadata: Optional metadata dict to attach to all chunks
        
        Returns:
            Collection object
        """

        # Always delete first so we start clean for every new PDF
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass  # Didn't exist yet — fine
        
        # Create fresh, not get_or_create
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.generate_embeddings(chunks)
        
        # Prepare data for storage
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in chunks]
        
        # Store in ChromaDB
        print("Storing embeddings in ChromaDB...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"Successfully stored {len(chunks)} chunks with embeddings")
        return collection
    
    def search(self, query, collection_name="pdf_embeddings", top_k=5):
        """
        Search for similar chunks using semantic search.
        
        Args:
            query: Query text
            collection_name: Name of the collection to search in
            top_k: Number of top results to return
        
        Returns:
            List of matching chunks with similarity scores
        """
        collection = self.get_or_create_collection(collection_name)
        
        # Manually embed the query using the same model for consistency
        query_embeddings = self.generate_embeddings([query])
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        
        if results['documents']:
            matches = []
            for i, doc in enumerate(results['documents'][0]):
                matches.append({
                    'text': doc,
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
            return matches
        
        return []
    
    def delete_collection(self, collection_name="pdf_embeddings"):
        """Delete a collection from ChromaDB."""
        self.client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted")

if __name__ == "__main__":
    # Example usage
    manager = EmbeddingManager()
    
    # Example chunks
    sample_chunks = [
        "This is the first text chunk about machine learning.",
        "Natural language processing is a subfield of AI.",
        "Deep learning models require large amounts of data.",
        "Vector databases are efficient for similarity search.",
        "Embeddings capture semantic meaning of text."
    ]
    
    # Store chunks with embeddings
    manager.store_chunks(sample_chunks, metadata={"source": "example"})
    
    # Search
    query = "Tell me about AI and machine learning"
    results = manager.search(query, top_k=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Similarity Distance: {result['distance']:.4f}")
        print(f"   Text: {result['text']}")
