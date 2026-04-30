import fitz  # PyMuPDF
from text_chunker import chunk_text
from embedding_manager import EmbeddingManager
from gemini_integration import GeminiIntegration
import sys

def load_pdf(file):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(file)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def load_pdf_chunks(file, chunk_size=1000, overlap=200):
    """
    Load PDF and return text chunks.
    
    Args:
        file: Path to PDF file
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    text = load_pdf(file)
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

def load_pdf_with_embeddings(file, chunk_size=1000, overlap=200, collection_name="pdf_embeddings"):
    """
    Load PDF, create chunks, and generate embeddings.
    
    Args:
        file: Path to PDF file
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        collection_name: Name of ChromaDB collection to store embeddings
    
    Returns:
        EmbeddingManager instance with stored embeddings
    """
    print(f"Loading PDF: {file}")
    chunks = load_pdf_chunks(file, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Created {len(chunks)} chunks")
    
    # Initialize embedding manager and store chunks
    manager = EmbeddingManager()
    metadata = {"source": file, "chunk_size": chunk_size, "overlap": overlap}
    manager.store_chunks(chunks, collection_name=collection_name, metadata=metadata)
    
    return manager, chunks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <pdf_file> [--chunks] [--embed] [--search QUERY] [--ask QUESTION] [--chunk-size SIZE] [--overlap OVERLAP]")
        print("\nOptions:")
        print("  --chunks             Display PDF text split into chunks")
        print("  --embed              Create and store embeddings in ChromaDB")
        print("  --search QUERY       Search embeddings for similar content")
        print("  --ask QUESTION       Ask a question about the PDF using Gemini with context")
        print("  --chunk-size SIZE    Size of each chunk (default: 1000)")
        print("  --overlap OVERLAP    Overlap between chunks (default: 200)")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    use_chunks = "--chunks" in sys.argv
    use_embed = "--embed" in sys.argv
    search_query = None
    ask_question = None
    
    # Parse optional parameters
    chunk_size = 1000
    overlap = 200
    if "--chunk-size" in sys.argv:
        idx = sys.argv.index("--chunk-size")
        if idx + 1 < len(sys.argv):
            chunk_size = int(sys.argv[idx + 1])
    if "--overlap" in sys.argv:
        idx = sys.argv.index("--overlap")
        if idx + 1 < len(sys.argv):
            overlap = int(sys.argv[idx + 1])
    if "--search" in sys.argv:
        idx = sys.argv.index("--search")
        if idx + 1 < len(sys.argv):
            search_query = sys.argv[idx + 1]
    if "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        if idx + 1 < len(sys.argv):
            ask_question = sys.argv[idx + 1]
    
    try:
        if use_embed or search_query or ask_question:
            manager, chunks = load_pdf_with_embeddings(pdf_file, chunk_size=chunk_size, overlap=overlap)
            
            if ask_question:
                # RAG: Search for relevant context and ask Gemini
                print(f"\nSearching PDF for context related to: '{ask_question}'")
                results = manager.search(ask_question, top_k=5)
                
                if results:
                    print(f"Found {len(results)} relevant sections. Asking Gemini...\n")
                    gemini = GeminiIntegration()
                    response = gemini.chat_with_context(ask_question, results)
                    print(f"Gemini Response:\n{response}")
                else:
                    print("No relevant content found in PDF.")
            elif search_query:
                results = manager.search(search_query, top_k=5)
                print(f"\nSearch Results for: '{search_query}'")
                for i, result in enumerate(results):
                    print(f"\n{i+1}. Distance: {result['distance']:.4f}")
                    print(f"   {result['text'][:100]}...")
        elif use_chunks:
            chunks = load_pdf_chunks(pdf_file, chunk_size=chunk_size, overlap=overlap)
            print(f"Total chunks: {len(chunks)}\n")
            for i, chunk in enumerate(chunks):
                print(f"--- Chunk {i+1} ---")
                print(chunk)
                print()
        else:
            text = load_pdf(pdf_file)
            print(text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)