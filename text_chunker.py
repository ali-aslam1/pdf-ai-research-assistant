def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters (default: 1000)
        overlap: Number of overlapping characters between chunks (default: 200)
    
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        
        # Move start position by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap
    
    return chunks
