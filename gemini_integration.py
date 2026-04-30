import google.genai as genai
import os
from typing import List, Dict
import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Mute the noisy third-party loggers by forcing them to only show WARNINGs or higher
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger("gemini")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False  # Don't bubble up to root logger

class GeminiIntegration:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini integration.
        
        Args:
            api_key: API key (optional). If not provided, reads from GEMINI_API_KEY env var.
        """
        # Get API key from argument or environment variable
        if api_key:
            api_key_to_use = api_key
            logger.info("Using provided API key")
        else:
            api_key_to_use = os.getenv('GEMINI_API_KEY')
            if not api_key_to_use:
                logger.error("[ERROR] GEMINI_API_KEY environment variable not set")
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. \n"
                    "Set it using: \n"
                    "  PowerShell: $env:GEMINI_API_KEY = 'your_api_key'\n"
                    "  Cmd: set GEMINI_API_KEY=your_api_key\n"
                    "  Linux/Mac: export GEMINI_API_KEY=your_api_key"
                )
            logger.info("Using API key from GEMINI_API_KEY environment variable")
        
        try:
            self.client = genai.Client(api_key=api_key_to_use)
            self.model = 'gemini-3-flash-preview'  
            logger.info("[SUCCESS] Gemini API configured successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to configure Gemini API: {str(e)}")
            raise
    
    def ask_question(self, question: str, context: List[Dict] = None) -> str:
        """
        Ask Gemini a question with optional context from embeddings.
        
        Args:
            question: The user's question
            context: List of context dicts from embedding search results
                    Each dict should have 'text' key with the chunk content
        
        Returns:
            Gemini's response
        """
        # Build prompt with context if provided
        if context:
            context_text = "\n\n".join([
                f"[Context {i+1}]:\n{item['text']}"
                for i, item in enumerate(context)
            ])
            prompt = f"""Based on the following context, please answer the question.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
        else:
            prompt = question
        
        # Debug: Check prompt size
        prompt_size = len(prompt)
        logger.debug(f"Prompt size: {prompt_size} characters")
        if prompt_size > 50000:
            logger.warning(f"[WARNING] Large prompt detected ({prompt_size} chars). This may cause slowdowns or timeouts.")
        
        try:
            logger.debug("Sending request to Gemini API...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            # Check if response was blocked by safety filters
            if not response or not response.text:
                logger.error("[ERROR] Response was blocked or empty. Checking safety feedback...")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    logger.error(f"Safety feedback: {response.prompt_feedback}")
                raise ValueError("Gemini API returned an empty response. This may be due to safety filters being triggered.")
            
            response_text = response.text
            logger.info("[SUCCESS] Successfully received response from Gemini")
            return response_text
            
        except Exception as e:
            logger.error(f"[ERROR] Error calling Gemini API: {str(e)}")
            logger.debug(f"Error type: {type(e).__name__}")
            raise
    
    def chat_with_context(self, question: str, context: List[Dict], max_context_size: int = 30000) -> str:
        """
        Chat with Gemini using retrieved context (RAG - Retrieval Augmented Generation).
        
        Args:
            question: The user's question
            context: List of context dicts from embedding search results
            max_context_size: Maximum characters to include in context (default: 30000)
        
        Returns:
            Gemini's response
        """
        # Limit context to prevent massive prompts
        limited_context = self._limit_context(context, max_context_size)
        
        if len(limited_context) < len(context):
            logger.info(f"[WARNING] Context limited from {len(context)} chunks to {len(limited_context)} to stay under {max_context_size} characters")
        
        return self.ask_question(question, limited_context)
    
    def _limit_context(self, context: List[Dict], max_size: int) -> List[Dict]:
        """
        Limit context to stay under max_size characters.
        
        Args:
            context: List of context chunks
            max_size: Maximum total characters
        
        Returns:
            Limited context list
        """
        limited = []
        current_size = 0
        
        for chunk in context:
            chunk_text = chunk.get('text', '')
            chunk_size = len(chunk_text)
            
            if current_size + chunk_size <= max_size:
                limited.append(chunk)
                current_size += chunk_size
            else:
                break
        
        return limited

if __name__ == "__main__":
    # Example usage
    # Set API key via environment variable:
    # Windows: set GEMINI_API_KEY=your_api_key
    # Linux/Mac: export GEMINI_API_KEY=your_api_key
    
    try:
        logger.info("[INFO] Initializing Gemini integration...")
        gemini = GeminiIntegration()
        
        # Simple question without context
        logger.info("[INFO] Test 1: Question without context")
        response = gemini.ask_question("What is machine learning?")
        print("Response without context:")
        print(response)
        print("\n" + "="*50 + "\n")
        
        # With context
        logger.info("[INFO] Test 2: Question with context")
        sample_context = [
            {"text": "Machine learning is a subset of artificial intelligence."},
            {"text": "Deep learning uses neural networks with multiple layers."}
        ]
        response = gemini.chat_with_context("Explain machine learning", sample_context)
        print("Response with context:")
        print(response)
        
    except ValueError as e:
        logger.error(f"[ERROR] Configuration error: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
