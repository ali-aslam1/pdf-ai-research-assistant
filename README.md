# AI PDF Research Assistant 📄🔍

A simple Retrieval-Augmented Generation (RAG) application that allows users to ask questions about their PDF documents. Instead of relying on generic LLM responses, this tool grounds its answers specifically in the text extracted from the uploaded document.

## 🎯 Purpose
This project was built to explore the fundamentals of RAG architectures. It demonstrates how to connect document parsing, local vector storage, semantic search, and the Gemini API into a cohesive, functional pipeline wrapped in a Streamlit interface.

## ✨ Features
- **PDF Processing:** Upload and automatically extract text from PDF files.
- **Intelligent Chunking:** Splits large documents into manageable text chunks for efficient processing.
- **Local Embeddings:** Generates embeddings and stores them locally using ChromaDB—no external database setup required.
- **Semantic Search:** Retrieves the most relevant document chunks based on the user's query.
- **Grounded Generation:** Sends the retrieved context alongside the query to the Gemini API for highly accurate, document-specific answers.
- **Interactive UI:** A clean, easy-to-use chat interface built with Streamlit.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM / Generative AI:** Google Gemini API
- **Vector Database:** ChromaDB (Local)
- **Language:** Python

### Usage
Run the Streamlit application using the following command:
\`\`\`bash
streamlit run app.py
\`\`\`
Navigate to the provided local URL in your browser, upload a PDF, and start asking questions!