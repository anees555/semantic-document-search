# SEMANTIC DOCUMENT SEARCH USING RETRIEVAL AUGMENTED GENERATION

#### A document question-answering system that uses Retrieval Augmented Generation (RAG) to answer questions based on your personal documents. The system converts documents into embeddings, stores them in a vector database, and retrieves relevant context to generate accurate answers.

### Oveview
This project demonstrates the implementation of a RAG pipeline using ChromaDB for vector storage, Sentence Transformers for embeddings, and Claude API for answer generation. It allows users to upload their documents and ask questions about the content, with the system retrieving relevant passages to provide contextually accurate responses.


### Features

* Document loading and processing for multiple file formats (TXT, PDF, Markdown)
* Intelligent text chunking with overlap to preserve context
* Vector embeddings using Sentence Transformers
* Semantic search with ChromaDB
* Cosine similarity-based retrieval
* RAG implementation for accurate question answering
* Simple web interface built with Streamlit