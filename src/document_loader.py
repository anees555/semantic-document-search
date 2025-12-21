"""
Document Loading and Chunking Module

Loads documents from formats (.txt, .md, .pdf)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


# for pdf processing
try:
    from PyPDF2 import PdfReader
    pdf_support = True

except ImportError:
    pdf_support = False
    print("Pdf not supported")

@dataclass

class DocumentChunk:
    """Represents a single chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentLoader:
    """Load and process document for vector storage"""
    def __init__(self, 
                 chunk_size:int = 500, 
                 chunk_overlap:int = 50, 
                 min_chunk_size:int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.supported_extension = ['.txt', '.md', '.pdf']

    def load_document(self, file_path:str) -> Optional[str]:
        "load a single document and return its text contents"
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found! {file_path}")
            return None
        extension = file_path.suffix.lower()

        try:
            if extension in ['.txt', '.md']:
                return self.load_text_file(file_path)
            elif extension == '.pdf':
                if pdf_support:
                    return self.load_pdf_file(file_path)
                else:
                    print("Pdf support not available!")
                    return None
            else:
                print(f"Unsupported file type: {extension}")
                return None
        
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
        
    def load_text_file(self, file_path: Path) -> str:
        """Load text from .txt and .md file"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                print(f"Loaded {file_path.name} ({len(content)} characters)")
                return content
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Couldnot decode {file_path} with any encodings")
    
    def load_pdf_file(self, file_path: Path) -> str:
        """Load text from pdf files"""
        reader = PdfReader(str(file_path))
        text_content = []

        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_content.append(f"\n--- Page {page_num} ---\n")
                text_content.append(page_text)
        full_text = ''.join(text_content)
        print(f"Loaded {file_path} ({len(full_text)} characters)")
        return full_text
    
    def load_directory(self, directory_path:str) -> Dict[str, str]:
        """Loads all the supported documents from the directory"""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return {}
        
        documents = {}
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extension:
                text = self.load_document(file_path)
                if text:
                    documents[file_path.name] = text
        
        print(f"Loaded {len(documents)} from {directory_path}")
        return documents
    
    def clean_text(self, text:str) -> str:
        """Clean and normalize text"""

        text = re.sub(r'\s+', ' ', text) # removes the excessive whitespaces

        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;\"]', '', text) # removes the special characters but keeps special punctuations

        text = re.sub(r'([\.!?]){2,}', r'\1', text) # removes multiple consecutive punctuation

        return text
    
    def chunk_text(self, text:str, filename:str = "Unknown") -> List[DocumentChunk]:
        """Split the text into documentchunk
            returns the list of DocumentChunk objects
        """
        text = self.clean_text(text)
        if len(text) < self.min_chunk_size:
            return [DocumentChunk(
                text=text,
                metadata={
                    'filename': filename,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'char_count': len(text)
                },
                chunk_id = f"{filename}_chunk_0"
            )]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # calculate the end position
            end = start + self.chunk_size

            if end < len(text):
                sentence_end = self.find_sentence_boundary(text, end, start)
                if sentence_end != -1:
                    end = sentence_end
                
            chunk_text = text[start:end].strip()
            # skip tiny chunks
            if len(chunk_text) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        'filename': filename,
                        'chunk_index': chunk_index,
                        'char_count': len(chunk_text),
                        'start_pos': start,
                        'end_pos': end
                    },
                    chunk_id=f"{filename}_chunk_{chunk_index}"

                )
                chunks.append(chunk)
                chunk_index += 1

            # move to next chunk with overlap
            start = end - self.chunk_overlap

            if start >= len(text):
                break

        # update the total chunk in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        print(f"Split into {len(chunks)} chunks (size: {self.chunk_size}), overlap: {self.chunk_overlap}")
        return chunks
    
    def find_sentence_boundary(self, text:str, position:int, min_pos:int) -> int:
        """Find the nearest sentence boundary before the given position
            Returns the position the boundary or -1 if not found.
        """
        search_start = max(min_pos, position-100)
        search_text = text[search_start:position]

        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        last_boundary = -1
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos != -1:
                actual_pos = search_start + pos + len(ending)
                last_boundary = max(last_boundary, actual_pos)

        return last_boundary
    
    def process_documents(self, documents: Dict[str, str]) -> List[DocumentChunk]:
        """Process the multiple documents in to chunks"""
        all_chunks = []

        print("\n---- Processing Documents...")
        for filename, text in documents.items():
            print(f"\nProcessing: {filename}")
            chunks = self.chunk_text(text, filename)
            all_chunks.extend(chunks)

        print(f"\nTotal chunk created: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_statistics(self, chunks:List[DocumentChunk]) -> Dict:
        """Get statistics about the chunk"""
        if not chunks:
            return {}
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        filenames = set(chunk.metadata['filename'] for chunk in chunks)

        stats = {
            'total_chunks': len(chunks),
            'total_documents': len(filenames),
            'avg_chunk_size': sum(chunk_sizes)/len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'documents': list(filenames)
        }
        return stats
    
    def display_statistics(self, chunks: List[DocumentChunk]):
        """
        Display chunk statistics in a readable format
        """
        stats = self.get_chunk_statistics(chunks)
        
        if not stats:
            print("No chunks to analyze")
            return
        
        print("\n" + "="*50)
        print(" DOCUMENT STATISTICS")
        print("="*50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"\nChunk Sizes:")
        print(f"  Average: {stats['avg_chunk_size']:.0f} characters")
        print(f"  Min: {stats['min_chunk_size']} characters")
        print(f"  Max: {stats['max_chunk_size']} characters")
        print(f"\nDocuments:")
        for doc in stats['documents']:
            doc_chunks = [c for c in chunks if c.metadata['filename'] == doc]
            print(f"  • {doc}: {len(doc_chunks)} chunks")
        print("="*50)

if __name__ == "__main__":
    print(f" Document Loader - Test Run")
    loader = DocumentLoader(
        chunk_size= 500,
        chunk_overlap=100,
        min_chunk_size=100
    )



    # print("Testing text chunking...")
    # chunks = loader.chunk_text(sample_text, "documents\sample_ml.txt")

    # print(f"\nCreated {len(chunks)} chunks:\n")
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"Chunk {i} ({len(chunk.text)} chars):")
    #     print(f"  {chunk.text[:100]}...")
    #     print(f"  Metadata: {chunk.metadata}\n")
    
    # # Display statistics
    # loader.display_statistics(chunks)
    
    # print("\nTest completed!")
    # print("\nTo use with your documents:")
    # print("1. Place your files in a 'documents/' folder")
    # print("2. Run: documents = loader.load_directory('documents/')")
    # print("3. Run: chunks = loader.process_documents(documents)")


    


                


                



        

        