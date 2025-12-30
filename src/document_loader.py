"""
Document Loading and Chunking Module

Loads documents from formats (.txt, .md, .pdf)
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from urllib.parse import urljoin


# for pdf processing
try:
    from PyPDF2 import PdfReader
    pdf_support = True
except ImportError:
    pdf_support = False
    print("PyPDF2 not supported")

# for Grobid processing
try:
    import requests
    grobid_support = True
except ImportError:
    grobid_support = False
    print("Requests library not available for Grobid support")

@dataclass
class DocumentChunk:
    """Represents a single chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

@dataclass
class AcademicSection:
    """Represents a section from an academic paper"""
    title: str
    content: str
    section_type: str  # abstract, introduction, methods, results, conclusion, etc.
    level: int  # heading level (1, 2, 3, etc.)
    references: List[str] = None

class DocumentLoader:
    """Load and process document for vector storage"""
    def __init__(self, 
                 chunk_size:int = 500, 
                 chunk_overlap:int = 50, 
                 min_chunk_size:int = 100,
                 grobid_server_url: str = "http://localhost:8070",
                 use_grobid: bool = True,
                 preserve_academic_structure: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.grobid_server_url = grobid_server_url
        self.use_grobid = use_grobid and grobid_support
        self.preserve_academic_structure = preserve_academic_structure

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
                # Try Grobid first for academic papers, fallback to PyPDF2
                if self.use_grobid and self.is_grobid_available():
                    print(f"Using Grobid for PDF processing: {file_path.name}")
                    result = self.load_pdf_with_grobid(file_path)
                    if result:
                        return result
                    else:
                        print("Grobid failed, falling back to PyPDF2")
                
                if pdf_support:
                    return self.load_pdf_file(file_path)
                else:
                    print("PDF support not available!")
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
        """Load text from pdf files using PyPDF2"""
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
    
    def is_grobid_available(self) -> bool:
        """Check if Grobid server is available"""
        try:
            response = requests.get(f"{self.grobid_server_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def load_pdf_with_grobid(self, file_path: Path) -> Optional[str]:
        """Load and process PDF using Grobid for academic paper structure"""
        try:
            # Process document with Grobid
            with open(file_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                
                # Use processFulltextDocument endpoint for complete processing
                response = requests.post(
                    f"{self.grobid_server_url}/api/processFulltextDocument",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Parse the TEI XML response
                    tei_xml = response.text
                    return self.parse_grobid_tei(tei_xml, file_path.name)
                else:
                    print(f"Grobid processing failed with status: {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"Error processing PDF with Grobid: {str(e)}")
            return None
    
    def parse_grobid_tei(self, tei_xml: str, filename: str) -> str:
        """Parse Grobid TEI XML output into structured text"""
        try:
            # Parse XML
            root = ET.fromstring(tei_xml)
            
            # Define namespaces
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            sections = []
            
            # Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title', namespaces)
            if title_elem is not None:
                sections.append(f"# {title_elem.text}\n")
            
            # Extract abstract
            abstract_elem = root.find('.//tei:abstract', namespaces)
            if abstract_elem is not None:
                abstract_text = self.extract_text_from_element(abstract_elem)
                if abstract_text.strip():
                    sections.append(f"## Abstract\n{abstract_text}\n")
            
            # Extract body sections
            body = root.find('.//tei:body', namespaces)
            if body is not None:
                divs = body.findall('.//tei:div', namespaces)
                for div in divs:
                    section_content = self.parse_section(div, namespaces)
                    if section_content:
                        sections.append(section_content)
            
            # Extract bibliography/references
            back_elem = root.find('.//tei:back', namespaces)
            if back_elem is not None:
                refs = back_elem.findall('.//tei:biblStruct', namespaces)
                if refs:
                    sections.append("## References\n")
                    for i, ref in enumerate(refs, 1):
                        ref_text = self.extract_reference_text(ref, namespaces)
                        if ref_text:
                            sections.append(f"{i}. {ref_text}\n")
            
            full_text = '\n'.join(sections)
            print(f"Processed {filename} with Grobid ({len(full_text)} characters)")
            return full_text
            
        except ET.ParseError as e:
            print(f"Error parsing Grobid XML: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing Grobid output: {str(e)}")
            return None
    
    def parse_section(self, div_elem, namespaces: Dict[str, str]) -> str:
        """Parse a section div element from Grobid TEI"""
        section_parts = []
        
        # Get section heading
        head = div_elem.find('./tei:head', namespaces)
        if head is not None and head.text:
            section_parts.append(f"## {head.text.strip()}\n")
        
        # Get section content
        paragraphs = div_elem.findall('.//tei:p', namespaces)
        for p in paragraphs:
            p_text = self.extract_text_from_element(p)
            if p_text.strip():
                section_parts.append(f"{p_text}\n")
        
        return '\n'.join(section_parts)
    
    def extract_text_from_element(self, element) -> str:
        """Extract clean text from XML element, handling nested elements"""
        if element is None:
            return ""
        
        # Get all text content, including from nested elements
        text_parts = []
        if element.text:
            text_parts.append(element.text)
        
        for child in element:
            text_parts.append(self.extract_text_from_element(child))
            if child.tail:
                text_parts.append(child.tail)
        
        text = ' '.join(text_parts)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_reference_text(self, biblstruct_elem, namespaces: Dict[str, str]) -> str:
        """Extract reference text from biblStruct element"""
        ref_parts = []
        
        # Authors
        authors = biblstruct_elem.findall('.//tei:author', namespaces)
        author_names = []
        for author in authors:
            forename = author.find('.//tei:forename', namespaces)
            surname = author.find('.//tei:surname', namespaces)
            if forename is not None and surname is not None:
                author_names.append(f"{forename.text} {surname.text}")
            elif surname is not None:
                author_names.append(surname.text)
        
        if author_names:
            ref_parts.append(', '.join(author_names))
        
        # Title
        title = biblstruct_elem.find('.//tei:title[@level="a"]', namespaces)
        if title is not None and title.text:
            ref_parts.append(f'"{title.text}"')
        
        # Journal/Conference
        journal = biblstruct_elem.find('.//tei:title[@level="j"]', namespaces)
        if journal is not None and journal.text:
            ref_parts.append(journal.text)
        
        # Year
        date = biblstruct_elem.find('.//tei:date', namespaces)
        if date is not None and date.get('when'):
            ref_parts.append(f"({date.get('when')})")
        
        return '. '.join(ref_parts)
    
    def extract_academic_sections(self, file_path: Path) -> List[AcademicSection]:
        """Extract structured academic sections using Grobid"""
        if not self.use_grobid or not self.is_grobid_available():
            return []
        
        try:
            with open(file_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                response = requests.post(
                    f"{self.grobid_server_url}/api/processFulltextDocument",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return self.parse_academic_sections_from_tei(response.text)
        except Exception as e:
            print(f"Error extracting academic sections: {str(e)}")
        
        return []
    
    def parse_academic_sections_from_tei(self, tei_xml: str) -> List[AcademicSection]:
        """Parse TEI XML to extract academic sections"""
        sections = []
        
        try:
            root = ET.fromstring(tei_xml)
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract abstract
            abstract_elem = root.find('.//tei:abstract', namespaces)
            if abstract_elem is not None:
                abstract_text = self.extract_text_from_element(abstract_elem)
                if abstract_text.strip():
                    sections.append(AcademicSection(
                        title="Abstract",
                        content=abstract_text,
                        section_type="abstract",
                        level=1
                    ))
            
            # Extract body sections
            body = root.find('.//tei:body', namespaces)
            if body is not None:
                divs = body.findall('.//tei:div', namespaces)
                for div in divs:
                    head = div.find('./tei:head', namespaces)
                    if head is not None and head.text:
                        content = []
                        paragraphs = div.findall('.//tei:p', namespaces)
                        for p in paragraphs:
                            p_text = self.extract_text_from_element(p)
                            if p_text.strip():
                                content.append(p_text)
                        
                        if content:
                            section_type = self.classify_section_type(head.text.strip().lower())
                            sections.append(AcademicSection(
                                title=head.text.strip(),
                                content='\n\n'.join(content),
                                section_type=section_type,
                                level=self.get_heading_level(head)
                            ))
            
        except Exception as e:
            print(f"Error parsing academic sections: {str(e)}")
        
        return sections
    
    def classify_section_type(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['abstract']):
            return 'abstract'
        elif any(word in title_lower for word in ['introduction', 'intro']):
            return 'introduction'
        elif any(word in title_lower for word in ['method', 'methodology', 'approach']):
            return 'methods'
        elif any(word in title_lower for word in ['result', 'finding', 'experiment']):
            return 'results'
        elif any(word in title_lower for word in ['discussion', 'analysis']):
            return 'discussion'
        elif any(word in title_lower for word in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(word in title_lower for word in ['related work', 'background', 'literature']):
            return 'background'
        else:
            return 'content'
    
    def get_heading_level(self, head_elem) -> int:
        """Determine heading level from XML element"""
        # This is a simplified approach - Grobid doesn't always provide clear heading levels
        return 2  # Default to level 2
    
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
        # For academic papers, try structure-aware chunking first
        if filename.endswith('.pdf') and self.preserve_academic_structure:
            academic_chunks = self.chunk_academic_document(text, filename)
            if academic_chunks:
                return academic_chunks
        
        # Fallback to regular text chunking
        return self.chunk_regular_text(text, filename)
    
    def chunk_academic_document(self, text: str, filename: str) -> List[DocumentChunk]:
        """Create chunks preserving academic paper structure"""
        chunks = []
        
        # Split text by academic sections (marked with ##)
        sections = re.split(r'\n## ', text)
        
        if len(sections) <= 1:
            return []  # No clear academic structure found
        
        chunk_index = 0
        
        # Handle the first section (usually title)
        if sections[0].strip():
            first_section = sections[0].strip()
            if len(first_section) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    text=first_section,
                    metadata={
                        'filename': filename,
                        'chunk_index': chunk_index,
                        'section_type': 'title',
                        'char_count': len(first_section),
                        'processing_method': 'grobid_academic'
                    },
                    chunk_id=f"{filename}_academic_chunk_{chunk_index}"
                ))
                chunk_index += 1
        
        # Process remaining sections
        for i, section in enumerate(sections[1:], 1):
            section_text = f"## {section}".strip()
            section_type = self.identify_section_type(section_text)
            
            if len(section_text) < self.chunk_size:
                # Section fits in one chunk
                if len(section_text) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        text=section_text,
                        metadata={
                            'filename': filename,
                            'chunk_index': chunk_index,
                            'section_type': section_type,
                            'char_count': len(section_text),
                            'processing_method': 'grobid_academic'
                        },
                        chunk_id=f"{filename}_academic_chunk_{chunk_index}"
                    ))
                    chunk_index += 1
            else:
                # Section needs to be split into multiple chunks
                section_chunks = self.split_large_section(section_text, section_type, filename, chunk_index)
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        # Update total chunks metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        print(f"Created {len(chunks)} academic structure-aware chunks")
        return chunks
    
    def chunk_regular_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """Regular text chunking method (original implementation)"""
        text = self.clean_text(text)
        if len(text) < self.min_chunk_size:
            return [DocumentChunk(
                text=text,
                metadata={
                    'filename': filename,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'char_count': len(text),
                    'processing_method': 'regular'
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
                        'end_pos': end,
                        'processing_method': 'regular'
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
    
    def identify_section_type(self, section_text: str) -> str:
        """Identify the type of academic section"""
        first_line = section_text.split('\n')[0].lower()
        
        if 'abstract' in first_line:
            return 'abstract'
        elif any(word in first_line for word in ['introduction', 'intro']):
            return 'introduction'
        elif any(word in first_line for word in ['method', 'methodology', 'approach']):
            return 'methods'
        elif any(word in first_line for word in ['result', 'finding', 'experiment']):
            return 'results'
        elif any(word in first_line for word in ['discussion', 'analysis']):
            return 'discussion'
        elif any(word in first_line for word in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(word in first_line for word in ['reference', 'bibliography']):
            return 'references'
        elif any(word in first_line for word in ['related work', 'background', 'literature']):
            return 'background'
        else:
            return 'content'
    
    def split_large_section(self, section_text: str, section_type: str, filename: str, start_index: int) -> List[DocumentChunk]:
        """Split large sections while preserving context"""
        chunks = []
        lines = section_text.split('\n')
        
        # Keep the section header
        header = lines[0] if lines else ""
        content_lines = lines[1:] if len(lines) > 1 else []
        
        current_chunk = header + '\n'
        chunk_index = start_index
        
        for line in content_lines:
            # Check if adding this line would exceed chunk size
            if len(current_chunk + line + '\n') > self.chunk_size and len(current_chunk) > len(header):
                # Finalize current chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        metadata={
                            'filename': filename,
                            'chunk_index': chunk_index,
                            'section_type': section_type,
                            'char_count': len(current_chunk.strip()),
                            'processing_method': 'grobid_academic_split'
                        },
                        chunk_id=f"{filename}_academic_chunk_{chunk_index}"
                    ))
                    chunk_index += 1
                
                # Start new chunk with section context
                current_chunk = f"{header}\n[...continued from previous chunk]\n{line}\n"
            else:
                current_chunk += line + '\n'
        
        # Add final chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                metadata={
                    'filename': filename,
                    'chunk_index': chunk_index,
                    'section_type': section_type,
                    'char_count': len(current_chunk.strip()),
                    'processing_method': 'grobid_academic_split'
                },
                chunk_id=f"{filename}_academic_chunk_{chunk_index}"
            ))
        
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
        
        # Count processing methods
        processing_methods = {}
        section_types = {}
        
        for chunk in chunks:
            method = chunk.metadata.get('processing_method', 'unknown')
            processing_methods[method] = processing_methods.get(method, 0) + 1
            
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1

        stats = {
            'total_chunks': len(chunks),
            'total_documents': len(filenames),
            'avg_chunk_size': sum(chunk_sizes)/len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'documents': list(filenames),
            'processing_methods': processing_methods,
            'section_types': section_types
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
        
        print("\n" + "="*60)
        print(" DOCUMENT PROCESSING STATISTICS")
        print("="*60)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        
        print(f"\nChunk Sizes:")
        print(f"  Average: {stats['avg_chunk_size']:.0f} characters")
        print(f"  Min: {stats['min_chunk_size']} characters")
        print(f"  Max: {stats['max_chunk_size']} characters")
        
        print(f"\nProcessing Methods:")
        for method, count in stats['processing_methods'].items():
            print(f"  • {method}: {count} chunks")
        
        if 'section_types' in stats and stats['section_types']:
            print(f"\nAcademic Section Types:")
            for section_type, count in stats['section_types'].items():
                if section_type != 'unknown':
                    print(f"  • {section_type}: {count} chunks")
        
        print(f"\nDocuments:")
        for doc in stats['documents']:
            doc_chunks = [c for c in chunks if c.metadata['filename'] == doc]
            grobid_chunks = len([c for c in doc_chunks if 'grobid' in c.metadata.get('processing_method', '')])
            print(f"  • {doc}: {len(doc_chunks)} chunks ({grobid_chunks} via Grobid)")
        
        print("="*60)
    
    def get_grobid_status(self) -> Dict[str, Any]:
        """Get Grobid server status and configuration"""
        status = {
            'grobid_support': grobid_support,
            'use_grobid': self.use_grobid,
            'server_url': self.grobid_server_url,
            'server_available': False,
            'preserve_academic_structure': self.preserve_academic_structure
        }
        
        if self.use_grobid:
            status['server_available'] = self.is_grobid_available()
        
        return status
    
    def print_grobid_status(self):
        """Print current Grobid configuration and status"""
        status = self.get_grobid_status()
        
        print("\n" + "="*50)
        print(" GROBID CONFIGURATION")
        print("="*50)
        print(f"Grobid Support Available: {status['grobid_support']}")
        print(f"Use Grobid: {status['use_grobid']}")
        print(f"Server URL: {status['server_url']}")
        print(f"Server Available: {status['server_available']}")
        print(f"Preserve Academic Structure: {status['preserve_academic_structure']}")
        
        if not status['server_available'] and status['use_grobid']:
            print("\n Grobid server not available!")
            print("   To use Grobid:")
            print("   1. Install Grobid: https://grobid.readthedocs.io/")
            print("   2. Start server: ./gradlew run")
            print("   3. Or use Docker: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        
        print("="*50)

if __name__ == "__main__":
    print(f"🔬 Enhanced Document Loader with Grobid Support - Test Run")
    
    # Create loader with Grobid support
    loader = DocumentLoader(
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_size=100,
        grobid_server_url="http://localhost:8070",
        use_grobid=True,
        preserve_academic_structure=True
    )
    
    # Display Grobid status
    loader.print_grobid_status()
    
    # Example usage with documents directory
    print(f"\n Ready to process documents!")
    print(f"Supported formats: {loader.supported_extension}")
    print(f"\nFor academic papers (PDFs):")
    print(f"  • Grobid processing: {'Enabled' if loader.use_grobid else 'Disabled'}")
    print(f"  • Structure preservation: {'Enabled' if loader.preserve_academic_structure else 'Disabled'}")
    print(f"\nTo test with your documents:")
    print(f"1. Place PDF papers in 'documents/' folder")
    print(f"2. Run: documents = loader.load_directory('documents/')")
    print(f"3. Run: chunks = loader.process_documents(documents)")
    print(f"4. Run: loader.display_statistics(chunks)")
    
    # Test with sample document if available
    import os
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        print(f"\nTesting with documents in '{documents_dir}/'...")
        documents = loader.load_directory(documents_dir)
        
        if documents:
            chunks = loader.process_documents(documents)
            loader.display_statistics(chunks)
            
            # Show sample chunks
            print(f"\nSample chunks:")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"\n--- Chunk {i} ---")
                print(f"Type: {chunk.metadata.get('section_type', 'unknown')}")
                print(f"Method: {chunk.metadata.get('processing_method', 'unknown')}")
                print(f"Preview: {chunk.text[:200]}...")
        else:
            print(f"No supported documents found in '{documents_dir}/'")
    else:
        print(f"\nCreate a 'documents/' folder and add PDF files to test!")
    
    print(f"\nTest completed!")
    
    # Configuration tips
    print(f"\n💡 Configuration Tips:")
    print(f"• For research papers: use_grobid=True, preserve_academic_structure=True")
    print(f"• For general docs: use_grobid=False")
    print(f"• Larger chunk_size (800-1000) works better for academic content")
    print(f"• Enable Grobid server for best results with scientific PDFs")


    


                


                



        

        