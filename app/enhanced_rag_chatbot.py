import os
import json
import pickle
import hashlib
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil

# Import the LLM folder components
from .llm.settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    USE_LOCAL_EMBEDDINGS, USE_LOCAL_LLM,
    BASE_DIR, UPLOAD_DIR, VECTOR_DIR, LLAMA_PARSE_ENABLED
)
from .llm.rag import get_index, add_files, query as llamaindex_query, clear_index
from .llm.llama_parser import parse_with_llamaparse

import fitz  # PyMuPDF
from docx import Document
import math
from collections import Counter

class EnhancedRAGChatbot:
    """
    Enhanced RAG-Powered Maritime AI Chatbot with LlamaIndex Integration
    
    This chatbot provides intelligent document processing and query capabilities:
    - Document upload and processing (PDF, DOCX, TXT) with LlamaParse
    - Advanced LlamaIndex-based retrieval with vector embeddings
    - Context-aware response generation using OpenAI GPT models
    - Maritime domain knowledge integration
    - Hybrid approach: LlamaIndex for retrieval + OpenAI for generation
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Storage directories
        self.documents_dir = "documents"
        self.chunks_dir = "chunks"
        self.index_dir = "index"
        
        # Create directories if they don't exist
        for directory in [self.documents_dir, self.chunks_dir, self.index_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Document storage
        self.documents = {}
        self.chunks = []
        self.document_metadata = {}
        
        # LLM Configuration
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.use_llm = bool(self.openai_api_key)
        
        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.use_llm = False
        else:
            self.logger.info("Running in offline mode without LLM API")
        
        # Initialize LlamaIndex components
        self._initialize_llamaindex()
        
        # Load existing data
        self._load_existing_data()
        
        self.logger.info("Enhanced RAG Chatbot initialized successfully")
    
    def _initialize_llamaindex(self):
        """Initialize LlamaIndex components"""
        try:
            # Import LlamaIndex components
            from llama_index.core import Settings
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.llms.openai import OpenAI as OpenAILLM
            
            # Configure embeddings
            if not USE_LOCAL_EMBEDDINGS and self.openai_api_key:
                Settings.embed_model = OpenAIEmbedding(
                    model=EMBEDDING_MODEL, 
                    api_key=self.openai_api_key
                )
                self.logger.info(f"LlamaIndex embeddings configured with {EMBEDDING_MODEL}")
            
            # Configure LLM
            if not USE_LOCAL_LLM and self.openai_api_key:
                Settings.llm = OpenAILLM(
                    model=LLM_MODEL, 
                    api_key=self.openai_api_key
                )
                self.logger.info(f"LlamaIndex LLM configured with {LLM_MODEL}")
            
            self.llamaindex_ready = True
            self.logger.info("LlamaIndex components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex: {e}")
            self.llamaindex_ready = False
    
    def _load_existing_data(self):
        """Load existing documents and chunks from storage"""
        try:
            # Load documents metadata
            metadata_file = os.path.join(self.documents_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
            
            # Load chunks
            chunks_file = os.path.join(self.chunks_dir, "chunks.pkl")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
            
            self.logger.info(f"Loaded {len(self.document_metadata)} documents and {len(self.chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Error loading existing data: {e}")
    
    def _save_data(self):
        """Save documents metadata and chunks to storage"""
        try:
            # Save documents metadata
            metadata_file = os.path.join(self.documents_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, indent=2, ensure_ascii=False)
            
            # Save chunks
            chunks_file = os.path.join(self.chunks_dir, "chunks.pkl")
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            self.logger.info("Data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def upload_document(self, file_path: str, filename: str, file_type: str) -> Dict[str, Any]:
        """Upload and process a document using ONLY LlamaIndex RAG pipeline"""
        try:
            self.logger.info(f"Processing document with LlamaIndex RAG: {filename}")
            
            # Generate document ID
            doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Store document metadata
            self.document_metadata[doc_id] = {
                "id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "upload_date": datetime.now().isoformat(),
                "processing_method": "llamaindex_rag_only"
            }
            
            # Use ONLY LlamaIndex RAG pipeline
            if not self.llamaindex_ready:
                return {
                    "success": False,
                    "error": "LlamaIndex RAG system is not ready",
                    "message": "The LlamaIndex RAG system is not available. Please check the configuration."
                }
            
            try:
                # Add file to LlamaIndex system
                copied_count, copied_files = add_files([file_path])
                
                if copied_count > 0:
                    # Update metadata with LlamaIndex info
                    self.document_metadata[doc_id]["llamaindex_processed"] = True
                    self.document_metadata[doc_id]["llamaindex_files"] = copied_files
                    
                    # Get file size
                    file_size = os.path.getsize(file_path)
                    self.document_metadata[doc_id]["file_size"] = file_size
                    
                    # Save data
                    self._save_data()
                    
                    self.logger.info(f"Document processed successfully with LlamaIndex RAG: {filename}")
                    
                    return {
                        "success": True,
                        "document_id": doc_id,
                        "processing_method": "llamaindex_rag_only",
                        "message": f"Document '{filename}' processed successfully with LlamaIndex RAG",
                        "llamaindex_files": copied_files
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to process document with LlamaIndex RAG",
                        "message": f"Failed to process document '{filename}' with LlamaIndex RAG system."
                    }
                    
            except Exception as e:
                self.logger.error(f"LlamaIndex RAG processing failed: {e}")
                return {
                    "success": False,
                    "error": f"LlamaIndex RAG processing error: {str(e)}",
                    "message": f"Failed to process document '{filename}' with LlamaIndex RAG system."
                }
                
        except Exception as e:
            self.logger.error(f"Error processing document {filename}: {e}")
            return {"success": False, "error": str(e)}
    
    def _fallback_document_processing(self, file_path: str, filename: str, file_type: str, doc_id: str) -> Dict[str, Any]:
        """Fallback document processing when LlamaIndex is not available"""
        try:
            # Extract text based on file type
            if file_type.lower() == 'pdf':
                text = self._extract_text_from_pdf(file_path)
            elif file_type.lower() == 'docx':
                text = self._extract_text_from_docx(file_path)
            elif file_type.lower() == 'txt':
                text = self._extract_text_from_txt(file_path)
            else:
                return {"success": False, "error": f"Unsupported file type: {file_type}"}
            
            if not text.strip():
                return {"success": False, "error": "No text content found in document"}
            
            # Create chunks
            chunks = self._chunk_text(text)
            
            # Update metadata
            self.document_metadata[doc_id]["text_length"] = len(text)
            self.document_metadata[doc_id]["chunk_count"] = len(chunks)
            self.document_metadata[doc_id]["processing_method"] = "fallback_basic"
            
            # Store chunks with metadata
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "content": chunk,
                    "chunk_index": i,
                    "filename": filename
                })
            
            # Save data
            self._save_data()
            
            self.logger.info(f"Document processed successfully with fallback method: {filename} ({len(chunks)} chunks)")
            
            return {
                "success": True,
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "processing_method": "fallback_basic",
                "message": f"Document '{filename}' processed successfully with fallback method"
            }
            
        except Exception as e:
            self.logger.error(f"Fallback processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error extracting text from TXT: {e}")
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_query(self, query_text: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query using ONLY LlamaIndex RAG pipeline"""
        try:
            self.logger.info(f"Processing query with LlamaIndex RAG: {query_text}")
            
            if not self.llamaindex_ready:
                return {
                    "success": False,
                    "error": "LlamaIndex RAG system is not ready",
                    "response": "The LlamaIndex RAG system is not available. Please check the configuration."
                }
            
            # Use ONLY LlamaIndex RAG
            try:
                answer, citations = llamaindex_query(query_text)
                
                if answer and not answer.startswith("Query failed"):
                    # Generate enhanced response using OpenAI
                    enhanced_response = self._generate_enhanced_response(
                        query_text, answer, citations, chat_history
                    )
                    
                    # Format source documents for better UI display
                    formatted_sources = []
                    for citation in citations:
                        if isinstance(citation, str):
                            formatted_sources.append({
                                "filename": citation,
                                "content": f"Referenced from {citation}",
                                "score": "N/A"
                            })
                        else:
                            formatted_sources.append(citation)
                    
                    return {
                        "success": True,
                        "response": enhanced_response,
                        "source_documents": formatted_sources,
                        "processing_method": "llamaindex_rag_only",
                        "llamaindex_answer": answer,
                        "suggestions": self._generate_suggestions(query_text)
                    }
                else:
                    return {
                        "success": False,
                        "error": "LlamaIndex query failed",
                        "response": "I couldn't find relevant information in the uploaded documents. Please try rephrasing your question or upload more documents.",
                        "llamaindex_error": answer
                    }
                    
            except Exception as e:
                self.logger.error(f"LlamaIndex RAG query failed: {e}")
                return {
                    "success": False,
                    "error": f"LlamaIndex RAG error: {str(e)}",
                    "response": "The LlamaIndex RAG system encountered an error. Please try again or check the document upload."
                }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your query. Please try again."
            }
    
    def _generate_enhanced_response(self, query: str, llamaindex_answer: str, citations: List[str], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate enhanced response using OpenAI with LlamaIndex context"""
        if not self.use_llm:
            return llamaindex_answer
        
        try:
            # Prepare system message
            system_message = """You are an intelligent maritime AI assistant with access to document context from LlamaIndex.
            Your role is to provide accurate, helpful, and conversational responses based on the provided context.
            
            Guidelines:
            - Use the provided LlamaIndex answer as your primary source
            - Enhance and expand the response to be more conversational and helpful
            - If the information is not in the context, say so clearly
            - Provide specific details and examples when available
            - Be conversational and engaging
            - If asked about maritime topics not in the context, use your general knowledge
            - Format responses clearly with proper structure
            - Cite sources when appropriate
            """
            
            # Prepare messages for the API
            messages = [{"role": "system", "content": system_message}]
            
            # Add chat history if provided
            if chat_history:
                for msg in chat_history[-6:]:  # Keep last 6 messages for context
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add the current query with LlamaIndex context
            user_message = f"""Based on the following LlamaIndex answer and document context, please provide an enhanced response to this question: {query}

LlamaIndex Answer:
{llamaindex_answer}

Document Sources: {', '.join(citations) if citations else 'No specific sources'}

Please provide a comprehensive, conversational, and enhanced response based on this context."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced response: {e}")
            return llamaindex_answer
    
    def _fallback_query_processing(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Fallback query processing using basic search"""
        try:
            # Search for relevant documents using basic method
            relevant_chunks = self._basic_search_documents(query, top_k=5)
            
            # Generate response
            response = self._generate_basic_response(query, relevant_chunks, chat_history)
            
            # Prepare source documents for response
            source_documents = []
            for chunk in relevant_chunks[:3]:  # Top 3 sources
                source_documents.append({
                    "filename": chunk['filename'],
                    "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                    "score": round(chunk['score'], 3)
                })
            
            return {
                "success": True,
                "response": response,
                "source_documents": source_documents,
                "processing_method": "fallback_basic",
                "suggestions": self._generate_suggestions(query)
            }
            
        except Exception as e:
            self.logger.error(f"Fallback query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your query. Please try again."
            }
    
    def _basic_search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Basic document search using word frequency and cosine similarity"""
        if not self.chunks:
            return []
        
        query_vector = self._get_word_frequency_vector(query)
        
        results = []
        
        for chunk in self.chunks:
            chunk_vector = self._get_word_frequency_vector(chunk['content'])
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_vector, chunk_vector)
            
            if similarity > 0.1:  # Minimum threshold
                results.append({
                    'chunk_id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'content': chunk['content'],
                    'filename': chunk['filename'],
                    'score': similarity
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _get_word_frequency_vector(self, text: str) -> Dict[str, float]:
        """Create word frequency vector for text"""
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(words)
        total_words = len(words)
        
        # Create frequency vector
        if total_words == 0:
            return {}
        
        return {word: count / total_words for word, count in word_counts.items()}
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two word frequency vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Get all unique words
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        if not all_words:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        mag1 = math.sqrt(sum(vec1.get(word, 0) ** 2 for word in all_words))
        mag2 = math.sqrt(sum(vec2.get(word, 0) ** 2 for word in all_words))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _generate_basic_response(self, query: str, chunks: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using basic method"""
        if not chunks:
            return "I don't have enough information to answer that question. Please upload some documents first."
        
        # If LLM is available, use it
        if self.use_llm:
            return self._generate_llm_response(query, chunks, chat_history)
        else:
            # Fallback to text-based generation
            return self._generate_text_based_response(query, chunks)
    
    def _generate_llm_response(self, query: str, chunks: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            # Prepare context from chunks
            context_text = "\n\n".join([chunk['content'] for chunk in chunks[:5]])  # Use top 5 chunks
            
            # Prepare system message
            system_message = """You are an intelligent maritime AI assistant with access to uploaded documents. 
            Your role is to provide accurate, helpful responses based on the document content provided.
            
            Guidelines:
            - Answer questions based on the provided document context
            - If the information is not in the documents, say so clearly
            - Provide specific details and examples when available
            - Be conversational and helpful
            - If asked about maritime topics not in the documents, use your general knowledge
            - Format responses clearly with proper structure
            """
            
            # Prepare messages for the API
            messages = [{"role": "system", "content": system_message}]
            
            # Add chat history if provided
            if chat_history:
                for msg in chat_history[-6:]:  # Keep last 6 messages for context
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add the current query with context
            user_message = f"""Based on the following document context, please answer this question: {query}

Document Context:
{context_text}

Please provide a comprehensive and accurate response based on the document content."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to text-based generation
            return self._generate_text_based_response(query, chunks)
    
    def _generate_text_based_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate response using text-based approach (fallback)"""
        if not chunks:
            return "I don't have enough information to answer that question."
        
        # Create a general response based on the most relevant chunks
        response_parts = ["Based on the available documents, here's what I found:"]
        
        for i, chunk in enumerate(chunks[:3], 1):
            response_parts.append(f"\n**Section {i}:**")
            response_parts.append(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
        
        return "\n".join(response_parts)
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """Generate suggestions for follow-up questions"""
        suggestions = [
            "Can you provide more details about the costs mentioned?",
            "What are the key recommendations from the documents?",
            "Are there any risks or challenges identified?",
            "What are the main findings or conclusions?",
            "Can you summarize the key points?",
            "What are the next steps or action items?"
        ]
        return suggestions
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "success": True,
            "status": {
                "documents_loaded": len(self.document_metadata),
                "chunks_available": len(self.chunks),
                "llamaindex_ready": self.llamaindex_ready,
                "openai_configured": self.use_llm,
                "llm_model": LLM_MODEL if self.use_llm else "Not configured",
                "embedding_model": EMBEDDING_MODEL if self.llamaindex_ready else "Not configured",
                "system_health": "Operational"
            }
        }
    
    def get_documents_list(self) -> Dict[str, Any]:
        """Get list of uploaded documents"""
        documents_list = []
        for doc_id, metadata in self.document_metadata.items():
            documents_list.append({
                "id": doc_id,
                "filename": metadata["filename"],
                "file_type": metadata["file_type"],
                "upload_date": metadata["upload_date"],
                "processing_method": metadata.get("processing_method", "unknown"),
                "text_length": metadata.get("text_length", 0),
                "chunk_count": metadata.get("chunk_count", 0)
            })
        
        return {
            "success": True,
            "documents": documents_list
        }
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and its chunks"""
        try:
            if document_id not in self.document_metadata:
                return {"success": False, "error": "Document not found"}
            
            # Get filename before deleting metadata
            filename = self.document_metadata[document_id]["filename"]
            
            # Remove document metadata
            del self.document_metadata[document_id]
            
            # Remove associated chunks
            self.chunks = [chunk for chunk in self.chunks if chunk['document_id'] != document_id]
            
            # Handle LlamaIndex document deletion
            if self.llamaindex_ready:
                try:
                    # Clear the entire index since LlamaIndex doesn't support selective deletion
                    # This is a limitation of the current implementation
                    clear_index()
                    self.logger.info(f"LlamaIndex index cleared after deleting: {filename}")
                    
                    # Rebuild index with remaining documents if any exist
                    if self.document_metadata:
                        try:
                            from .llm.rag import get_index
                            get_index()  # This will rebuild the index with remaining documents
                            self.logger.info("LlamaIndex index rebuilt with remaining documents")
                        except Exception as rebuild_error:
                            self.logger.warning(f"Failed to rebuild LlamaIndex index: {rebuild_error}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to clear LlamaIndex index: {e}")
            
            # Save updated data
            self._save_data()
            
            self.logger.info(f"Document deleted: {filename}")
            
            return {
                "success": True,
                "message": f"Document '{filename}' deleted successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return {"success": False, "error": str(e)}
    
    def get_maritime_knowledge_base(self) -> Dict[str, Any]:
        """Get comprehensive maritime domain knowledge"""
        return {
            "vessel_types": {
                "capesize": {
                    "description": "Large bulk carriers that cannot pass through the Panama Canal",
                    "dwt_range": "100,000-400,000 MT",
                    "typical_cargo": "Iron ore, coal, bauxite",
                    "routes": "Brazil-China, Australia-China, South Africa-China"
                },
                "panamax": {
                    "description": "Ships designed to fit through the Panama Canal",
                    "dwt_range": "60,000-80,000 MT",
                    "typical_cargo": "Grain, coal, minerals",
                    "routes": "US Gulf-China, Brazil-Europe, Australia-Asia"
                },
                "supramax": {
                    "description": "Medium-sized bulk carriers",
                    "dwt_range": "50,000-60,000 MT",
                    "typical_cargo": "Grain, coal, fertilizers",
                    "routes": "Global, versatile trading"
                },
                "handysize": {
                    "description": "Small bulk carriers for regional trade",
                    "dwt_range": "10,000-40,000 MT",
                    "typical_cargo": "Grain, steel, cement",
                    "routes": "Regional and short-sea trade"
                }
            },
            "key_metrics": {
                "bdi": "Baltic Dry Index - measures shipping costs for dry bulk commodities",
                "tce": "Time Charter Equivalent - daily earnings for vessel owners",
                "pda": "Port Disbursement Account - estimated port costs",
                "freight_rate": "Cost per metric ton for cargo transportation"
            },
            "major_routes": {
                "iron_ore": ["Brazil-China", "Australia-China", "South Africa-China"],
                "coal": ["Australia-China", "Indonesia-China", "Colombia-Europe"],
                "grain": ["US Gulf-China", "Brazil-Europe", "Argentina-Asia"],
                "bauxite": ["Guinea-China", "Australia-China"]
            },
            "cost_factors": {
                "bunker_costs": "Fuel expenses (VLSFO, HSFO, MGO)",
                "port_charges": "Port fees, pilotage, towage",
                "canal_costs": "Suez Canal, Panama Canal tolls",
                "operational_costs": "Crew, insurance, maintenance"
            },
            "market_indicators": {
                "supply": "Available vessel capacity",
                "demand": "Cargo volume requiring transportation",
                "seasonality": "Agricultural harvest cycles, weather patterns",
                "geopolitics": "Trade policies, sanctions, conflicts"
            }
        }
