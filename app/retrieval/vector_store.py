import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

class CodeVectorStore:
    """Manages index operations, embedding generation, and Chroma DB searches."""
    
    def __init__(
        self,
        persist_dir: str | Path,
        embeddings_model_name: str = "models/text-embedding-004"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory where Chroma files are persisted.
            embeddings_model_name: Name of the Google Gemini embedding model.
        """
        self.persist_dir = str(Path(persist_dir).resolve())
        self.embeddings_model_name = embeddings_model_name
        
        # Initialize Google GenAI embeddings using API
        api_key = os.getenv("GEMINI_API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embeddings_model_name,
            google_api_key=api_key
        )
        
        # Initialize Chroma DB
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="codebase_copilot"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], repository_id: str) -> None:
        """
        Creates LangChain Documents and indexes them in Chroma.
        
        Args:
            chunks: List of dictionaries containing 'content' and 'metadata'.
            repository_id: Unique string identifier for the target repository.
        """
        documents = []
        for chunk in chunks:
            # Deep copy metadata to avoid mutating the original dict
            meta = dict(chunk["metadata"])
            # Inject repository_id to facilitate repo-specific queries
            meta["repository_id"] = repository_id
            
            # Chroma DB requires metadata values to be simple types (str, int, float, bool)
            doc = Document(
                page_content=chunk["content"],
                metadata=meta
            )
            documents.append(doc)
            
        if documents:
            self.db.add_documents(documents)

    def search(
        self,
        query: str,
        repository_id: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Performs semantic similarity search over codebase chunks.
        
        Args:
            query: The natural language search query.
            repository_id: Filters results belonging only to this repo.
            k: Top k documents to return.
            filter_dict: Optional dictionary of additional metadata filter pairs.
            
        Returns:
            List of matching LangChain Document instances.
        """
        # Combine repository filter with additional parameters using $and if multiple filters exist
        filters = [{"repository_id": repository_id}]
        if filter_dict:
            for key, val in filter_dict.items():
                filters.append({key: val})
                
        if len(filters) == 1:
            where_clause = filters[0]
        else:
            where_clause = {"$and": filters}
                
        return self.db.similarity_search(
            query=query,
            k=k,
            filter=where_clause
        )

    def clear_repository(self, repository_id: str) -> None:
        """
        Removes all documents linked to the specified repository.
        
        Args:
            repository_id: Unique string identifier of the repository to clear.
        """
        collection = self.db._collection
        # Directly delete via native Chroma client delete method using metadata filters
        collection.delete(where={"repository_id": repository_id})
