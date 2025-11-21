# import os
# import faiss
# import numpy as np
# import pickle
# from typing import List, Any
# from sentence_transformers import SentenceTransformer
# from src.embedding import EmbeddingPipeline

# class FaissVectorStore:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size; int = 1000, chunk_overlap: int = 200):
#         self.persist_dir = persist_dir
#         os.makedirs(self.persist_dir, exist_ok=True)
#         self.index = None
#         self.metadata = []
#         self.embedding_model = embedding_model
#         self.model = SentenceTransformer(embedding_model)
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         print(f"[INFO] Loaded embedding model: {embedding_model}")

#     def build_from_documents(self, documents: List[Any]):
#         print(f" [INFO] Building vector store from {len(documents)} raw documents ...")
#         emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
#         chunks = emb_pipe.chunk_documents(documents)
#         embeddings = emb_pipe.embed_chunks(chunks)
#         metadatas = [{"text": chunk.page_content} for chunk in chunks]
#         self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
#         self.save()
#         print(f"[INFO] Vector store built and saved to {self.persist_dir}")

#     def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
#         dim = embeddings.shape[1]
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)
#         if metadatas:
#             self.metadata.extend(metadatas)
#         print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index")

#     def save(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         faiss.write_index(self.index, faiss_path)
#         with open(meta_path, "wb") as f:
#             pickle.dump(self.metadata, f)
#         print(f"[INFO] Saved Faiss index and metadara to {self.persist_dir}")

#     def load(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         self.index = faiss.read_index(faiss_path)
#         with open(meta_path, "rb") as f:
#             self.metadata = pickle.load(f)
#         print(f"[INFO] Loaded Faiss index and metadata from{self.persist_dir}")

#     def search(self, query_embedding: np.ndarray, top_k: int = 5):
#         D, I = self.index.search(query_embedding, top_k)
#         results = []
#         for idx, dist in zip(I[0], D[0]):
#             meta = self.metadata[idx] if idx < len(self.metadata) else None
#             results.append({"index": idx, "distance": dist, "metadata": meta})
#         return results

#     def query(self, query_text: str, top_k: int = 5):
#         print(f"[INFO] Querying vector store for: '{query_text}'")
#         query_emb= self.model.encode([query_text]).astype('float32')
#         return self.search(query_emb, top_k=top_k)

"""
Vector Store Manager using ChromaDB for document storage and retrieval
"""

import os
import uuid
import chromadb
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings in ChromaDB"""

    def __init__(
        self,
        collection_name: str = "company_documents",
        persist_directory: str = "./data/vector_store"
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Company PDF embeddings for voice bot RAG"}
            )

            logger.info(f"✅ Vector store initialized: {self.collection_name}")
            logger.info(
                f"📊 Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            raise

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        batch_size: int = 100
    ):
        """
        Add documents and embeddings to the vector store in batches

        Args:
            documents: List of LangChain Document objects
            embeddings: Corresponding embeddings (numpy array)
            batch_size: Number of documents to add per batch
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings")

        logger.info(f"Adding {len(documents)} documents to vector store...")

        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                # Generate unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i+j}"
                ids.append(doc_id)

                # Prepare metadata
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i + j
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                # Document content
                documents_text.append(doc.page_content)

                # Embedding
                embeddings_list.append(embedding.tolist())

            # Add batch to collection
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
                logger.info(
                    f"✅ Added batch {i//batch_size + 1}: {len(batch_docs)} documents")

            except Exception as e:
                logger.error(f"❌ Error adding batch to vector store: {e}")
                raise

        logger.info(f"🎉 Successfully added all documents")
        logger.info(
            f"📊 Total documents in collection: {self.collection.count()}")

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Company PDF embeddings for voice bot RAG"}
            )
            logger.info(f"🗑️ Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the vector store

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return

        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error querying vector store: {e}")
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }


if __name__ == "__main__":
    # Test vector store
    logger.info("Testing VectorStore...")
    vs = VectorStore()
    info = vs.get_collection_info()
    print(f"\n✅ Vector Store Info:")
    print(f"   Collection: {info['name']}")
    print(f"   Document count: {info['count']}")
    print(f"   Directory: {info['persist_directory']}")
