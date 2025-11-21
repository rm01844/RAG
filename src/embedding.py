# from typing import List, Any
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from src.data_loader import load_all_documents

# class EmbeddingPipeline:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.model = SentenceTransformer(model_name)
#         print(f"[INFO] Loaded embedding model: {model_name}")

#     def chunk_documents(self, documents: List[Any]) -> List[Any];
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             seperators=["\n\n", "\n", " ", ""]
#         )
#         chunks = splitter.split_documents(documents)
#         print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
#         return chunks

#     def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
#         texts = [chunk.page_content for chunk in chunks]
#         print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         print(f"[INFO] Embeddings shape: {embeddings.shape}")
#         return embeddings

"""
Embedding Manager for generating document embeddings using BAAI/bge-small-en-v1.5
Optimized for Intel MacOS using sentence-transformers
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generates embeddings using BAAI/bge models via sentence-transformers"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding model

        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")

        # Device selection for Intel Mac
        self.device = "cpu"  # Intel Mac - use CPU
        logger.info(f"Using device: {self.device}")

        # Load model using sentence-transformers
        self.model = SentenceTransformer(model_name, device=self.device)

        logger.info(f"✅ Model loaded successfully on {self.device}")
        logger.info(
            f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Generate embeddings (sentence-transformers handles batching and normalization)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.generate_embeddings([text])[0]


if __name__ == "__main__":
    # Test the embedding manager
    logger.info("Testing EmbeddingManager...")
    em = EmbeddingManager()

    test_texts = [
        "This is a test sentence.",
        "Another example for embedding generation."
    ]

    embeddings = em.generate_embeddings(test_texts)
    print(f"\n✅ Test successful!")
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")

    # Test single embedding
    single_emb = em.generate_single_embedding("Single test")
    print(f"Single embedding shape: {single_emb.shape}")
