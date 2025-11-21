"""
Main pipeline for Voice Bot RAG System
Uses Vertex AI Gemini 2.5 Pro as LLM
"""

import os
import logging
from dotenv import load_dotenv

# --- Vertex AI ---
import vertexai
from vertexai.generative_models import GenerativeModel

# --- RAG Components ---
from src.data_loader import DocumentLoader
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever, RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# 1. Initialize Vertex AI LLM (Gemini 2.5 Pro)
# --------------------------------------------------------------------
def init_vertex_llm():
    load_dotenv()

    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project_id or not location:
        raise ValueError("❌ Missing PROJECT_ID or LOCATION in .env file")

    if not creds_env or not os.path.exists(creds_env):
        raise ValueError(
            "❌ GOOGLE_APPLICATION_CREDENTIALS path is missing or invalid")

    logger.info(f"🔑 Using credentials: {creds_env}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_env

    # Initialize Vertex AI client
    vertexai.init(project=project_id, location=location)

    logger.info("🤖 Initializing Gemini 2.5 Pro...")
    llm = GenerativeModel(
        "gemini-2.5-pro",
        generation_config={"temperature": 0.0}
    )

    return llm


# --------------------------------------------------------------------
# 2. Set up RAG Pipeline
# --------------------------------------------------------------------
def setup_rag_pipeline(
    pdf_directory: str = "../data/pdf",
    force_rebuild: bool = False,
    company_name: str = "Our Company"
):
    logger.info("🚀 Starting RAG Pipeline Setup")

    # Initialize embedding + vector store
    logger.info("📦 Initializing components...")
    embedding_manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5")
    vector_store = VectorStore(collection_name="company_documents")

    # Check collection
    existing_docs = vector_store.get_collection_info().get("count", 0)

    if existing_docs == 0 or force_rebuild:
        logger.info("🔨 Building vector store from PDFs...")

        if force_rebuild and existing_docs > 0:
            logger.info("🗑 Clearing old vector store...")
            vector_store.clear_collection()

        # Load PDFs
        doc_loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
        chunks = doc_loader.process_pdfs(pdf_directory)

        if not chunks:
            logger.error("❌ No PDFs found in directory.")
            return None

        # Generate embeddings
        logger.info("🧠 Generating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(
            texts, batch_size=16)

        # Store in vector DB
        logger.info("💾 Storing embeddings...")
        vector_store.add_documents(chunks, embeddings)
    else:
        logger.info(f"📚 Using existing {existing_docs} embedded documents")

    # Initialize Vertex LLM
    llm = init_vertex_llm()

    # Create Retriever + RAG Pipeline
    logger.info("🔗 Creating RAG pipeline...")
    retriever = RAGRetriever(vector_store, embedding_manager)
    rag_pipeline = RAGPipeline(retriever, llm, company_name=company_name)

    logger.info("✅ RAG Pipeline successfully initialized!")
    return rag_pipeline


# --------------------------------------------------------------------
# 3. Test RAG Pipeline
# --------------------------------------------------------------------
def test_rag_pipeline(rag_pipeline: RAGPipeline):
    test_queries = [
        "What does your company do?",
        "Tell me about your products",
        "What is the weather today?",
        "Who are the key team members?"
    ]

    for query in test_queries:
        print("\n📝 Query:", query)
        print("-" * 60)

        result = rag_pipeline.query(query, top_k=3, min_score=0.1)

        print("🤖 Answer:", result["answer"])
        print("📊 Confidence:", result["confidence"])
        print("📚 Sources:", result["sources"])
        print("\n")


# --------------------------------------------------------------------
# 4. Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    rag = setup_rag_pipeline(
        pdf_directory="./data/pdf",
        force_rebuild=False,
        company_name=os.getenv("COMPANY_NAME")
    )

    if rag:
        test_rag_pipeline(rag)

        print("\n💬 Interactive Mode (type 'quit' to exit)")
        while True:
            query = input("\n🎤 Your question: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break

            result = rag.query(query, top_k=3, min_score=0.1)
            print("\n🤖 Answer:", result["answer"])
            print("📊 Confidence:", result["confidence"])
