# from pathlib import Path
# from typing import List, Any
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders.excel import UnstructuredExcel
# from langchain_community.document_loaders import JSONLoader

# def load_all_documents(data_dir:str) -> List[Any]:
#     """
#     Load all supported files from the data directory and convert to LandChain document structure
#     Supported: PDF, TXT, CSV, Excel, Word, JSON
#     """

#     # Use project root data folder
#     data_path = Path(data_dir).resolve()
#     print(f"[DEBUG] Data path: {data_path}")
#     documents =[]

#     # PDF files
#     pdf_files = list(data_path.glob('**/*.pdf'))
#     print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
#     for pdf_file in pdf_files:
#         print(f"[DEBUG] Loading PDF: {pdf_file}")
#         try:
#             loader = PyPDFLoader(str(pdf_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

#     return documents

# Text files

# CSV files

# sql files

"""
Document loader for processing PDF files and splitting them into chunks

Document loader for processing PDF and Excel files
Fixed version compatible with your app.py
"""

from pathlib import Path
from typing import List
import pandas as pd
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class matching LangChain interface"""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class DocumentLoader:
    """Handles loading and processing of PDF and Excel documents"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize document loader

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file using pypdf

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects
        """
        try:
            reader = PdfReader(pdf_path)
            documents = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            'source_file': Path(pdf_path).name,
                            'page': page_num,
                            'file_type': 'pdf'
                        }
                    )
                    documents.append(doc)

            logger.info(
                f"✅ Loaded {len(documents)} pages from {Path(pdf_path).name}")
            return documents

        except Exception as e:
            logger.error(f"❌ Error loading {pdf_path}: {e}")
            return []

    def load_excel(self, excel_path: str) -> List[Document]:
        """
        Load a single Excel file using pandas

        Args:
            excel_path: Path to Excel file

        Returns:
            List of Document objects
        """
        try:
            df = pd.read_excel(excel_path)
            text = df.to_string()

            documents = [Document(
                page_content=text,
                metadata={
                    'source_file': Path(excel_path).name,
                    'page': 1,
                    'file_type': 'excel'
                }
            )]

            logger.info(f"✅ Loaded Excel file: {Path(excel_path).name}")
            return documents

        except Exception as e:
            logger.error(f"❌ Error loading {excel_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks with overlap

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("⚠️ No documents to split")
            return []

        chunks = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()

            # Split text into chunks with overlap
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                if chunk_text.strip():
                    chunk = Document(
                        page_content=chunk_text.strip(),
                        metadata=metadata
                    )
                    chunks.append(chunk)

                # Move start position with overlap
                start = end - self.chunk_overlap

                # Break if we're at the end
                if start >= len(text):
                    break

        logger.info(
            f"✂️ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        Process all PDF and Excel files from a directory

        Args:
            pdf_directory: Path to directory containing files

        Returns:
            List of chunked Document objects ready for embedding
        """
        logger.info(f"🚀 Processing files in: {pdf_directory}")

        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            logger.error(f"❌ Directory not found: {pdf_directory}")
            return []

        all_documents = []

        # Process PDF files
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            documents = self.load_pdf(str(pdf_file))
            all_documents.extend(documents)

        # Process Excel files
        excel_files = list(pdf_dir.glob("**/*.xlsx")) + \
            list(pdf_dir.glob("**/*.xls"))
        for excel_file in excel_files:
            documents = self.load_excel(str(excel_file))
            all_documents.extend(documents)

        logger.info(f"📄 Loaded {len(all_documents)} total pages/sheets")

        if not all_documents:
            logger.error("❌ No documents loaded. Check directory contents.")
            return []

        # Split into chunks
        chunks = self.split_documents(all_documents)
        logger.info(f"✅ Processing complete: {len(chunks)} chunks ready")

        return chunks


if __name__ == "__main__":
    # Test document loader
    loader = DocumentLoader()
    chunks = loader.process_pdfs("./data/pdf")
    print(f"Processed {len(chunks)} chunks")
