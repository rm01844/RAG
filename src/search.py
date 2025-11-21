# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVecorStore
# from langchain_groq import ChatGroq


# load_dotenv()

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_overlap: int = 200):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
#         groq_api_key = "API_KEY"
#         self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
#         print(f"[INFO] Groq LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         resuls = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
#         prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n"""
#         response = self.llm.invoke([invoke])
#         return response.content

"""
RAG Search and Retrieval with topic filtering for company-specific queries
"""

from typing import List, Dict, Any
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Handles retrieval and filtering for RAG queries"""

    def __init__(self, vector_store, embedding_manager):
        """
        Initialize retriever

        Args:
            vector_store: VectorStore instance
            embedding_manager: EmbeddingManager instance
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            top_k: Number of top results
            score_threshold: Minimum similarity score

        Returns:
            List of retrieved documents with metadata
        """
        logger.info(f"🔍 Retrieving documents for: '{query}'")
        logger.info(
            f"   Parameters - Top K: {top_k}, Threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_single_embedding(
            query)

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                logger.info(
                    f"✅ Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                logger.warning("⚠️ No documents found")

            return retrieved_docs

        except Exception as e:
            logger.error(f"❌ Error during retrieval: {e}")
            return []


class RAGPipeline:
    """Complete RAG pipeline with LLM integration and topic filtering"""

    def __init__(self, retriever, llm, company_name: str = "your company"):
        """
        Initialize RAG pipeline

        Args:
            retriever: RAGRetriever instance
            llm: Vertex AI GenerativeModel instance
            company_name: Name of the company for context
        """
        self.retriever = retriever
        self.llm = llm
        self.company_name = company_name
        self.history = []
        self.temp_memory = {}  # session-based memory

    def normalize_query(self, question: str) -> str:
        """Rewrite user query to match PDF phrasing for better retrieval."""

        # Special handling for team/people queries
        team_keywords = ['team', 'member', 'employee', 'staff',
                         'manager', 'director', 'contact', 'email', 'phone']
        is_team_query = any(keyword in question.lower()
                            for keyword in team_keywords)

        if is_team_query:
            prompt = f"""
            Rewrite this question to find specific people information in a document.
            Focus on: names, roles, titles, contact details, departments.
            Keep it keyword-focused.
            
            Question: "{question}"
            
            Rewritten query (keywords only):
            """
        else:
            prompt = f"""
            Rewrite the following question into short, keyword-style search terms
            that match text chunks in a PDF.
            
            Make it specific, direct, and keyword-based.
            No explanations. Only return the rewritten query.

        Question: "{question}"
        """

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return question  # fallback if LLM fails

    def is_company_related(self, query: str) -> bool:
        """
        Simple check if query is company-related
        Can be enhanced with more sophisticated NLP

        Args:
            query: User query

        Returns:
            Boolean indicating if query is company-related
        """
        # Keywords that indicate off-topic queries
        off_topic_keywords = [
            'weather', 'news', 'sports', 'recipe', 'movie', 'game',
            'song', 'celebrity', 'politics', 'joke', 'python code',
            'write code', 'program', 'algorithm'
        ]

        query_lower = query.lower()

        # Check for off-topic keywords
        for keyword in off_topic_keywords:
            if keyword in query_lower:
                return False

        return True

    def extract_facts(self, answer: str):
        """
        Extract simple entity-role facts like:
        'Ben is the Marketing Manager'
        'Saqib works as Sales Manager'

        Store in temp memory ONLY during session.
        """
        import re

        # Pattern: "<name> is the <role>"
        pattern1 = r"([A-Z][a-zA-Z]+)\s+is\s+the\s+([A-Za-z\s]+)"
        matches = re.findall(pattern1, answer)

        for name, role in matches:
            role_clean = role.strip().lower()
            self.temp_memory[role_clean] = name

        # Pattern: "<name> works as <role>"
        pattern2 = r"([A-Z][a-zA-Z]+)\s+works\s+as\s+([A-Za-z\s]+)"
        matches2 = re.findall(pattern2, answer)

        for name, role in matches2:
            role_clean = role.strip().lower()
            self.temp_memory[role_clean] = name

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.1,
        stream: bool = False,
        summarize: bool = False,
        show_citations: bool = False
    ) -> dict:
        """
        Generate response for a query with topic filtering

        Args:
            question: User question
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
            stream: Whether to stream response
            summarize: Whether to include summary

        Returns:
            Dictionary with answer, sources, and metadata
        """
        team_keywords = ['team', 'member', 'employee', 'staff', 'manager', 'director',
                         'contact', 'email', 'phone', 'role', 'title']
        is_team_query = any(keyword in question.lower()
                            for keyword in team_keywords)

        # Adjust parameters for team queries
        if is_team_query:
            top_k = max(top_k, 10)  # Retrieve more documents
            min_score = 0.1  # Lower threshold for team queries
            logger.info(
                f"📋 Team query detected - using top_k={top_k}, min_score={min_score}")

        # Check if query is company-related
        if not self.is_company_related(question):
            return {
                'question': question,
                'answer': f"I'm sorry, but I can only answer questions about {self.company_name}. "
                f"Please ask me something related to our company, products, or services.",
                'sources': [],
                'confidence': 0.0,
                'is_relevant': False,
                'history': self.history
            }

        # Check session memory (roles previously learned)
        lower_q = question.lower()
        for role, name in self.temp_memory.items():
            if role in lower_q:
                return {
                    "question": question,
                    "answer": f"{name} is the {role}. (From session memory)",
                    "sources": [],
                    "confidence": 0.99,
                    "is_relevant": True,
                    "history": self.history
                }

        # Normalize query for better retrieval
        normalized_query = self.normalize_query(question)
        logger.info(f"🔄 Normalized Query: {normalized_query}")

        # Retrieve using normalized query
        results = self.retriever.retrieve(
            normalized_query, top_k=top_k, score_threshold=min_score
        )

        if not results:
            return {
                'question': question,
                'answer': f"I don't have enough information to answer that question about {self.company_name}. "
                f"Could you please rephrase or ask something else?",
                'sources': [],
                'confidence': 0.0,
                'is_relevant': True,
                'history': self.history
            }

        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in results])

        # Prepare sources
        sources = [{
            'source': doc['metadata'].get('source_file', 'unknown'),
            'page': doc['metadata'].get('page', 'N/A'),
            'score': round(doc['similarity_score'], 3),
            'preview': doc['content'][:200] + '...'
        } for doc in results]

        confidence = max([doc['similarity_score'] for doc in results])

        # Detect if this is a team/directory query
        is_team_query = any(kw in question.lower() for kw in [
                            'team', 'member', 'employee', 'staff', 'contact'])

        # Generate answer using Vertex AI
        if is_team_query:
            prompt = f"""You are a helpful assistant for {self.company_name}. 
            Extract team member information from the context below.
            
            When listing team members, use this format:
            - **Name**: [Full Name]
              - Role: [Job Title]
            
            Rules:
            - Owner of the company is not part of the team members list
            - Extract ALL team members mentioned in the context
            - Use ONLY information from the provided context
            - If contact details are not in context, don't include them
            - Present information in clear, structured format
            
            Context:
            {context}
            
            Question: {question}
            
            Answer (extract all relevant team information):"""
        else:
            prompt = f"""You are a helpful assistant for {self.company_name}. 
            Use ONLY the text inside the "Context" section below.

            Rules:
            - Answer based ONLY on the provided context
            - If the context doesn't contain the answer, say: "I cannot find that information in the company documents."
            - Keep your answer clear, professional, and conversational
            - Do not make up information

            Context:
            {context}

            Question: {question}

            Answer (grounded in context only):"""

        try:
            if stream:
                # Streaming response
                answer = ""
                for chunk in self.llm.generate_content(prompt, stream=True):
                    if chunk.text:
                        answer += chunk.text
            else:
                response = self.llm.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0}
                )
                answer = response.text
                self.extract_facts(answer)
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {e}")
            answer = "I'm having trouble generating a response. Please try again."

        # Add citations
        citations = [
            f"[{i+1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        if show_citations and citations:
            answer_final = answer + "\n\nCitations:\n" + "\n".join(citations)
        else:
            answer_final = answer

        # Optional summary
        summary = None
        if summarize:
            try:
                summary_prompt = f"Summarize this answer in 2 sentences:\n{answer}"
                summary_resp = self.llm.generate_content(summary_prompt)
                summary = summary_resp.text
            except:
                pass

        # Save to history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        # Prepare final response
        result = {
            'question': question,
            'answer': answer_final,
            'sources': sources,
            'confidence': round(confidence, 3),
            'is_relevant': True,
            'summary': summary,
            'history': self.history
        }

        logger.info(
            f"✅ Generated response with confidence: {result['confidence']}")
        return result


if __name__ == "__main__":
    # This file requires vector_store, embedding_manager, and llm to test
    logger.info("RAGRetriever and RAGPipeline classes defined")
