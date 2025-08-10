from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from ..config import config
import os
import traceback
import re

class RetrievalEngine:
    def __init__(self):
        print("Initializing enhanced embedding and language models...")
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_ID, 
            google_api_key=config.GOOGLE_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.GENERATOR_MODEL_ID, 
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=config.MAX_NEW_TOKENS
        )
        self.qa_chain = None
        self.vectorstore = None
        print("Models initialized.")

    #Create a custom prompt template optimized for financial data queries.
    def _create_custom_prompt(self) -> PromptTemplate:

        template = """You are a financial data analyst assistant. Use the following pieces of context to answer the question at the end. 

When dealing with financial data:
1. Always look for specific numbers, percentages, and dollar amounts in the context
2. Pay special attention to tables and structured data
3. When comparing periods, clearly state both values and the change
4. If exact figures are mentioned in tables, use them precisely
5. If you cannot find specific numerical data, state that clearly

Context:
{context}

Question: {question}

Instructions:
- Provide specific numbers when available
- Show calculations when comparing periods
- Reference table data when present
- Be precise with financial terminology
- If data is incomplete, explain what's missing

Answer:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _preprocess_documents(self, docs: list[Document]) -> list[Document]:
        """Preprocess documents to enhance table data retrieval."""
        processed_docs = []
        
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Enhance table documents with keywords
            if metadata.get("type", "").startswith("table"):
                # Add financial keywords for better retrieval
                financial_keywords = self._extract_financial_keywords(content)
                if financial_keywords:
                    enhanced_content = f"{content}\n\nKeywords: {', '.join(financial_keywords)}"
                    doc.page_content = enhanced_content
                    metadata["enhanced"] = True
            
            processed_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        
        return processed_docs

    def  _extract_financial_keywords(self, text: str) -> list[str]:
        """Extract financial keywords from text to improve searchability."""
        keywords = []
        
        # Common financial terms
        financial_terms = [
            'gross margin', 'net sales', 'revenue', 'income', 'percentage', 
            'Q1', 'Q2', 'Q3', 'Q4', 'quarter', 'year-over-year', 'billion',
            'million', 'products', 'services', 'iPhone', 'Mac', 'iPad',
            'Americas', 'Europe', 'China', 'Japan', 'Asia Pacific'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term.lower() in text_lower:
                keywords.append(term)
        
        # Extract percentages
        percentage_matches = re.findall(r'\d+\.?\d*\s*%', text)
        keywords.extend(percentage_matches)
        
        # Extract dollar amounts
        dollar_matches = re.findall(r'\$\s*\d+[,\d]*\.?\d*\s*(?:billion|million)?', text)
        keywords.extend(dollar_matches)
        
        return list(set(keywords))

    def build_pipeline(self, docs: list[Document]):
        print("Building enhanced retrieval pipeline...")
        
        # Preprocess documents
        processed_docs = self._preprocess_documents(docs)

        # Check if vectorstore exists
        if os.path.exists(config.VECTORSTORE_PATH):
            print(f"Loading vectorstore from {config.VECTORSTORE_PATH}...")
            self.vectorstore = FAISS.load_local(
                config.VECTORSTORE_PATH, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vectorstore from documents...")
            self.vectorstore = FAISS.from_documents(processed_docs, self.embedding_model)
            print(f"Saving vectorstore to {config.VECTORSTORE_PATH}...")
            self.vectorstore.save_local(config.VECTORSTORE_PATH)

        # Create retrievers with different strategies
        faiss_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.RETRIEVER_K * 2,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        bm25_retriever = BM25Retriever.from_documents(processed_docs)
        bm25_retriever.k = config.RETRIEVER_K

        # Create ensemble retriever with emphasis on semantic search for financial data
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7],  # Favor semantic search for financial data
            search_type="mmr",
            k=config.RETRIEVER_K,
        )

        # Create custom prompt
        custom_prompt = self._create_custom_prompt()

        # Build QA chain with custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": custom_prompt
            }
        )

        print("Enhanced retrieval pipeline is built and ready.")

    def query(self, query_text: str) -> dict:
        if not self.qa_chain:
            raise RuntimeError("Pipeline has not been built yet.")

        print(f"\nExecuting Query: '{query_text}'")
        try:
            # Enhanced query preprocessing
            processed_query = self._preprocess_query(query_text)
            
            # Get relevant documents for debugging
            retrieved_docs = self.qa_chain.retriever.invoke(processed_query)
            
            # Debug: Show retrieved documents with focus on tables
            table_docs = [doc for doc in retrieved_docs if 'table' in doc.metadata.get('type', '')]
            if table_docs:
                print(f"Retrieved {len(table_docs)} table documents:")
                for i, doc in enumerate(table_docs[:3]):  # Show first 3 table docs
                    print(f"--- Table Document {i+1} ---")
                    print(doc.page_content[:800])  # Show more content for tables
                    print("Metadata:", doc.metadata)
                    print("---------------------")
            else:
                print(f"Retrieved {len(retrieved_docs)} documents (no table documents found):")
                for i, doc in enumerate(retrieved_docs[:5]):
                    print(f"--- Document {i+1} ---")
                    print(doc.page_content[:400])
                    print("Metadata:", doc.metadata)
                    print("---------------------")

            # Execute the query
            result = self.qa_chain.invoke({"query": processed_query})
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            traceback.print_exc()
            return {"result": "An error occurred, please check the logs."}

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval of financial data."""
        # Add context keywords for better retrieval
        financial_context_words = []
        
        if any(term in query.lower() for term in ['q3', 'quarter', 'third quarter']):
            financial_context_words.extend(['Q3 2022', 'third quarter', 'June 25 2022'])
        
        if any(term in query.lower() for term in ['margin', 'percentage']):
            financial_context_words.extend(['gross margin', 'percentage'])
        
        if any(term in query.lower() for term in ['sales', 'revenue']):
            financial_context_words.extend(['net sales', 'revenue'])
        
        # Enhance query with context
        if financial_context_words:
            enhanced_query = f"{query} {' '.join(set(financial_context_words))}"
            return enhanced_query
        
        return query