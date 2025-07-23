import os
import logging
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State for the RAG pipeline"""
    question: str
    retrieved_docs: List[str]
    answer: str
    metadata: Dict[str, Any]

class SimpleRAGSystem:
    def __init__(self, documents_path: str = "documents", openrouter_api_key: str = None):
        """
        Initialize the RAG system with OpenRouter API
        
        Args:
            documents_path: Path to directory containing documents
            openrouter_api_key: OpenRouter API key for unlimited access
        """
        self.documents_path = Path(documents_path)
        self.openrouter_api_key = openrouter_api_key
        self.vector_store = None
        self.llm = None
        self.graph = None
        
        # Create documents directory if it doesn't exist
        self.documents_path.mkdir(exist_ok=True)
        
        # Initialize components
        self._setup_embeddings()
        self._setup_llm()
        self._build_vector_store()
        self._build_graph()

    def _setup_embeddings(self):
        """Setup embeddings model"""
        logger.info("Setting up embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _setup_llm(self):
        """Setup language model with OpenRouter"""
        logger.info("Setting up LLM with OpenRouter...")
        if self.openrouter_api_key:
            self.llm = ChatOpenAI(
                model="microsoft/wizardlm-2-8x22b",  # Free model on OpenRouter
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=1000
            )
        else:
            # Fallback to mock LLM
            logger.warning("No OpenRouter API key provided. Using mock LLM.")
            self.llm = self._create_mock_llm()

    def _create_mock_llm(self):
        """Create a simple mock LLM for demonstration"""
        class MockLLM:
            def invoke(self, messages):
                context = str(messages)
                if "Python" in context:
                    return type('Response', (), {'content': "Python is a popular programming language known for its simplicity and versatility. It features easy-to-read syntax, extensive libraries, and cross-platform compatibility."})()
                elif "machine learning" in context.lower():
                    return type('Response', (), {'content': "Machine learning is a subset of AI with three main types: supervised learning (uses labeled data), unsupervised learning (finds patterns), and reinforcement learning (learns through rewards)."})()
                elif "data science" in context.lower():
                    return type('Response', (), {'content': "Data science combines programming, statistics, and domain expertise. Key skills include Python/R programming, SQL, statistics, visualization, and machine learning."})()
                else:
                    return type('Response', (), {'content': "Based on the provided context, I can help answer questions about programming, machine learning, and data science topics from the knowledge base."})()
        return MockLLM()

    def _build_vector_store(self):
        """Load documents and build vector store"""
        logger.info("Building vector store...")
        
        documents = []
        for txt_file in self.documents_path.glob("*.txt"):
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        
        if not documents:
            raise ValueError(f"No documents found in {self.documents_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        logger.info("Vector store created successfully")

    def _build_graph(self):
        """Build LangGraph workflow"""
        logger.info("Building LangGraph workflow...")
        
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        self.graph = workflow.compile()
        logger.info("LangGraph workflow built successfully")

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieval node - finds relevant documents"""
        logger.info(f"Retrieving documents for: {state['question']}")
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        docs = retriever.invoke(state["question"])
        retrieved_texts = [doc.page_content for doc in docs]
        
        state["retrieved_docs"] = retrieved_texts
        state["metadata"] = {
            "num_retrieved": len(retrieved_texts),
            "sources": [doc.metadata.get("source", "unknown") for doc in docs]
        }
        
        return state

    def _generate_node(self, state: RAGState) -> RAGState:
        """Generation node - creates answer using LLM"""
        logger.info("Generating answer...")
        
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert assistant specializing in Python programming, machine learning, data science, and AI.
Use the provided context to give detailed, accurate answers. Include specific examples and practical insights when relevant.

Context from knowledge base:
{context}

Question: {question}

Provide a comprehensive answer based on the context above. If the context doesn't contain complete information, acknowledge this and provide what information is available.

Answer:
        """)
        
        context = "\n\n".join(state["retrieved_docs"])
        
        if hasattr(self.llm, 'invoke'):
            messages = prompt_template.format_messages(
                context=context,
                question=state["question"]
            )
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
        else:
            formatted_prompt = prompt_template.format(
                context=context,
                question=state["question"]
            )
            answer = self.llm.invoke(formatted_prompt)
        
        state["answer"] = answer
        return state

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question to the RAG system"""
        initial_state = RAGState(
            question=question,
            retrieved_docs=[],
            answer="",
            metadata={}
        )
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final_state["answer"],
            "retrieved_docs": final_state["retrieved_docs"],
            "metadata": final_state["metadata"]
        }

    def interactive_chat(self):
        """Interactive chat interface"""
        print("\n🤖 RAG System Interactive Chat")
        print("=" * 50)
        print("Ask questions about Python, ML, Data Science, and AI!")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the RAG system!")
                    break
                
                if not question:
                    print("❓ Please enter a question.")
                    continue
                
                print("\n🔍 Searching knowledge base...")
                result = self.ask_question(question)
                
                print(f"\n🤖 **Answer:**")
                print(result['answer'])
                print(f"\n📊 Retrieved {result['metadata']['num_retrieved']} relevant chunks")
                sources = [os.path.basename(s) for s in result['metadata']['sources']]
                print(f"📚 Sources: {', '.join(set(sources))}")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using the RAG system!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    # Load environment variables from .env file
    load_dotenv()
    
    print("🚀 Initializing RAG System with OpenRouter...")
    
    # Get OpenRouter API key from environment
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    # If no env variable, you can set it directly here:
    # openrouter_api_key = "your_openrouter_api_key_here"
    
    if not openrouter_api_key or openrouter_api_key == "your_openrouter_api_key_here":
        print("⚠️  No OpenRouter API key found in .env file. Please create a .env file and add: OPENROUTER_API_KEY='your_key_here'")
        print("🔄 Running in demo mode with mock LLM...")
        openrouter_api_key = None
    
    rag_system = SimpleRAGSystem(openrouter_api_key=openrouter_api_key)
    
    print("\n" + "="*80)
    print("🎮 Interactive Mode Starting...")
    print("="*80)
    rag_system.interactive_chat()

if __name__ == "__main__":
    main()