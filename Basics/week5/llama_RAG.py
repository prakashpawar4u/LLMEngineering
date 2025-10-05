# expert_knowledge_worker.py
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import groq
from typing import List, Dict, Any, Generator

class ExpertKnowledgeWorker:
    def __init__(self, knowledge_base_path: str = "knowledge-base", db_name: str = "vector_db"):
        """
        Initialize the Expert Knowledge Worker with BAAI/bge embeddings and Llama3 via Groq.
        
        Args:
            knowledge_base_path: Path to the knowledge base directory
            db_name: Name for the vector database
        """
        self.knowledge_base_path = knowledge_base_path
        self.db_name = db_name
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.conversation_chain = None
        self.groq_client = None
        
        # Load environment variables
        self._load_environment()
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_llm()
        self._initialize_conversation_chain()
    
    def _load_environment(self):
        """Load environment variables"""
        load_dotenv(override=True)
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
    
    def _initialize_embeddings(self):
        """Initialize BAAI/bge embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store with documents"""
        # Load and process documents
        documents = self._load_documents()
        chunks = self._split_documents(documents)
        
        # Delete existing vector store if it exists
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_name
        )
        print(f"Vectorstore created with {self.vectorstore._collection.count()} documents")
    
    def _load_documents(self) -> List[Document]:
        """Load documents from knowledge base directory"""
        folders = glob.glob(f"{self.knowledge_base_path}/*")
        
        def add_metadata(doc, doc_type):
            doc.metadata["doc_type"] = doc_type
            return doc
        
        text_loader_kwargs = {'encoding': 'utf-8'}
        
        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs=text_loader_kwargs
            )
            folder_docs = loader.load()
            documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
        
        print(f"Loaded {len(documents)} documents")
        print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def _initialize_llm(self):
        """Initialize Llama3 via Groq"""
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama-3.1-8b-instant",
            groq_api_key=self.groq_api_key
        )
        self.groq_client = groq.Groq(api_key=self.groq_api_key)
    
    def _initialize_conversation_chain(self):
        """Initialize the conversation chain with RAG"""
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever=retriever, 
            memory=memory
        )
    
    def query(self, question: str) -> str:
        """
        Query the knowledge worker with a question
        
        Args:
            question: The question to ask
            
        Returns:
            Answer from the knowledge worker
        """
        result = self.conversation_chain.invoke({"question": question})
        return result["answer"]
    
    def get_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context from vector store
        
        Args:
            query: The query to search for
            k: Number of chunks to retrieve
            
        Returns:
            Combined context from relevant documents
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def stream_answer(self, query: str) -> Generator[str, None, str]:
        """
        Stream answer using Groq API with custom formatting
        
        Args:
            query: The question to ask
            
        Yields:
            Chunks of the response as they are generated
        """
        context = self.get_context(query)
        
        stream = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant for answering company knowledge base questions. Use the provided context to answer accurately. If the context doesn't contain the answer, say so."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                },
            ],
            stream=True,
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        return full_response
    
    def chat_interface(self, question: str, history) -> str:
        """
        Wrapper function for Gradio chat interface
        
        Args:
            question: The question from the user
            history: Chat history (for Gradio compatibility)
            
        Returns:
            Answer from the knowledge worker
        """
        return self.query(question)
    
    def launch_gradio_interface(self, share: bool = False, inbrowser: bool = True):
        """
        Launch Gradio chat interface
        
        Args:
            share: Whether to create a public share link
            inbrowser: Whether to open in browser automatically
        """
        interface = gr.ChatInterface(
            fn=self.chat_interface,
            type="messages",
            title="Expert Knowledge Worker",
            description="Ask questions about the company knowledge base"
        )
        
        interface.launch(share=share, inbrowser=inbrowser)
    
    def reset_conversation(self):
        """Reset conversation memory"""
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever=retriever, 
            memory=memory
        )


def main():
    """Main function to run the Expert Knowledge Worker"""
    try:
        # Initialize the knowledge worker
        knowledge_worker = ExpertKnowledgeWorker()
        
        print("Expert Knowledge Worker initialized successfully!")
        print("Available methods:")
        print("- knowledge_worker.query('Your question') for direct queries")
        print("- knowledge_worker.stream_answer('Your question') for streaming responses")
        print("- knowledge_worker.launch_gradio_interface() for web interface")
        
        # Example usage
        while True:
            print("\n" + "="*50)
            print("1. Ask a question")
            print("2. Launch web interface")
            print("3. Exit")
            choice = input("Choose an option (1-3): ").strip()
            
            if choice == "1":
                question = input("Enter your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Choose response type
                print("\nResponse type:")
                print("1. Standard response")
                print("2. Streaming response")
                response_type = input("Choose response type (1-2): ").strip()
                
                if response_type == "1":
                    answer = knowledge_worker.query(question)
                    print(f"\nAnswer: {answer}")
                elif response_type == "2":
                    print("\nStreaming response:")
                    full_response = ""
                    for chunk in knowledge_worker.stream_answer(question):
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    print()  # New line after streaming
                else:
                    print("Invalid choice, using standard response")
                    answer = knowledge_worker.query(question)
                    print(f"\nAnswer: {answer}")
            
            elif choice == "2":
                print("Launching web interface...")
                knowledge_worker.launch_gradio_interface()
                break
            
            elif choice == "3":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"Error initializing Expert Knowledge Worker: {e}")
        print("Please make sure:")
        print("1. GROQ_API_KEY is set in your .env file")
        print("2. knowledge-base directory exists with markdown files")
        print("3. Required packages are installed: pip install langchain-groq groq chromadb gradio")


if __name__ == "__main__":
    main()