from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_groq import ChatGroq  # Changed to Groq
from langchain.chains import RetrievalQA
import streamlit as st
from sentence_transformers import SentenceTransformer
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Suppress warnings
import uuid
import asyncio
import os

try:
    loop = asyncio.get_running_loop()
    if loop.is_closed():
        raise RuntimeError
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_website(url):
    try:
        # Setup Selenium in Headless Mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # No UI, runs in background
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"🌐 Fetching {url} using Selenium...")
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        # Get fully rendered page content
        page_source = driver.page_source
        driver.quit()

        # Extract meaningful text using BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")
        text_content = soup.get_text()

        # Improved chunking: Extract meaningful paragraphs
        paragraphs = [p.strip() for p in text_content.split("\n") if len(p.strip()) > 50]

        if not paragraphs:
            print("⚠ No valid text found. The website might be blocking scrapers.")

        return paragraphs

    except Exception as e:
        print(f"❌ Error fetching website data: {e}")
        return []

# Step 2: Index data into ChromaDB
def index_data(chunks):
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = "01crew_content"

        # Get collection by name using get_collection
        try:
            collection = client.get_collection(collection_name)
            print(f"✅ Collection {collection_name} found.")
        except chromadb.errors.CollectionNotFoundError:
            collection = client.create_collection(name=collection_name)
            print(f"🌱 Created new collection {collection_name}.")

        # Add documents to the collection
        for chunk in chunks:
            collection.add(
                documents=[chunk],
                metadatas=[{"source": "01crew"}],
                ids=[str(uuid.uuid4())]
            )

        # Check how many documents exist
        stored_data = collection.get(include=["documents"])
        num_docs = len(stored_data["documents"]) if "documents" in stored_data else 0
        print(f"✅ Indexed {num_docs} documents in ChromaDB.")

        return client, collection, collection_name

    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None, None, None

# Step 3: Set up retrieval with LangChain using Sentence Transformers
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        """Convert a single text query into an embedding"""
        return self.model.encode([text])[0].tolist()

    def embed_documents(self, texts):
        """Convert multiple documents into embeddings"""
        return [self.model.encode([text])[0].tolist() for text in texts]

    def __call__(self, texts):
        """Make class callable (needed by Chroma)"""
        return self.embed_documents(texts)

def setup_retriever(client, collection_name):
    try:
        embeddings = SentenceTransformerEmbeddings(model)

        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # Print stored documents for debugging
        stored_data = vectorstore._collection.get(include=["documents"])
        stored_docs = stored_data["documents"] if "documents" in stored_data else []
        print("📄 Stored documents in ChromaDB:")
        for doc in stored_docs[:3]:  # Show just first 3 to avoid clutter
            print(f"🔹 {doc[:100]}...")

        return retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        return None

def setup_qa_chain(retriever):
    """Set up RetrievalQA with Groq instead of Ollama"""
    try:
        # Hardcode the Groq API key directly in the code
        api_key = "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"  # Replace with your actual API key
        
        if not api_key:
            st.error("⚠ GROQ_API_KEY is not set. Please provide a valid Groq API key.")
            return None
            
        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",  # Use appropriate Groq model
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=1000
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error setting up QA chain: {e}")
        return None

# Step 5: Build the chatbot interface with Streamlit
def chatbot_interface(qa_chain):
    st.title("01crew Chatbot")
    st.subheader("Powered by Groq LLM and ChromaDB")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("Ask me anything about 01crew:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if qa_chain:
            try:
                with st.spinner("Generating response..."):
                    response = qa_chain.invoke({"query": user_input})
                
                # Extract retrieved documents
                retrieved_docs = [doc.page_content for doc in response.get("source_documents", [])]
                
                if retrieved_docs:
                    # Use context to generate response
                    context = "\n".join(retrieved_docs)
                    prompt = f"""You are an AI assistant providing factual answers about 01Crew.
                    Answer ONLY based on the following context:

                    {context}

                    If the context does not contain the answer, reply: "I do not have enough information to answer that."

                    User question: {user_input}
                    """

                    response_text = response["result"]
                else:
                    response_text = "I couldn't find any relevant information in my database."

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Optional: Show sources (can be toggled)
                with st.expander("View sources"):
                    for i, doc in enumerate(retrieved_docs[:5]):  # Show top 5 sources
                        st.markdown(f"*Source {i+1}:*\n{doc[:200]}...")

            except Exception as e:
                st.error(f"Error generating response: {e}")
        
        else:
            st.error("Groq API is not configured correctly. Please check your API key.")

def main():
    st.sidebar.title("Chatbot Settings")
    url = st.sidebar.text_input("Website URL to scrape", "https://www.geeksforgeeks.org/")
    
    if st.sidebar.button("Scrape and Initialize"):
        with st.spinner("Scraping website and initializing database..."):
            chunks = scrape_website(url)
            if not chunks:
                st.error("No data scraped. Please check the URL or try a different website.")
                return

            client, collection, collection_name = index_data(chunks)
            if not client or not collection:
                st.error("ChromaDB initialization failed.")
                return

            retriever = setup_retriever(client, collection_name)
            if not retriever:
                st.error("Retriever setup failed.")
                return

            qa_chain = setup_qa_chain(retriever)
            if not qa_chain:
                st.error("QA chain setup failed. Please check the Groq API key.")
                return
                
            st.session_state.qa_chain = qa_chain
            st.success(f"Successfully scraped {len(chunks)} chunks of content and initialized the chatbot!")
    
    # Use the QA chain from session state if available
    if "qa_chain" in st.session_state:
        chatbot_interface(st.session_state.qa_chain)
    else:
        st.info("Click 'Scrape and Initialize' to start the chatbot.")
        
        # Show instructions for API key setup
        st.sidebar.subheader("API Key Setup")
        st.sidebar.info("""
        The API key is already set in the code.
        If you'd like to change it, modify the `setup_qa_chain` function with your own Groq API key.
        """)

if __name__ == "__main__":
    main()
