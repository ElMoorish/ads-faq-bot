# FAQ Chatbot - Production (Supabase via HTTP only)
# Uses: Groq (free LLM), HuggingFace (free embeddings), Supabase pgvector

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ.pop("OPENAI_API_KEY", None)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import tempfile
import httpx
import json

# Page config
st.set_page_config(
    page_title="Ads Mastery FAQ Bot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        background-color: #1a1d27;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .stChatMessage.user {
        border-left: 3px solid #3B82F6;
    }
    .stChatMessage.assistant {
        border-left: 3px solid #10B981;
    }
    .cta-box {
        background: linear-gradient(135deg, #3B82F6 0%, #7C3AED 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
    .cta-box a {
        color: #FCD34D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper class for Supabase via HTTP
class SupabaseHTTP:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}"
        }
        # Storage might be on a different subdomain
        # Try both: project.supabase.co and project.storage.supabase.co
        if ".storage.supabase" not in url:
            self.storage_url = url.replace(".supabase.co", ".storage.supabase.co")
        else:
            self.storage_url = url
    
    def list_files(self, bucket: str):
        try:
            # Use the list endpoint with POST - requires prefix property
            resp = httpx.post(
                f"{self.storage_url}/storage/v1/object/list/{bucket}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefix": ""}  # Empty prefix = list all files
            )
            if resp.status_code != 200:
                st.warning(f"Storage list API returned {resp.status_code}: {resp.text[:300]}")
                return []
            data = resp.json()
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            st.warning(f"Error listing files: {e}")
            return []
    
    def download_file(self, bucket: str, path: str):
        resp = httpx.get(
            f"{self.storage_url}/storage/v1/object/{bucket}/{path}",
            headers=self.headers
        )
        return resp.content if resp.status_code == 200 else None

# Sidebar
with st.sidebar:
    st.title("📚 Ads Mastery FAQ")
    st.markdown("---")
    
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    supabase_url = st.secrets.get("SUPABASE_URL", "")
    supabase_key = st.secrets.get("SUPABASE_SERVICE_KEY", "")
    
    if groq_key and supabase_url and supabase_key:
        st.success("✅ Ready to chat!")
    
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    - How do I set up a Meta ad campaign?
    - What's the best TikTok ad strategy?
    - How to monetize ads effectively?
    - What budget should I start with?
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Get Full Guides")
    st.markdown("""
    [📦 Buy Complete Bundle](https://gumroad.com/l/ads-mastery-bundle-2026)
    """)

# Main content
st.title("🤖 Ads Mastery FAQ Bot")
st.markdown("Ask questions about Meta & TikTok ads based on our expert guides")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_system(groq_key: str, supabase_url: str, supabase_key: str):
    """Initialize embeddings, load docs, and create QA chain"""
    
    # Connect to Supabase via HTTP
    sb_http = SupabaseHTTP(supabase_url, supabase_key)
    
    # Load embeddings
    with st.spinner("Loading AI models..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Download PDFs from Supabase storage
    with st.spinner("Loading documents from storage..."):
        try:
            files = sb_http.list_files("pdfs")
            documents = []
            
            for file in files:
                filename = file.get("name", "")
                if filename.endswith(('.pdf', '.txt')):
                    data = sb_http.download_file("pdfs", filename)
                    if data:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name
                        
                        try:
                            if filename.endswith('.pdf'):
                                loader = PyPDFLoader(tmp_path)
                            else:
                                loader = TextLoader(tmp_path, encoding='utf-8')
                            
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata['source'] = filename
                            documents.extend(docs)
                        except Exception as e:
                            st.warning(f"Could not load {filename}: {e}")
                        
                        os.unlink(tmp_path)
            
            st.info(f"📄 Loaded {len(documents)} document sections")
            
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return None, None
    
    if not documents:
        st.error("No documents found!")
        return None, None
    
    # Split documents
    with st.spinner("Processing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
    
    # Create local Chroma vectorstore
    with st.spinner("Building knowledge base..."):
        persist_dir = os.path.join(tempfile.gettempdir(), "chroma_ads")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    # Create LLM
    with st.spinner("Connecting to AI..."):
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7
        )
    
    # Create retriever and chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert ads consultant helping users master Meta (Facebook/Instagram) and TikTok advertising. 

Answer questions based ONLY on the provided context from our ad mastery guides. Be helpful, specific, and actionable.

Context from guides:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context provided
- Be specific and actionable with clear steps
- Include relevant tips and best practices
- If the answer is not in the context, say "I don't have specific information about that in the guides, but I recommend checking our full PDF guides for more details"
- Format with clear sections, bullet points, and bold text when helpful

Answer:
""")
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# Chat interface
if groq_key and supabase_url and supabase_key:
    if not st.session_state.initialized:
        with st.spinner("Initializing Ads Mastery AI..."):
            st.session_state.chain, st.session_state.retriever = initialize_system(
                groq_key, supabase_url, supabase_key
            )
        
        if st.session_state.chain:
            st.session_state.initialized = True
            st.success("✅ Ready! Ask your ads questions below.")
    
    if st.session_state.initialized and st.session_state.chain:
        # Show chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📖 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- **{source}**")
        
        # Chat input
        if prompt := st.chat_input("Ask about Meta & TikTok ads..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        docs = st.session_state.retriever.invoke(prompt)
                        sources = list(set(doc.metadata.get("source", "Unknown") for doc in docs))
                        answer = st.session_state.chain.invoke(prompt)
                        
                        st.markdown(answer)
                        
                        with st.expander("📖 Sources"):
                            for source in sources:
                                st.markdown(f"- **{source}**")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        if len(st.session_state.messages) >= 4:
                            st.markdown("""
                            <div class="cta-box">
                                <h4>📚 Want the Complete System?</h4>
                                <p>Get all 4 PDF guides + templates + community!</p>
                                <a href="https://gumroad.com/l/ads-mastery-bundle-2026" target="_blank">
                                    → Get Ads Mastery Bundle
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #6B7280; font-size: 14px;">
        Built with 🍋 by <a href="https://github.com/ElMoorish" target="_blank" style="color: #3B82F6; text-decoration: none; font-weight: 600;">ElMoorish</a>
        <br/>
        <a href="https://github.com/ElMoorish/ads-faq-bot" target="_blank" style="color: #6B7280; text-decoration: none;">⭐ Star on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Missing credentials! Add to Streamlit secrets:")
    st.code("""
GROQ_API_KEY = "your_key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_KEY = "your_service_role_key"
    """)
