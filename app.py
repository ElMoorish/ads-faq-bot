# FAQ Chatbot - Production (Supabase via HTTP)
# Uses: Groq (free LLM), HuggingFace (free embeddings), Supabase pgvector

import os
# Disable OpenAI completely - we use Groq + HuggingFace
os.environ["OPENAI_API_KEY"] = ""
os.environ.pop("OPENAI_API_KEY", None)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
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
    .source-box {
        background-color: #0f1117;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 12px;
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
class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    
    def list_files(self, bucket: str):
        """List files in storage bucket"""
        resp = httpx.post(
            f"{self.url}/storage/v1/object/list/{bucket}",
            headers=self.headers,
            json={}
        )
        resp.raise_for_status()
        return resp.json()
    
    def download_file(self, bucket: str, path: str):
        """Download file from storage"""
        resp = httpx.get(
            f"{self.url}/storage/v1/object/{bucket}/{path}",
            headers={k: v for k, v in self.headers.items() if k != "Content-Type"}
        )
        resp.raise_for_status()
        return resp.content
    
    def check_documents(self):
        """Check if documents table has data"""
        resp = httpx.get(
            f"{self.url}/rest/v1/documents?select=id&limit=1",
            headers={**self.headers, "Prefer": "count=exact"}
        )
        if resp.status_code == 200:
            data = resp.json()
            return len(data) if data else 0
        return 0

# Sidebar
with st.sidebar:
    st.title("📚 Ads Mastery FAQ")
    st.markdown("---")
    
    # Get secrets
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    supabase_url = st.secrets.get("SUPABASE_URL", "")
    supabase_key = st.secrets.get("SUPABASE_SERVICE_KEY", "")
    
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found in secrets")
    if not supabase_url:
        st.error("❌ SUPABASE_URL not found in secrets")
    if not supabase_key:
        st.error("❌ SUPABASE_SERVICE_KEY not found in secrets")
    
    if groq_key and supabase_url and supabase_key:
        st.success("✅ All credentials configured")
    
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
    
    Includes:
    - 4 PDF Guides
    - AI Chatbot Access
    - 8 Templates
    - Community Access
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

# Format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize the system
def initialize_system(groq_key: str, supabase_url: str, supabase_key: str):
    """Initialize Supabase connection, embeddings, and QA chain"""
    
    # Connect to Supabase via HTTP
    with st.spinner("Connecting to database..."):
        client = SupabaseClient(supabase_url, supabase_key)
    
    # Free embeddings from HuggingFace
    with st.spinner("Loading AI models..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Check if documents exist in Supabase
    with st.spinner("Checking knowledge base..."):
        try:
            doc_count = client.check_documents()
        except Exception as e:
            st.warning(f"Could not check documents: {e}")
            doc_count = 0
    
    # Import supabase only for vectorstore (it handles this internally)
    import supabase as sb
    sb_client = sb.create_client(supabase_url, supabase_key)
    
    if doc_count == 0:
        st.warning("⚠️ No documents in database. Loading from PDFs...")
        
        # Download PDFs from Supabase storage
        with st.spinner("Loading documents from storage..."):
            try:
                # List files in pdfs bucket
                files = client.list_files("pdfs")
                documents = []
                
                for file in files:
                    filename = file.get("name", "")
                    if filename.endswith(('.pdf', '.txt')):
                        # Download file
                        data = client.download_file("pdfs", filename)
                        
                        # Save to temp file for loading
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name
                        
                        # Load document
                        if filename.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_path)
                        else:
                            loader = TextLoader(tmp_path)
                        
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['source'] = filename
                        documents.extend(docs)
                        
                        os.unlink(tmp_path)
                
                st.info(f"📄 Loaded {len(documents)} document sections")
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                # Store in Supabase
                with st.spinner("Building knowledge base (this may take a few minutes)..."):
                    vectorstore = SupabaseVectorStore.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        client=sb_client,
                        table_name="documents",
                        query_name="search_documents"
                    )
                
                st.success(f"✅ Knowledge base built with {len(splits)} chunks")
                
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                st.code(str(e))
                return None, None
    else:
        st.info(f"📚 Knowledge base ready ({doc_count} chunks)")
        vectorstore = SupabaseVectorStore(
            client=sb_client,
            embedding=embeddings,
            table_name="documents",
            query_name="search_documents"
        )
    
    # Free LLM from Groq
    with st.spinner("Connecting to AI..."):
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7
        )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create modern chain with LCEL
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
- End with a relevant follow-up question or tip

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
    # Initialize on first run
    if not st.session_state.initialized:
        with st.spinner("Initializing Ads Mastery AI..."):
            st.session_state.chain, st.session_state.retriever = initialize_system(groq_key, supabase_url, supabase_key)
        
        if st.session_state.chain:
            st.session_state.initialized = True
            st.success("✅ Ready! Ask your ads questions below.")
        else:
            st.error("❌ Failed to initialize. Check your Supabase configuration.")
    
    # Only show chat if initialized
    if st.session_state.initialized and st.session_state.chain:
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📖 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- **{source}**")
        
        # Chat input
        if prompt := st.chat_input("Ask about Meta & TikTok ads..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get relevant documents for sources
                        docs = st.session_state.retriever.invoke(prompt)
                        sources = list(set([
                            doc.metadata.get("source", "Unknown")
                            for doc in docs
                        ]))
                        
                        # Get answer from chain
                        answer = st.session_state.chain.invoke(prompt)
                        
                        st.markdown(answer)
                        
                        # Show sources
                        with st.expander("📖 Sources"):
                            for source in sources:
                                st.markdown(f"- **{source}**")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        # Show CTA after a few questions
                        if len(st.session_state.messages) >= 4:
                            st.markdown("""
                            <div class="cta-box">
                                <h4>📚 Want the Complete System?</h4>
                                <p>Get all 4 PDF guides, templates, and join our community!</p>
                                <a href="https://gumroad.com/l/ads-mastery-bundle-2026" target="_blank">
                                    → Get Ads Mastery Bundle (98% OFF)
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.code(str(e))
                        st.info("💡 Try refreshing the page if the error persists.")

else:
    # Show setup instructions
    st.error("⚠️ Missing credentials! Add these to Streamlit secrets:")
    st.code("""
GROQ_API_KEY = "your_groq_key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_KEY = "your_service_role_key"
    """)
    
    st.markdown("""
    ### Setup Steps:
    
    1. **Groq API Key** - Get free at [groq.com](https://groq.com)
    2. **Supabase Project** - Create at [supabase.com](https://supabase.com)
    3. **Run SQL Setup** - Execute `supabase-setup.sql` in SQL Editor
    4. **Upload PDFs** - Add to `pdfs` storage bucket
    5. **Get Keys** - From Project Settings → API
    """)
