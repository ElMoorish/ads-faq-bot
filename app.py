# FAQ Chatbot - Zero Cost
# Uses: Groq (free LLM), HuggingFace (free embeddings), ChromaDB (local vector store)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
import tempfile

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

# Sidebar
with st.sidebar:
    st.title("📚 Ads Mastery FAQ")
    st.markdown("---")
    
    # API Key input (check secrets first)
    default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    groq_key = st.text_input(
        "Groq API Key (Free)", 
        value=default_key,
        type="password",
        help="Get free key at groq.com"
    )
    
    if not groq_key:
        st.info("👆 Enter your free Groq API key to start")
        st.markdown("""
        ### Get Free API Key:
        1. Go to [groq.com](https://groq.com)
        2. Sign in with Google
        3. Copy your API key
        4. Paste above
        """)
        st.markdown("---")
        st.markdown("### 📖 Source Documents")
        st.markdown("""
        - Mastering Meta Ads 2026
        - Mastering TikTok Ads 2026
        - Ad Profits Unlocked
        - Monetizing Strategies
        """)
    
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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Load and process documents
@st.cache_resource
def load_documents(pdf_dir: str):
    """Load PDFs and TXTs from directory"""
    documents = []
    
    for file in Path(pdf_dir).glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")
    
    for file in Path(pdf_dir).glob("*.txt"):
        try:
            loader = TextLoader(str(file))
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")
    
    return documents

@st.cache_resource
def create_vectorstore(documents, _embeddings, persist_dir: str):
    """Create vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=_embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

# Initialize the system
def initialize_system(groq_key: str):
    """Initialize embeddings, vectorstore, and QA chain"""
    
    # Use temp directory for ChromaDB (Streamlit Cloud is read-only)
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_db_ads")
    os.makedirs(persist_dir, exist_ok=True)
    
    # Free embeddings from HuggingFace
    with st.spinner("Loading AI models..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Load documents
    pdf_dir = "./pdfs"
    if not os.path.exists(pdf_dir):
        st.error("❌ PDFs folder not found!")
        return None, None
    
    with st.spinner("Loading documents..."):
        documents = load_documents(pdf_dir)
    
    if not documents:
        st.error("❌ No documents found in pdfs folder!")
        return None, None
    
    st.info(f"📄 Loaded {len(documents)} document sections")
    
    # Create vectorstore
    with st.spinner("Creating knowledge base..."):
        vectorstore = create_vectorstore(documents, embeddings, persist_dir)
    
    # Free LLM from Groq
    with st.spinner("Connecting to AI..."):
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7
        )
    
    # Custom prompt
    prompt_template = """
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
"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, vectorstore

# Chat interface
if groq_key:
    # Initialize on first run
    if not st.session_state.initialized:
        with st.spinner("Initializing Ads Mastery AI..."):
            st.session_state.qa_chain, st.session_state.vectorstore = initialize_system(groq_key)
        
        if st.session_state.qa_chain:
            st.session_state.initialized = True
            st.success("✅ Ready! Ask your ads questions below.")
        else:
            st.error("❌ Failed to initialize. Check that PDFs are in the pdfs folder.")
    
    # Only show chat if initialized
    if st.session_state.initialized and st.session_state.qa_chain:
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
                        result = st.session_state.qa_chain.invoke({"query": prompt})
                        answer = result["result"]
                        
                        # Extract sources
                        sources = list(set([
                            doc.metadata.get("source", "Unknown").split("/")[-1]
                            for doc in result.get("source_documents", [])
                        ]))
                        
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
                        st.info("💡 Try refreshing the page if the error persists.")

else:
    # Show demo interface when no API key
    st.info("👈 Enter your free Groq API key in the sidebar to start chatting")
    
    # Show demo chat
    st.markdown("### Preview")
    with st.chat_message("user"):
        st.markdown("How do I set up my first Meta ad campaign?")
    with st.chat_message("assistant"):
        st.markdown("""
        Based on the Mastering Meta Ads 2026 guide:
        
        **Step 1: Campaign Objective**
        Choose your objective (Awareness, Traffic, or Conversions)
        
        **Step 2: Audience Targeting**
        Start with interest-based targeting, then refine with lookalikes
        
        **Step 3: Budget**
        The guide recommends starting with $5-10/day for testing
        
        **Step 4: Creative**
        Use attention-grabbing visuals in the first 3 seconds
        
        💡 *Enter your API key above to get real answers!*
        """)
    
    # CTA for demo users
    st.markdown("""
    <div class="cta-box">
        <h4>📚 Want the Complete System?</h4>
        <p>Get all 4 PDF guides + AI chatbot access + community!</p>
        <a href="https://gumroad.com/l/ads-mastery-bundle-2026" target="_blank">
            → Get Ads Mastery Bundle (Launch Price: $47)
        </a>
    </div>
    """, unsafe_allow_html=True)
