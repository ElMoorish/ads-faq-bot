# FAQ Chatbot - Production (Supabase via HTTP)
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
import base64

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
    .analysis-card {
        background-color: #1a1d27;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .score-good { color: #10B981; font-weight: bold; }
    .score-medium { color: #F59E0B; font-weight: bold; }
    .score-bad { color: #EF4444; font-weight: bold; }
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
        if ".storage.supabase" not in url:
            self.storage_url = url.replace(".supabase.co", ".storage.supabase.co")
        else:
            self.storage_url = url
    
    def list_files(self, bucket: str):
        try:
            resp = httpx.post(
                f"{self.storage_url}/storage/v1/object/list/{bucket}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefix": ""}
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
    st.title("📚 Ads Mastery")
    st.markdown("---")
    
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    supabase_url = st.secrets.get("SUPABASE_URL", "")
    supabase_key = st.secrets.get("SUPABASE_SERVICE_KEY", "")
    
    if groq_key and supabase_url and supabase_key:
        st.success("✅ Ready to chat!")
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **FAQ Chat** - Ask ads questions
    - **Creative Analysis** - Upload & analyze
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
    """)

# Main content - Tabs
tab1, tab2 = st.tabs(["💬 FAQ Chat", "🎨 Creative Analysis"])

# ==================== TAB 1: FAQ CHAT ====================

with tab1:
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
        sb_http = SupabaseHTTP(supabase_url, supabase_key)
        
        with st.spinner("Loading AI models..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
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
                                pass
                            
                            os.unlink(tmp_path)
                
                st.info(f"📄 Loaded {len(documents)} document sections")
                
            except Exception as e:
                st.error(f"Error loading documents: {e}")
                return None, None
        
        if not documents:
            st.error("No documents found!")
            return None, None
        
        with st.spinner("Processing documents..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
        
        with st.spinner("Building knowledge base..."):
            persist_dir = os.path.join(tempfile.gettempdir(), "chroma_ads")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        
        with st.spinner("Connecting to AI..."):
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
        
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
- If the answer is not in the context, say "I don't have specific information about that in the guides"
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
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("📖 Sources"):
                            for source in message["sources"]:
                                st.markdown(f"- **{source}**")
            
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
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

# ==================== TAB 2: CREATIVE ANALYSIS ====================

with tab2:
    st.title("🎨 Creative Analysis")
    st.markdown("Upload your ad creative for AI-powered analysis and recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Creative")
        uploaded_file = st.file_uploader(
            "Upload ad creative (image)",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a Facebook, Instagram, or TikTok ad creative for analysis"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Your Creative", use_container_width=True)
    
    with col2:
        st.markdown("### ⚙️ Analysis Settings")
        
        platform = st.selectbox(
            "Platform",
            ["Meta (Facebook/Instagram)", "TikTok", "Both"],
            help="Select the platform this creative is for"
        )
        
        niche = st.selectbox(
            "Niche/Industry",
            ["General", "E-commerce", "Real Estate", "Coaching/Consulting", "SaaS/Apps", "Local Business", "Affiliate Marketing"],
            help="Select your niche for tailored recommendations"
        )
        
        analysis_type = st.multiselect(
            "Analysis Focus",
            ["Hook/Attention", "Visual Design", "Copy/Text", "Call-to-Action", "Overall Score"],
            default=["Hook/Attention", "Visual Design", "Overall Score"],
            help="Select what aspects to analyze"
        )
    
    if uploaded_file and st.button("🔍 Analyze Creative", type="primary"):
        with st.spinner("Analyzing your creative..."):
            try:
                # Convert image to base64
                image_bytes = uploaded_file.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Use Groq vision model for analysis
                from openai import OpenAI
                
                client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_key
                )
                
                # Build analysis prompt based on settings
                analysis_focus = ", ".join(analysis_type) if analysis_type else "overall performance"
                
                system_prompt = f"""You are an expert ads creative analyst specializing in {platform} advertising for the {niche} niche.

Analyze the uploaded ad creative and provide:

1. **HOOK SCORE (1-10)**: How well does it grab attention in the first 1-3 seconds?
2. **VISUAL SCORE (1-10)**: Quality of design, colors, composition
3. **COPY SCORE (1-10)**: Text clarity, persuasiveness, call-to-action
4. **OVERALL SCORE (1-10)**: Overall effectiveness prediction

For each score, explain WHY and give 2-3 specific improvements.

Focus areas: {analysis_focus}

Platform: {platform}
Niche: {niche}

Be brutally honest. The goal is to help improve conversion rates.

Format your response with clear sections and emoji headers."""
                
                # Call vision API (llava for image analysis)
                response = client.chat.completions.create(
                    model="llava-v1.5-7b-4096-preview",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this ad creative and give me actionable feedback to improve performance."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000
                )
                
                analysis = response.choices[0].message.content
                
                # Display analysis
                st.markdown("### 📊 Analysis Results")
                st.markdown(analysis)
                
                # Download button
                st.download_button(
                    "📥 Download Analysis",
                    analysis,
                    file_name=f"creative_analysis_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                if "model" in str(e).lower() or "decommissioned" in str(e).lower():
                    st.info("💡 Vision model issue. Check Groq API status or try again later.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #6B7280; font-size: 14px;">
        Built with 🍋 by <a href="https://github.com/ElMoorish" target="_blank" style="color: #3B82F6; text-decoration: none; font-weight: 600;">ElMoorish</a>
        <br/>
        <a href="https://github.com/ElMoorish/ads-faq-bot" target="_blank" style="color: #6B7280; text-decoration: none;">⭐ Star on GitHub</a>
    </div>
    """, unsafe_allow_html=True)
