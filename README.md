# Ads Mastery FAQ Chatbot

**Zero-cost FAQ chatbot** that answers questions from your PDF guides.

## Tech Stack (All Free)

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq (Llama 3.1) | Free |
| Embeddings | HuggingFace | Free |
| Vector DB | ChromaDB (local) | Free |
| Frontend | Streamlit | Free |
| Hosting | Streamlit Cloud | Free |

## Local Setup (5 minutes)

### 1. Get Free Groq API Key
1. Go to https://groq.com
2. Sign in with Google (no credit card)
3. Go to API Keys → Create
4. Copy your key

### 2. Install Dependencies
```bash
cd projects/faq-chatbot
pip install -r requirements.txt
```

### 3. Add Your PDFs
```bash
# Create pdfs folder
mkdir pdfs

# Copy your PDFs
cp /path/to/your/pdfs/*.pdf pdfs/
cp /path/to/your/txts/*.txt pdfs/
```

### 4. Run Locally
```bash
streamlit run app.py
```

Open http://localhost:8501

## Deploy to Streamlit Cloud (Free Forever)

### 1. Create GitHub Repo
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ads-faq-bot.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Click "Deploy"

### 3. Add Environment Variable
In Streamlit Cloud dashboard:
1. Go to your app → Settings → Secrets
2. Add:
```toml
GROQ_API_KEY = "your_groq_key_here"
```

3. Reboot app

## Usage

Users can ask questions like:
- "How do I set up a Meta ad campaign?"
- "What's the best TikTok ad strategy?"
- "How to monetize ads effectively?"
- "What budget should I start with?"

The bot answers based ONLY on your PDF content.

## Customize

### Change LLM Model
In `app.py`, change:
```python
model_name="llama-3.1-70b-versatile"  # Options: llama-3.1-8b-instant, mixtral-8x7b-32768
```

### Change Embeddings
```python
model_name="sentence-transformers/all-MiniLM-L6-v2"  # Faster: "all-MiniLM-L6-v2"
```

### Adjust Chunk Size
```python
chunk_size=1000,  # Smaller = more precise, Larger = more context
chunk_overlap=200
```

## Monetization Ideas

1. **Embed on Your Site** - Add as widget to your landing page
2. **Lead Magnet** - Free chat, paid full PDF access
3. **Consultation Funnel** - Chat leads to booking call
4. **White Label** - Build for clients, charge setup fee

## Files Structure

```
faq-chatbot/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── pdfs/               # Your PDF documents
│   ├── meta-ads.pdf
│   ├── tiktok-ads.pdf
│   └── ...
└── chroma_db/          # Vector database (auto-created)
```

## Support

- Groq docs: https://console.groq.com/docs
- LangChain docs: https://python.langchain.com
- Streamlit docs: https://docs.streamlit.io

---

Created for Master A | 2026-02-21
