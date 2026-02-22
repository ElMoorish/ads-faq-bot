# Production Deployment Guide

## Supabase Setup (5 minutes)

### 1. Create Project
- Go to https://supabase.com
- Create new project "ads-faq-bot"
- ✅ Enable Data API
- ✅ Enable Automatic RLS

### 2. Run SQL Setup
1. Go to **SQL Editor** in Supabase
2. Copy contents of `supabase-setup.sql`
3. Click **Run**

### 3. Upload PDFs
1. Go to **Storage** → Create bucket `pdfs` (should auto-create from SQL)
2. Upload all PDFs/TXTs from `pdfs/` folder
3. Make sure bucket is **private** (not public)

### 4. Get API Keys
1. Go to **Settings** → **API**
2. Copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role key** → `SUPABASE_SERVICE_KEY` (not anon key!)

---

## Streamlit Cloud Setup

### 1. Push to GitHub
```bash
cd projects/faq-chatbot
git add .
git commit -m "Production: Add Supabase integration"
git push
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. New app → Select `ElMoorish/ads-faq-bot`
3. Add secrets:
```toml
GROQ_API_KEY = "gsk_your_key"
SUPABASE_URL = "https://xxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```
4. Deploy

---

## Architecture

```
Streamlit Cloud (Free)
       │
       ├── User asks question
       │
       ▼
    Groq API (Free LLM)
       │
       ▼
HuggingFace Embeddings (Free)
       │
       ▼
   Supabase (Free)
       │
       ├── pgvector (Vector Search)
       │
       └── Storage (Private PDFs)
```

## Costs: $0/month

| Service | Tier | Cost |
|---------|------|------|
| Streamlit Cloud | Free | $0 |
| Groq | Free | $0 |
| HuggingFace | Free | $0 |
| Supabase | Free | $0 |

**Total: FREE forever** 🎉
