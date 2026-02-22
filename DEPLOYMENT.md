# Chatbot Deployment Guide

## Quick Deploy to Streamlit Cloud (Free)

### Prerequisites
1. Groq API key (free at groq.com)
2. GitHub account
3. Streamlit Cloud account (free at share.streamlit.io)

---

## Step 1: Create GitHub Repository

```bash
cd C:\home\node\.openclaw\workspace\projects\faq-chatbot

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Ads Mastery FAQ Bot"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/ads-faq-bot.git
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `YOUR_USERNAME/ads-faq-bot`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

---

## Step 3: Add Secrets

In Streamlit Cloud dashboard:

1. Go to your app → Settings → Secrets
2. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

3. Click "Save"
4. App will restart

---

## Your URLs

| Service | URL |
|---------|-----|
| **Chatbot** | https://YOUR_APP_NAME.streamlit.app |
| **Community** | https://skool.com/ads-mastery-community-9023 |
| **Gumroad** | https://gumroad.com/l/ads-mastery-bundle-2026 |

---

## Local Testing

```bash
cd C:\home\node\.openclaw\workspace\projects\faq-chatbot

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Troubleshooting

### Error: "No module named 'langchain'"
```bash
pip install langchain langchain-community langchain-groq
```

### Error: "PDFs folder not found"
Make sure pdfs/ folder is in the repository root with your PDFs.

### Error: "Rate limit exceeded"
Groq has rate limits. Wait a minute and try again.

### Streamlit Cloud: App won't start
1. Check logs in Streamlit Cloud dashboard
2. Make sure requirements.txt has all dependencies
3. Check that GROQ_API_KEY is in secrets

---

## File Structure

```
faq-chatbot/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (if needed)
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── .streamlit/
│   ├── config.toml        # Streamlit theme config
│   └── secrets.toml       # Local secrets (don't commit!)
└── pdfs/                  # Your PDF documents
    ├── Ad_Profits_Unlocked_Final.pdf
    ├── Mastering_Meta_Ads_2026_5DollarDay.pdf
    ├── Mastering_TikTok_Ads_2026_Ebook.pdf
    ├── Mastering Meta Ads 2026 - The Profitability Playbook.pdf
    ├── Mastering Meta Ads for Profitability.txt
    ├── Mastering TikTok Ads Fast.txt
    └── Monetizing TikTok and Meta Ads.txt
```

---

## After Deployment

1. Test the chatbot with sample questions
2. Update Gumroad listing with chatbot URL
3. Post on TikTok with chatbot link
4. Update Skool community with chatbot link

---

Created for Master A | 2026-02-21
