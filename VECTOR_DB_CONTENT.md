# Vector DB Content Plan

## Current Content (Already in PDFs)

### 4 Core PDFs ✅
1. **Ad_Profits_Unlocked_Final.pdf** (504 KB) - Monetization strategies
2. **Mastering_Meta_Ads_2026_5DollarDay.pdf** (57 KB) - Budget Meta campaigns
3. **Mastering_TikTok_Ads_2026_Ebook.pdf** (1.4 MB) - Complete TikTok ads guide
4. **Mastering Meta Ads 2026 - The Profitability Playbook.pdf** (597 KB) - Advanced Meta strategies

### 3 Supporting TXT Files ✅
5. **Mastering Meta Ads for Profitability.txt** (51 KB)
6. **Mastering TikTok Ads Fast.txt** (59 KB)
7. **Monetizing TikTok and Meta Ads.txt** (35 KB)

**Total: ~2.7 MB of content**

---

## Recommended Additions (Free to Add)

### Meta Official Documentation
| Document | URL | Why Add |
|----------|-----|---------|
| Ads Policies | business.facebook.com/ads/policies | Prevent ad rejections |
| Commerce Policies | facebook.com/policies/commerce | E-commerce rules |
| Targeting Specs | developers.facebook.com/docs/marketing-api/targeting-specs | Advanced targeting |
| Creative Best Practices | facebook.com/business/ads/creative-best-practices | Better ad designs |
| iOS 14+ Guide | facebook.com/business/ios-14-guidance | Attribution fixes |

### TikTok Official Documentation
| Document | URL | Why Add |
|----------|-----|---------|
| Ads Policies | ads.tiktok.com/i18n/official/policy | Compliance |
| Creative Center | ads.tiktok.com/creativecenter | Trending formats |
| Targeting Options | ads.tiktok.com/help/targeting | Audience building |
| Video Specs | ads.tiktok.com/help/video-specs | Technical requirements |

### Bonus Content (Create Once, Add Forever)

| Content | Description | Size |
|---------|-------------|------|
| **FAQ Bank** | 50 common Q&A from your experience | ~20 KB |
| **Glossary** | Ads terminology definitions | ~10 KB |
| **Checklists** | Campaign setup checklists | ~5 KB |
| **Template Library** | Ad copy templates | ~15 KB |

---

## How to Add Official Docs (Free)

### Method 1: Copy-Paste to TXT
```
1. Go to facebook.com/business/help
2. Find relevant article
3. Copy text
4. Create new file: docs/meta-ads-policies.txt
5. Paste and save
```

### Method 2: Web Scraping (Automated)
```python
import requests
from bs4 import BeautifulSoup

# Simple scraper for Meta help articles
url = "https://www.facebook.com/business/help/..."
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()

with open('meta-help-article.txt', 'w') as f:
    f.write(text)
```

### Method 3: PDF Export
```
1. Open official docs in browser
2. Print → Save as PDF
3. Add to pdfs/ folder
```

---

## Content to Create (You Can Do This)

### 1. FAQ Bank (Add to vector DB)
```
Q: What's the minimum budget for Meta ads?
A: Start with $5/day per ad set. Meta's minimum daily budget is $1 for impressions, $5 for conversions.

Q: How long should TikTok ads be?
A: 15-30 seconds perform best. Hook in first 3 seconds is critical.

Q: When should I use lookalike audiences?
A: After you have 1,000+ purchases or 10,000+ website visitors.

Q: What's better: Meta or TikTok ads?
A: Depends on your audience. TikTok: <35 years old. Meta: all ages, better for 35+.

Q: How do I fix iOS 14 tracking issues?
A: Use Conversions API, verify domain, use 8 conversion events max.

[... continue with 50+ Q&As]
```

### 2. Glossary (Add to vector DB)
```
A/B Testing - Running two ad variations to see which performs better
Ad Set - Group of ads targeting the same audience
CPM - Cost per 1,000 impressions
ROAS - Return on Ad Spend (revenue / ad spend)
Lookalike Audience - Audience similar to your current customers
Retargeting - Showing ads to people who visited your site
Pixel - Code that tracks user actions on your website
Creative - The visual/text elements of an ad
CTR - Click-through rate (clicks / impressions)
Conversion - Desired action (purchase, signup, etc.)

[... continue with 100+ terms]
```

### 3. Checklists (Add to vector DB)
```
# Meta Ads Launch Checklist

□ Business Manager created
□ Pixel installed and verified
□ Domain verified
□ Conversion events set up (8 max)
□ Audience research complete
□ Lookalike audiences created
□ Ad creatives designed (3+ variations)
□ Ad copy written (3+ variations)
□ Campaign objective selected
□ Budget set ($5/day minimum per ad set)
□ Tracking parameters added (UTM)
□ Landing page optimized
□ A/B test plan created
□ Review policies before publishing
□ Launch and monitor after 24 hours

# TikTok Ads Launch Checklist

□ TikTok Ads Manager account
□ Pixel installed
□ Audience research complete
□ Video created (15-30 seconds)
□ Hook in first 3 seconds
□ Music/sound selected
□ Text overlay added
□ CTA button selected
□ Budget set ($20/day minimum)
□ Targeting configured
□ Landing page mobile-optimized
□ TikTok policies reviewed
□ Launch and test for 48 hours
```

---

## Vector DB Organization

```
pdfs/
├── core-guides/
│   ├── Ad_Profits_Unlocked_Final.pdf
│   ├── Mastering_Meta_Ads_2026_5DollarDay.pdf
│   ├── Mastering_TikTok_Ads_2026_Ebook.pdf
│   └── Mastering_Meta_Ads_2026_Profitability_Playbook.pdf
│
├── supplementary/
│   ├── Mastering_Meta_Ads_for_Profitability.txt
│   ├── Mastering_TikTok_Ads_Fast.txt
│   └── Monetizing_TikTok_Meta_Ads.txt
│
├── official-docs/
│   ├── meta-ads-policies.txt
│   ├── meta-creative-best-practices.txt
│   ├── meta-ios14-guide.txt
│   ├── tiktok-ads-policies.txt
│   └── tiktok-video-specs.txt
│
└── bonus-content/
    ├── faq-bank.txt
    ├── ads-glossary.txt
    ├── meta-checklist.txt
    └── tiktok-checklist.txt
```

---

## Priority Order

### Must Have (Already Done ✅)
1. 4 Core PDFs
2. 3 Supporting TXT files

### Should Add (High Value)
3. Meta Ads Policies (prevents rejections)
4. TikTok Ads Policies (compliance)
5. FAQ Bank (instant answers)

### Nice to Have
6. Glossary
7. Checklists
8. Official best practices

### Can Wait
9. Detailed technical docs
10. API documentation

---

## Implementation

### Step 1: Create Folders
```bash
mkdir projects/faq-chatbot/pdfs/core-guides
mkdir projects/faq-chatbot/pdfs/official-docs
mkdir projects/faq-chatbot/pdfs/bonus-content
```

### Step 2: Move Existing Files
```bash
# Move PDFs to core-guides
mv *.pdf core-guides/

# Move TXTs to supplementary
mv *.txt supplementary/ (or keep in pdfs/ root)
```

### Step 3: Add New Content
```bash
# Create bonus files
touch bonus-content/faq-bank.txt
touch bonus-content/ads-glossary.txt
touch bonus-content/checklists.txt

# Paste content from templates above
```

### Step 4: Update app.py
```python
# Change from:
pdf_dir = "./pdfs"

# To:
pdf_dirs = [
    "./pdfs/core-guides",
    "./pdfs/supplementary", 
    "./pdfs/official-docs",
    "./pdfs/bonus-content"
]

# Load from all directories
for pdf_dir in pdf_dirs:
    if os.path.exists(pdf_dir):
        documents.extend(load_documents(pdf_dir))
```

---

Created for Master A | 2026-02-21
