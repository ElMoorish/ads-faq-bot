-- Ads Mastery FAQ Bot - Supabase Setup
-- Run this in Supabase SQL Editor

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table for PDF content
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(384),  -- HuggingFace all-MiniLM-L6-v2 dimension
    source TEXT,
    page_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create vector index for fast similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Create storage bucket for PDFs
INSERT INTO storage.buckets (id, name, public)
VALUES ('pdfs', 'pdfs', false)
ON CONFLICT (id) DO NOTHING;

-- 5. Allow service role to access bucket (for backend)
CREATE POLICY "Service role can access PDFs"
ON storage.objects FOR ALL
TO service_role
USING (bucket_id = 'pdfs');

-- 6. RLS policies for documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (backend uses this)
CREATE POLICY "Service role full access"
ON documents FOR ALL
TO service_role
USING (true);

-- 7. Function to search documents by similarity
CREATE OR REPLACE FUNCTION search_documents(
    query_embedding VECTOR(384),
    match_count INTEGER DEFAULT 4
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    source TEXT,
    page_number INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        d.source,
        d.page_number,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM documents d
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Done! Now:
-- 1. Upload PDFs to storage bucket 'pdfs'
-- 2. Use service_role key in Streamlit secrets
