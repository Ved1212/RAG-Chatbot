# RAG Chatbot: Chat with your PDFs (Python + GenAI)

Upload PDFs, ask questions, get answers with page citations.

## Pipeline
PDF -> extract text -> chunk -> embed -> vector DB -> retrieve top-k -> LLM answers from context

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # add API key
streamlit run app.py
```

## Interview talking points
- Why chunking + overlap? Keeps each piece focused; overlap avoids cutting facts in half.
- Why embeddings + cosine similarity? Finds meaning, not just keywords.
- How do you reduce hallucination? Answer only from context, cite sources, say "not found".
- Prompt-injection defence: retrieved text is treated as data.

## Upgrades
- Persistent Chroma (chromadb.PersistentClient) so you don't re-index
- Hybrid search (BM25 + vectors) and a re-ranker
- Evaluation set of Q&A pairs to measure answer accuracy
- Streaming responses, DOCX/web page loaders
