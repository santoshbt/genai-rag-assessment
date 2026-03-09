# RAG Pipeline

I have mainly focussed on the following layers

1) Ingestion of PDF
      - The uploaded document will be parsed via docling parser, the reason being it maintains a context - aware extraction
      - It maintains the headings, paragraphs, tables, OCR and different formats.
      - It maintains, the structural boudaries, tables are not split mid-row.
      - It can attach meaningful metadata to every chunk.
      - This in turn helps in coherance and accuracy while retrieval.
      - Redacting the sensitive information, even before sending it to the chunking phase
      - Remove any greetings or normal words which may not add up much weightage to the information. This is done to avoid chunking unnecessary data. 
      - It acts as a guardrail for pre-chunking phase and redacts any PII data or sonract details wgile chunking.
      - Tradeoff : It can add up some latency and slower compared to simple parsers
      - It can use more memory due to added metadata
        

2) HybridChunker
      - Semantic chunking to avoid data loss due to fixed length character or sentence level chinking
      - It also uses structural chunking for structural awareness and semantic chunking for meaning based boundary detection
      - Trade off can be it would be bit complex and may be out of control

3) Deduplication
      - After retrieval, compares embeddings of retrieved chunks.
      - If two chunks are > 95% similar (cosine similar), it keeps only one.

4) Embedding layer
      - Using sentence-transformer/MiniLLM
      - 384 dimensional dense vector
      - Pretrained on 1B+ sentence pairs
      - It is free to use and locally installed, hence no API calls
      - Trade off : It is smaller and may not be of good quality compared to higher dimensional model

5) ChromaDB - Dense indexing
      - It is useful for storing the embeddings and dense vector search
      - It also stores the metadata to filter search results before the retrieval to narrow down the relevant documents
      - It finds synonyms and handles paraphrasing
      - It is a fallback when BM25 (Keyword search) is empty
      - It acts as a primary source of text corpus to keyword search (BM25)
      - This application runs chroma db in local, hence no api key is required.
      - Tradeoffs: No advanced features like hybrid search or managed service

6) BM25 - Sparse Indexing
      - It is the keyword matching for simple queries
      - Based on the query length it will route to keyword search
      - It is good for static data like names, Ids or readily available information
      - It is much more faster comare to vector search and 4x faster comparaively
      - When app starts, it fetches all chunks from ChromaDB and loads temporarily into RAM.
      - It builds the search index.
      - Tradeoffs: Now it is in memory only and no persistance

7) ConversationSummaryMemory
      - This is for reference to the prevous conversations in asummarized format
      - This is useful for reducing the context window while invoking the LLM
      - Tradeoffs: It adds up to the LLM calls which may add up to the token budget.
      - It might cause a latency overhead

8) Cross Encoder Retriever
      - This is done to pick only the relevant information from the top k results from the vector search
      - It ranks chunks by relevance
      - Tradeoff: Can add up to the latency

9) Text Generation
      - This step uses the RAG to produce the grounded, factual answers based on the retieved document chunks
      - It is currently configured witht he Groq client and model is llama-3.3-70b-versatile. It is open source and good for development / poc stage
      - LLM is instructed with some specific instructions to act upon and generate the relevant text.
      - We can also pass the hyper-parameters with temperature, Top-p, top-k based on our requirements
      - Tradeoffs: As it is a free tier, it has a rate limit

10) Post generation Guradrails
      - After the LLM generates the answer, the system applies 2 guardrails
      - Faithfulness score - Verifies the answer is grounded in retrieved context
      - Completeness check - Ensures the answer adequately addresses the query
      - It gives the transprent warnings
      - It uses Premetheus metrics to track the quality
      - This acts as an observability layer for the system.
      - Tradeoffs: Sometime there would be variations and might missout. Right now it is manual maintainance and need to update manually for new rule.


pipeline.py orchestrates the complete workflow
Ingestion -> Guardrails → Cache → Memory → Routing → Retrieval → Generation → Post-guardrails


---

****** Next scope of improvements *********
1) Query rewriting to understand typos and user intentions
2) Multi Vector retriever implementation -> Useful when single chunk not being findable from different query styles.
3) Semantic caching, right now I am using only static caching which is not very efficient
4) Use more sophesticated vector DBs like Pinecone in production mode
5) To store persistant metadata we can use PostgreSQL or other vector DBs which supports both vector search and full text search
6) Using premium models for better performance and effeciency.


---


## Setup

```bash
# 1. Clone and enter directory
cd classical_rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY and HF_API_TOKEN

# 5. Start the server
python main.py
# API available at http://localhost:8000
# Prometheus metrics at http://localhost:8001
# Swagger UI at http://localhost:8000/docs
```


pip install streamlit
# Terminal 1 — backend
python main.py

# Terminal 2 — frontend
streamlit run app.py
