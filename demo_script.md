# Demo Video Script
## RAG-Powered AI/ML Domain Expert Chatbot
**Target length:** 7–9 minutes  |  **Recommended tool:** Loom or OBS

---

## Pre-recording checklist
- [ ] App is running at localhost:8501 or live Streamlit URL
- [ ] Browser zoom at 110–125% so text is readable
- [ ] Terminal open to the side (for architecture walkthrough)
- [ ] Dark mode off (better contrast on recording)
- [ ] Silence notifications

---

## SEGMENT 1 — Introduction (0:00–0:40)

**[On-screen: App homepage / landing]**

> "Hi, I'm [Name]. In this demo I'll walk you through our RAG-powered domain
> expert chatbot for Artificial Intelligence and Machine Learning. The system
> retrieves relevant passages from a knowledge base of 55 documents, generates
> grounded answers, and shows the exact sources it used — all without
> hallucinating or making things up."

> "The stack is: LangChain for orchestration, ChromaDB as our persistent vector
> database, OpenAI GPT-4o-mini as the LLM, and Streamlit for the UI. The app
> is deployed publicly on Streamlit Community Cloud."

---

## SEGMENT 2 — Architecture Overview (0:40–2:00)

**[On-screen: Draw or display the pipeline diagram — or show the README]**

> "Let me quickly explain how it works. When a user asks a question, the system
> goes through five stages."

> "First, if this is a follow-up question, we use the conversation history to
> rephrase it into a standalone question — so the retriever gets something
> unambiguous."

> "Second, we run Hybrid Search — our advanced feature. We combine BM25 keyword
> retrieval at 30% weight with ChromaDB semantic search at 70% weight, and merge
> the rankings using Reciprocal Rank Fusion to get the top-5 most relevant chunks."

> "Third, those chunks become the context that GPT-4o-mini uses to generate
> an answer. It cites each source inline with [Source N] notation."

> "The sidebar shows our knowledge base stats — 55 documents, over 4,000 chunks."

**[On-screen: Sidebar — point to doc count and chunk count]**

---

## SEGMENT 3 — Live Demo (2:00–6:00)

### Query 1: Simple factual question
**[Type in chat:]** `What is a Transformer model and how does it work?`

**[While waiting, narrate:]**
> "I'm asking about Transformer models. Watch the system retrieve chunks and
> generate a cited answer."

**[After response appears:]**
> "Notice the inline [Source 1], [Source 2] citations. Let me open the sources
> panel to show exactly which chunks were used."

**[Click "Sources" expander]**
> "Here we can see the document title, source file, chunk ID, and a snippet of
> the exact text that was used to construct the answer. This is full
> traceability — no hallucination possible since every claim maps to a source."

---

### Query 2: Follow-up question (tests memory)
**[Type in chat:]** `What are its main limitations?`

**[After response:]**
> "Crucially, I just said 'its' — I didn't specify Transformers. The system
> used conversation memory to understand this is a follow-up and rephrased
> it internally to 'What are the main limitations of the Transformer model?'
> before retrieval. The memory window keeps the last 7 exchanges."

---

### Query 3: Technical deep-dive
**[Type in chat:]** `Explain the difference between LSTM and GRU, and when would you use each?`

**[After response:]**
> "This query is a good test of our Hybrid Search. BM25 catches exact terms
> like 'LSTM' and 'GRU', while semantic search finds conceptually related
> passages about gating mechanisms. The fused result gives us the best of both."

---

### Query 4: Cross-topic / Ethics query
**[Type in chat:]** `What are the main concerns about bias in AI systems?`

**[After response:]**
> "The system pulled from our Ethics of AI and Bias in AI articles.
> Notice it cites multiple sources — this question spans two documents,
> and the retriever correctly identified both as relevant."

---

### Show K slider
**[Adjust slider from 5 to 8]**
> "Users can increase K to retrieve more context for complex questions,
> or decrease it to 3 for faster, more focused answers."

---

## SEGMENT 4 — Code Walkthrough (6:00–7:30)

**[Open VS Code or terminal, show file structure]**

> "Let me quickly show the key files."

**[Open `app/retriever.py`]**
> "This is our advanced feature — the HybridRetriever. Line [X] creates the
> BM25 retriever, line [Y] creates the semantic ChromaDB retriever, and the
> EnsembleRetriever on line [Z] fuses them with RRF."

**[Open `app/document_processor.py`]**
> "DocumentProcessor handles PDF, HTML, and text files. The
> RecursiveCharacterTextSplitter here tries paragraph breaks, then line breaks,
> then sentence endings — always preferring the cleanest natural boundary."

**[Open `app/rag_pipeline.py`]**
> "The RAGPipeline ties it all together. The `_contextualize` method handles
> follow-up questions, and `_build_context` creates the numbered source list
> that maps to the [Source N] citations in the answer."

---

## SEGMENT 5 — Deployment + Closing (7:30–8:30)

**[Show browser with live Streamlit Cloud URL]**

> "The app is deployed at this public URL — anyone can access it right now
> without any local setup. Deployment was a one-click push to Streamlit
> Community Cloud from our GitHub repo."

**[Show GitHub repo briefly]**
> "The GitHub repo has a comprehensive README, the technical write-up,
> and all source code. The pre-built ChromaDB vector store is committed to
> the repo so Streamlit Cloud loads it instantly without re-ingesting."

> "To summarize: we built a production-ready RAG chatbot with persistent
> vector storage, conversation memory, hybrid search, and full source
> citations — deployed publicly and ready for real users. Thank you."

---

## Recording tips
- Keep mouse movements slow and deliberate
- Pause 1–2 seconds after typing before hitting Enter (gives the audience time to read)
- If a response takes >5s, say "while this loads…" and continue talking
- Record in 1920×1080 at minimum
- Trim dead silence in post-production
