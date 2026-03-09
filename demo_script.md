# Demo Script — Ask Vibs RAG Chatbot
## BANA275 · RAG-Powered NLP Professor Chatbot
**Target length:** 7–9 minutes  |  **Recommended tool:** Loom or OBS

---

## Pre-recording checklist
- [ ] App is running (local or live Streamlit URL)
- [ ] Browser zoom at 110–125% so text is readable
- [ ] Dark mode off (better contrast on recording)
- [ ] Silence notifications

---

## SEGMENT 1 — Introduction (0:00–0:40)

**[On-screen: App homepage]**

> "Hi, I'm [Name]. In this demo I'll walk you through Ask Vibs — our RAG-powered
> NLP professor chatbot for BANA275. The system retrieves relevant passages from a
> knowledge base of 65 documents, including lecture slides and Wikipedia AI/ML articles,
> and generates grounded answers — all without hallucinating or making things up."

> "The stack is: LangChain for orchestration, ChromaDB as our persistent vector
> database, Google Gemini 2.5 Flash as the LLM, and Streamlit for the UI. The app
> is deployed publicly on Streamlit Community Cloud."

---

## SEGMENT 2 — Architecture Overview (0:40–2:00)

**[On-screen: Show the README or architecture diagram]**

> "Let me quickly explain how it works. When a student asks a question, the system
> goes through four stages."

> "First, if this is a follow-up question, we use the conversation history to
> rephrase it into a standalone question — so the retriever gets something unambiguous."

> "Second, we run Hybrid Search — our advanced feature. We combine BM25 keyword
> retrieval at 30% weight with ChromaDB semantic search at 70% weight, and merge
> the rankings using Reciprocal Rank Fusion to get the top-5 most relevant chunks."

> "Third, those chunks become the context that Gemini uses to generate an answer
> as Vibs, the NLP professor."

> "Finally, the Streamlit UI renders the answer with expandable source citations."

---

## SEGMENT 3 — Live Demo (2:00–6:00)

### Query 1: Simple factual question
**[Type in chat:]** `What is a Transformer model and how does it work?`

**[While waiting, narrate:]**
> "I'm asking about Transformer models. Watch the system retrieve chunks and
> generate an answer."

**[After response appears:]**
> "Let me open the sources panel to show exactly which chunks were used."

**[Click "Sources" expander]**
> "Here we can see the document title, source file, chunk ID, and a snippet of
> the exact text used. This is full traceability — every claim maps to a source."

---

### Query 2: Follow-up question (tests memory)
**[Type in chat:]** `What are its main limitations?`

**[After response:]**
> "Crucially, I said 'its' without specifying Transformers. The system used
> conversation memory to understand this is a follow-up and rephrased it internally
> to 'What are the main limitations of the Transformer model?' before retrieval.
> The memory window keeps the last 7 exchanges."

---

### Query 3: Technical deep-dive
**[Type in chat:]** `Explain the difference between LSTM and GRU`

**[After response:]**
> "This is a good test of our Hybrid Search. BM25 catches exact terms like 'LSTM'
> and 'GRU', while semantic search finds conceptually related passages about gating
> mechanisms. The fused result gives us the best of both."

---

### Show Quiz Mode
**[Click "🧠 Take a Quiz" in sidebar]**

> "Students can also take a quiz. I'll select 'Transformers and attention' and
> generate 10 multiple choice questions from the course materials."

**[Generate quiz, answer a few, submit]**
> "After submitting, the system grades the answers, assigns a letter grade, and
> shows which answers were correct. The quiz score is saved in the sidebar history,
> and students can retake for a fresh set of questions."

---

### Show Multiple Conversations
**[Click "➕ New Conversation" in sidebar]**
> "Students can create separate conversation logs — for example one for studying
> LLMs and another for RAG systems — each with its own memory context."

---

## SEGMENT 4 — Code Walkthrough (6:00–7:30)

**[Open VS Code or terminal, show file structure]**

**[Open `app/retriever.py`]**
> "This is our advanced feature — the HybridRetriever. It creates a BM25 retriever,
> a ChromaDB semantic retriever, and fuses them with EnsembleRetriever using RRF."

**[Open `app/document_processor.py`]**
> "DocumentProcessor handles PDF, HTML, and text files using
> RecursiveCharacterTextSplitter — it tries paragraph breaks first, then line breaks,
> then sentence endings."

**[Open `app/rag_pipeline.py`]**
> "The RAGPipeline ties it all together. The `_contextualize` method handles
> follow-up questions, and `_build_context` creates the numbered source list
> for the citation cards."

---

## SEGMENT 5 — Deployment + Closing (7:30–8:30)

**[Show live Streamlit Cloud URL]**

> "The app is deployed at this public URL — anyone can access it right now.
> Deployment was a one-click push to Streamlit Community Cloud from our GitHub repo."

**[Show GitHub repo briefly]**
> "The pre-built ChromaDB vector store is committed to the repo so Streamlit Cloud
> loads it instantly without re-ingesting."

> "To summarize: we built Ask Vibs — a production-ready RAG chatbot with persistent
> vector storage, conversation memory, hybrid search, quiz mode, and full source
> citations — deployed publicly and ready for students. Thank you."

---

## Recording tips
- Keep mouse movements slow and deliberate
- Pause 1–2 seconds after typing before hitting Enter
- If a response takes >5s, say "while this loads…" and continue talking
- Record in 1920×1080 at minimum
- Trim dead silence in post-production
