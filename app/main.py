"""
Ask Vibs — NLP Professor Chatbot
Powered by RAG (Retrieval-Augmented Generation)

Features:
  - Multiple named conversation logs
  - Quiz mode: 10 questions on a chosen topic, graded with retake support
  - Quiz history log in sidebar
  - Fun fact on page load
  - Quick-start prompt chips
  - Export conversation as a text file

Environment variable required: GOOGLE_API_KEY
  Local: add to .env
"""

import os
import sys
import re
import logging
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask Vibs",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-badge {
    display: inline-block; background: #6A1B9A; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75em; margin-right: 4px;
}
.doc-type-badge {
    display: inline-block; background: #757575; color: white;
    padding: 2px 8px; border-radius: 12px; font-size: 0.72em;
}
.citation-card {
    background: #f3e5f5; border: 1px solid #ab47bc;
    border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
}
.snippet {
    border-left: 3px solid #ccc; padding-left: 0.6rem;
    color: #555; font-size: 0.88em; font-style: italic; margin-top: 0.3rem;
}
.funfact-box {
    background: #4A148C;
    border-left: 4px solid #CE93D8; border-radius: 8px;
    padding: 0.8rem 1rem; margin-bottom: 1rem; color: white;
}
.conv-active {
    background: #ede7f6; border-radius: 6px;
    padding: 4px 8px; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
FUN_FACT_TOPICS = [
    "transformer architecture", "RLHF", "large language models",
    "RAG systems", "attention mechanisms", "prompt engineering",
    "natural language processing", "word embeddings", "agentic AI",
]
QUICK_PROMPTS = [
    "What is RLHF and how does it work?",
    "How do RAG systems work?",
    "Explain prompt engineering techniques",
]
QUIZ_TOPICS = [
    "NLP fundamentals", "Large language models", "Prompt engineering",
    "RAG systems", "RLHF and alignment", "Agentic AI",
    "Transformers and attention", "Text classification", "Word embeddings",
]
TOP_K = 5


# ── Pipeline loader ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_pipeline():
    from rag_pipeline import RAGPipeline
    from document_processor import DocumentProcessor
    from config import DATA_DIR

    pipeline = RAGPipeline()
    if not pipeline.is_ready():
        data_path = DATA_DIR
        if os.path.isdir(data_path) and any(os.scandir(data_path)):
            processor = DocumentProcessor()
            chunks = processor.process_directory(data_path)
            if chunks:
                pipeline.vsm.add_documents(chunks)
    return pipeline


# ── Quiz parser ───────────────────────────────────────────────────────────

def parse_quiz(text: str) -> list:
    """Parse LLM-generated quiz text into a list of question dicts."""
    questions = []
    blocks = re.split(r'\n(?=\d+[\.\)])', text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if not lines:
            continue
        q_text = re.sub(r'^\d+[\.\)]\s*', '', lines[0])
        options = {}
        correct = None
        for line in lines[1:]:
            m = re.match(r'^([A-D])[.)]\s*(.*)', line)
            if m:
                options[m.group(1)] = m.group(2).strip()
            m2 = re.match(r'^(?:Correct|Answer)[:\s]+([A-D])', line, re.IGNORECASE)
            if m2:
                correct = m2.group(1).upper()
        if q_text and len(options) >= 3 and correct:
            questions.append({
                "question": q_text,
                "options": options,
                "correct": correct,
                "user_answer": None,
            })
    return questions[:10]


# ── Citation renderer ─────────────────────────────────────────────────────

def render_citations(sources: list):
    if not sources:
        return
    label = f"📚 **{len(sources)} source{'s' if len(sources) > 1 else ''}** — click to expand"
    with st.expander(label, expanded=False):
        for src in sources:
            i, title = src["index"], src["title"]
            source_file, doc_type = src["source"], src["doc_type"].upper()
            chunk_id, page, url = src["chunk_id"], src.get("page"), src.get("url")
            snippet = src["snippet"]
            st.markdown(
                "<div class='citation-card'>"
                f'<span class="source-badge">[{i}]</span> '
                f"<strong>{title}</strong> "
                f'<span class="doc-type-badge">{doc_type}</span>',
                unsafe_allow_html=True,
            )
            meta = [f"📄 `{source_file}`", f"Chunk: `{chunk_id}`"]
            if page:
                meta.append(f"Page {page}")
            if url:
                meta.append(f"[🔗 Link]({url})")
            st.caption(" · ".join(meta))
            st.markdown(f'<div class="snippet">{snippet[:350]}…</div></div>', unsafe_allow_html=True)
            st.write("")


# ── Session state init ────────────────────────────────────────────────────

def init_state():
    if "conversations" not in st.session_state:
        st.session_state.conversations = {
            "conv_1": {"name": "Conversation 1", "messages": [], "chat_history": []}
        }
    if "active_conv" not in st.session_state:
        st.session_state.active_conv = "conv_1"
    if "conv_counter" not in st.session_state:
        st.session_state.conv_counter = 1
    if "fun_fact" not in st.session_state:
        st.session_state.fun_fact = None
    if "quick_prompt" not in st.session_state:
        st.session_state.quick_prompt = None
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"
    if "quiz" not in st.session_state:
        st.session_state.quiz = None  # dict when active
    if "quiz_log" not in st.session_state:
        st.session_state.quiz_log = []  # list of {topic, score, total, date}


# ── Quiz page ─────────────────────────────────────────────────────────────

def render_quiz_page(pipeline):
    st.title("🧠 Quiz Mode")

    if st.button("← Back to Chat"):
        st.session_state.mode = "chat"
        st.session_state.quiz = None
        st.rerun()

    st.divider()

    quiz = st.session_state.quiz

    # Step 1: Topic selection
    if quiz is None:
        st.subheader("Choose a topic for your 10-question quiz")
        topic_choice = st.selectbox("Pick a topic:", QUIZ_TOPICS)
        custom = st.text_input("Or type your own topic:")
        topic = custom.strip() if custom.strip() else topic_choice

        if st.button("Generate Quiz", type="primary"):
            with st.spinner(f"Generating 10 questions about {topic}…"):
                result = pipeline.query(
                    question=(
                        f"Generate exactly 10 multiple choice questions about '{topic}' "
                        "based on the course materials. Format each question exactly like this:\n\n"
                        "1. Question text?\n"
                        "A) Option A\n"
                        "B) Option B\n"
                        "C) Option C\n"
                        "D) Option D\n"
                        "Correct: A\n\n"
                        "Output only the questions, no extra commentary."
                    ),
                    chat_history=[],
                    top_k=8,
                )
            questions = parse_quiz(result["answer"])
            if len(questions) < 3:
                st.error("Couldn't generate enough questions. Try a different topic.")
            else:
                st.session_state.quiz = {
                    "topic": topic,
                    "questions": questions,
                    "submitted": False,
                    "score": None,
                }
                st.rerun()
        return

    # Step 2: Answer questions
    if not quiz["submitted"]:
        st.subheader(f"Quiz: {quiz['topic']}")
        st.caption(f"{len(quiz['questions'])} questions — select your answers, then submit.")
        st.divider()

        answers = {}
        for i, q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            opts = [f"{k}) {v}" for k, v in sorted(q["options"].items())]
            choice = st.radio(
                f"q{i+1}",
                opts,
                index=None,
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
            )
            answers[i] = choice
            st.write("")

        if st.button("Submit Quiz", type="primary"):
            score = 0
            for i, q in enumerate(quiz["questions"]):
                selected = answers.get(i)
                if selected and selected.startswith(q["correct"]):
                    score += 1
            quiz["submitted"] = True
            quiz["score"] = score
            quiz["answers"] = answers
            st.session_state.quiz_log.append({
                "topic": quiz["topic"],
                "score": score,
                "total": len(quiz["questions"]),
                "date": datetime.now().strftime("%b %d, %H:%M"),
            })
            st.rerun()
        return

    # Step 3: Results
    score = quiz["score"]
    total = len(quiz["questions"])
    pct = int(score / total * 100)

    if pct >= 80:
        grade, emoji = "A", "🏆"
    elif pct >= 70:
        grade, emoji = "B", "🎉"
    elif pct >= 60:
        grade, emoji = "C", "👍"
    else:
        grade, emoji = "F", "📚"

    st.subheader(f"{emoji} Quiz Results — {quiz['topic']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{score}/{total}")
    col2.metric("Percentage", f"{pct}%")
    col3.metric("Grade", grade)

    st.divider()

    # Show answers
    for i, q in enumerate(quiz["questions"]):
        selected_raw = quiz["answers"].get(i)
        selected_letter = selected_raw[0] if selected_raw else None
        correct = q["correct"]
        is_correct = selected_letter == correct

        icon = "✅" if is_correct else "❌"
        st.markdown(f"**{icon} Q{i+1}. {q['question']}**")
        for letter, text in sorted(q["options"].items()):
            if letter == correct:
                st.markdown(f"&nbsp;&nbsp;&nbsp;**{letter}) {text} ← Correct**")
            elif letter == selected_letter:
                st.markdown(f"&nbsp;&nbsp;&nbsp;~~{letter}) {text}~~")
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;{letter}) {text}")
        st.write("")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Retake Quiz (new questions)", use_container_width=True):
            st.session_state.quiz = None
            st.rerun()
    with col_b:
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state.mode = "chat"
            st.session_state.quiz = None
            st.rerun()


# ── Chat page ─────────────────────────────────────────────────────────────

def render_chat_page(pipeline):
    st.title("🎓 Ask Vibs")
    st.caption(
        "Hi! I'm Vibs, your NLP professor. Ask me anything from class — "
        "lectures, concepts, assignments, or study help."
    )

    conv = st.session_state.conversations[st.session_state.active_conv]

    # Fun fact (once per session)
    if st.session_state.fun_fact is None:
        topic = random.choice(FUN_FACT_TOPICS)
        with st.spinner("Loading a fun fact…"):
            result = pipeline.query(
                question=f"Share one surprising fun fact about {topic} in 2–3 sentences. Be concise and engaging.",
                chat_history=[],
                top_k=3,
            )
        st.session_state.fun_fact = result["answer"]

    st.markdown(
        f'<div class="funfact-box">💡 <strong>Did you know?</strong><br>{st.session_state.fun_fact}</div>',
        unsafe_allow_html=True,
    )

    # Quick prompts (when chat is empty)
    if not conv["messages"]:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        for idx, pt in enumerate(QUICK_PROMPTS):
            with cols[idx % 3]:
                if st.button(pt, key=f"chip_{idx}", use_container_width=True):
                    st.session_state.quick_prompt = pt
                    st.rerun()

    # Render messages
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                render_citations(msg["sources"])

    # Input
    if st.session_state.quick_prompt:
        prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None
    else:
        prompt = st.chat_input("Ask Vibs anything from class…")

    if prompt:
        conv["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching class materials…"):
                result = pipeline.query(
                    question=prompt,
                    chat_history=conv["chat_history"],
                    top_k=TOP_K,
                )
            st.write(result["answer"])
            if result.get("sources"):
                render_citations(result["sources"])

        conv["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
        conv["chat_history"].append((prompt, result["answer"]))
        from config import MAX_MEMORY_EXCHANGES
        conv["chat_history"] = conv["chat_history"][-MAX_MEMORY_EXCHANGES:]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    pipeline = load_pipeline()
    init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎓 Ask Vibs")
        st.caption("Your AI/NLP Professor · Powered by RAG")
        st.divider()

        # New conversation
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.conv_counter += 1
            cid = f"conv_{st.session_state.conv_counter}"
            st.session_state.conversations[cid] = {
                "name": f"Conversation {st.session_state.conv_counter}",
                "messages": [],
                "chat_history": [],
            }
            st.session_state.active_conv = cid
            st.session_state.mode = "chat"
            st.rerun()

        # Conversation list
        st.markdown("**Chats**")
        for cid, conv in st.session_state.conversations.items():
            is_active = cid == st.session_state.active_conv and st.session_state.mode == "chat"
            label = f"{'▶ ' if is_active else ''}{conv['name']}"
            if st.button(label, key=f"conv_{cid}", use_container_width=True):
                st.session_state.active_conv = cid
                st.session_state.mode = "chat"
                st.rerun()

        st.divider()

        # Quiz section
        st.markdown("**Quiz**")
        if st.button("🧠 Take a Quiz", use_container_width=True):
            st.session_state.mode = "quiz"
            st.session_state.quiz = None
            st.rerun()

        # Quiz log
        if st.session_state.quiz_log:
            st.markdown("**Quiz History**")
            for entry in reversed(st.session_state.quiz_log[-5:]):
                grade_pct = int(entry["score"] / entry["total"] * 100)
                st.caption(
                    f"📝 {entry['topic']}\n"
                    f"{entry['score']}/{entry['total']} ({grade_pct}%) · {entry['date']}"
                )

        st.divider()

        # Export current conversation
        conv = st.session_state.conversations.get(st.session_state.active_conv, {})
        if conv.get("messages"):
            transcript = "\n\n".join(
                f"{'You' if m['role'] == 'user' else 'Vibs'}: {m['content']}"
                for m in conv["messages"]
            )
            st.download_button(
                label="💾 Export chat",
                data=transcript,
                file_name=f"{conv['name'].replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()
        st.caption("Built with LangChain · ChromaDB\nHybrid Search (BM25 + Semantic)")

    # ── Main content ──────────────────────────────────────────────────────
    if not pipeline.is_ready():
        st.warning("Knowledge base is empty. Run:\n\n```bash\npython scripts/ingest.py\n```")
        st.stop()

    if st.session_state.mode == "quiz":
        render_quiz_page(pipeline)
    else:
        render_chat_page(pipeline)


if __name__ == "__main__":
    main()
