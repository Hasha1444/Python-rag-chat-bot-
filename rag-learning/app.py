
from pathlib import Path

import streamlit as st
import faiss
import ollama
from sentence_transformers import SentenceTransformer

from first_rag import TOP_K, SIMILARITY_THRESHOLD, load_pdf

DOCUMENTS_DIR = Path("documents")

st.set_page_config(
    page_title="Research Paper RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper Chatbot")
st.caption("Chat with PDFs or compare research papers with your resume.")


@st.cache_resource
def build_rag():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DOCUMENTS_DIR.resolve()}"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = {}

    for pdf_path in pdf_files:
        chunks = load_pdf(pdf_path)

        if not chunks:
            continue

        embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        documents[pdf_path.name] = {
            "chunks": chunks,
            "index": index,
        }

    if not documents:
        raise ValueError("No readable text found in the PDFs.")

    return model, documents


try:
    with st.spinner("Loading PDFs and building search indexes..."):
        model, documents = build_rag()
except Exception as exc:
    st.error(f"Could not load documents: {exc}")
    st.stop()


pdf_names = list(documents.keys())

resume_files = [
    name for name in pdf_names
    if "resume" in name.lower() or "cv" in name.lower()
]

research_files = [
    name for name in pdf_names
    if name not in resume_files
]

with st.sidebar:
    st.subheader("Indexed documents")

    for name, data in documents.items():
        st.write(f"📄 {name}")
        st.caption(f"{len(data['chunks'])} text chunks")

    st.divider()

    mode = st.radio(
        "Chat mode",
        ["Ask a PDF", "Compare with my resume"],
    )

    selected_pdf = None

    if mode == "Ask a PDF":
        selected_pdf = st.selectbox(
            "Choose a document",
            pdf_names,
        )

    if mode == "Compare with my resume":
        if not resume_files:
            st.warning(
                "No filename containing 'resume' or 'cv' was found. "
                "Rename your resume PDF accordingly and refresh."
            )

        if not research_files:
            st.warning("Add at least one research paper PDF.")

        if resume_files:
            st.caption(f"Resume: {resume_files[0]}")

        if research_files:
            selected_research = st.selectbox(
                "Research paper to compare",
                research_files,
            )
        else:
            selected_research = None

    if st.button("Refresh PDF index"):
        build_rag.clear()
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def retrieve(question, filename):
    data = documents[filename]
    chunks = data["chunks"]
    index = data["index"]

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    ).astype("float32")

    scores, results = index.search(
        query_vector,
        min(TOP_K, len(chunks)),
    )

    found = []

    for chunk_id, score in zip(results[0], scores[0]):
        if chunk_id >= 0 and float(score) >= SIMILARITY_THRESHOLD:
            found.append((chunks[chunk_id], float(score)))

    return found


def generate_answer(question, source_files, comparison=False):
    evidence = []

    for filename in source_files:
        for chunk, score in retrieve(question, filename):
            evidence.append((chunk, score))

    if not evidence:
        return (
            "I couldn't find sufficiently relevant information "
            "in the selected documents. Try rephrasing the question.",
            [],
        )

    context = "\n\n".join(
        f"[Source: {chunk['source']}, page {chunk['page']}]\n"
        f"{chunk['text']}"
        for chunk, score in evidence
    )

    if comparison:
        instructions = (
            "Compare the supplied resume and research paper using "
            "only the supplied excerpts. Organize the answer into "
            "Direct matches, Transferable skills, and Missing or "
            "unverified skills when evidence supports those categories. "
            "Explain which evidence comes from which document. Do not "
            "claim that a skill or experience is present unless the "
            "resume excerpt explicitly supports it. A shared general "
            "skill does not prove direct subject-matter experience. "
            "If there is no demonstrated direct match, say so. Cite "
            "source filenames and page numbers for factual claims."
        )
    else:
        instructions = (
            "Answer using only facts explicitly stated in the supplied "
            "document excerpts. Do not invent facts. If the excerpts "
            "do not provide enough information, say so. Cite source "
            "filenames and page numbers for factual claims."
        )

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    return response["message"]["content"], evidence


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.write(
                        f"- {source['filename']}, "
                        f"page {source['page']} "
                        f"(similarity: {source['score']:.3f})"
                    )

if mode == "Compare with my resume":
    placeholder = (
        "Compare this paper with my resume. Identify direct matches, "
        "transferable skills, and missing skills..."
    )
else:
    placeholder = "Ask a question about the selected PDF..."

question = st.chat_input(placeholder)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if mode == "Compare with my resume":
                if not resume_files or not selected_research:
                    st.error(
                        "Please ensure a resume and a research paper "
                        "are both indexed."
                    )
                    st.stop()

                source_files = [
                    resume_files[0],
                    selected_research,
                ]
            else:
                source_files = [selected_pdf]

            with st.spinner("Retrieving evidence and generating answer..."):
                answer, evidence = generate_answer(
                    question,
                    source_files,
                    comparison=(mode == "Compare with my resume"),
                )

            st.markdown(answer)

            sources = [
                {
                    "filename": chunk["source"],
                    "page": chunk["page"],
                    "score": score,
                }
                for chunk, score in evidence
            ]

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        st.write(
                            f"- {source['filename']}, "
                            f"page {source['page']} "
                            f"(similarity: {source['score']:.3f})"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
