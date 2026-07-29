import pandas as pd
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import gradio as gr

DATA_DIR = "course_materials"  


def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def load_csv(path):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(f"{col}: {row[col]}" for col in df.columns))
    return "\n".join(rows)

FILES = [
    ("course1_python_book.pdf", "Intro to Python Programming", load_pdf),
    ("course1_syllabus.txt",    "Intro to Python Programming", load_txt),
    ("course2_faq.csv",         "Data Structures",             load_csv),
    ("course3_ml_notes.docx",   "Machine Learning Basics",     load_docx),
]

def clean_text(text):
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def build_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for filename, course, loader in FILES:
        path = f"{DATA_DIR}/{filename}"
        raw_text = loader(path)
        cleaned = clean_text(raw_text)
        for chunk in splitter.split_text(cleaned):
            chunks.append({"text": chunk, "course": course, "source": filename})
    return chunks

print("Loading and chunking course materials...")
all_chunks = build_chunks()
print(f"Total chunks: {len(all_chunks)}")

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in all_chunks]
metadatas = [{"course": c["course"], "source": c["source"], "text": c["text"]} for c in all_chunks]

print("Embedding chunks...")
embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"Stored {index.ntotal} chunks in FAISS index")

print("Loading LLM...")
llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
)

def retrieve(query, k=3, course_filter=None):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k=k * 3 if course_filter else k)

    results = []
    for idx in indices[0]:
        meta = metadatas[idx]
        if course_filter and meta["course"] != course_filter:
            continue
        results.append(meta)
        if len(results) == k:
            break
    return results

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the answer is not in the context, say "I don't know based on the course materials."

Context:
{context}

Question: {query}

Answer:"""

def answer_question(query, course_filter=None, k=3):
    retrieved = retrieve(query, k=k, course_filter=course_filter)
    prompt = build_prompt(query, retrieved)
    response = llm(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    answer = response[len(prompt):].strip()
    sources = list({c["source"] for c in retrieved})
    return answer, sources

COURSES = ["All Courses", "Intro to Python Programming", "Data Structures", "Machine Learning Basics"]

def gradio_answer(question, course):
    if not question.strip():
        return "", ""
    course_filter = None if course == "All Courses" else course
    answer, sources = answer_question(question, course_filter=course_filter, k=3)
    sources_str = "\n".join(f"- {s}" for s in sources)
    return answer, sources_str

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="0 1px 3px rgba(0,0,0,0.08)",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

custom_css = """
.gradio-container { max-width: 880px !important; margin: auto !important; }
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #6b7280; margin-top: 4px; margin-bottom: 24px; }
#answer_box textarea { font-size: 16px; line-height: 1.55; }
#sources_box textarea { font-size: 14px; color: #374151; }
footer { display: none !important; }
"""

with gr.Blocks(title="Course Q&A") as demo:
    gr.Markdown("# Course Q&A", elem_id="title")
    gr.Markdown("Ask a question about your course materials and get an answer with source attribution.", elem_id="subtitle")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Your question", placeholder="e.g. What is a stack?", lines=2)
        with gr.Column(scale=1):
            course = gr.Dropdown(choices=COURSES, value="All Courses", label="Course")

    ask_btn = gr.Button("Ask", variant="primary")

    answer = gr.Textbox(label="Answer", lines=6, interactive=False, elem_id="answer_box")
    sources = gr.Textbox(label="Sources", lines=2, interactive=False, elem_id="sources_box")

    gr.Examples(
        examples=[
            ["What is a stack?", "Data Structures"],
            ["What is the difference between supervised and unsupervised learning?", "Machine Learning Basics"],
            ["How do you define a function in Python?", "Intro to Python Programming"],
        ],
        inputs=[question, course],
    )

    ask_btn.click(fn=gradio_answer, inputs=[question, course], outputs=[answer, sources])
    question.submit(fn=gradio_answer, inputs=[question, course], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
