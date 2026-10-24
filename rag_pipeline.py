import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import uuid
from functools import lru_cache

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN  # optional, only speeds up/authorizes downloads

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = "./data"
PERSIST_DIR = "./vectorstore/chroma_db"
GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert industrial technical assistant. "
    "Answer ONLY from the provided context. "
    "Cite sources as [filename, p.X] after each claim. "
    "If the answer is not in the context, reply exactly: "
    "'I cannot find this information in the technical documentation.' "
    "Never guess values like torque, voltage, or safety limits."
)

@lru_cache(maxsize=1)
def load_local_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.cuda.is_available():
        name = GPU_MODEL
        print(f"CUDA found, loading {name} in 4-bit")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map="auto"
        )
    else:
        name = CPU_MODEL
        print(f"No CUDA, loading {name} on CPU (slow)")
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)

    tok = AutoTokenizer.from_pretrained(name)
    return tok, model


def local_generate(tok, model, messages, max_new_tokens=400):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index(files: dict):
    """files = {filename: pdf_bytes}. Returns an in-memory Chroma store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for name, data in files.items():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            docs = PyPDFLoader(path).load()
        finally:
            os.remove(path)
        for d in docs:
            d.metadata["source"] = name  # real filename for citations
        chunks.extend(splitter.split_documents(docs))

    if not chunks:
        raise ValueError("No text could be extracted (scanned PDF?)")
    return Chroma.from_documents(
        chunks, get_embeddings(), collection_name=f"docs_{uuid.uuid4().hex}"
    )


def make_asker(vectorstore, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    tok, model = load_local_llm()

    def ask(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[{Path(d.metadata.get('source', '?')).name}, "
            f"p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return local_generate(tok, model, messages), docs

    return ask