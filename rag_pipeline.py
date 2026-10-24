import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

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


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_pdfs(data_dir=DATA_DIR):
    """Rebuild the vector store from every PDF in data_dir."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for pdf in Path(data_dir).glob("*.pdf"):
        chunks.extend(splitter.split_documents(PyPDFLoader(str(pdf)).load()))
    if not chunks:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)  # avoid duplicate chunks
    return Chroma.from_documents(
        chunks, get_embeddings(), persist_directory=PERSIST_DIR
    )


def load_vector_store():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())


def build_rag_chain(vectorstore, k=4):
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


if __name__ == "__main__":
    vs = ingest_pdfs()
    ask = build_rag_chain(vs)
    answer, _ = ask("What is this document about?")
    print(answer)