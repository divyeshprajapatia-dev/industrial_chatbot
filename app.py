from pathlib import Path
import streamlit as st
from rag_pipeline import ingest_pdfs, load_vector_store, build_rag_chain, PERSIST_DIR

st.set_page_config(page_title="Industrial Chatbot", page_icon="🏭")
st.title("🏭 Industrial Document Chatbot")


@st.cache_resource
def get_chain():
    vs = load_vector_store() if Path(PERSIST_DIR).exists() else ingest_pdfs()
    return build_rag_chain(vs)


with st.sidebar:
    st.write("Put PDFs in the `data/` folder, then rebuild.")
    if st.button("Rebuild index"):
        ingest_pdfs()
        st.cache_resource.clear()
        st.success("Index rebuilt")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            answer, docs = get_chain()(question)
        st.markdown(answer)
        with st.expander("Sources"):
            for d in docs:
                name = Path(d.metadata.get("source", "?")).name
                st.caption(f"{name}, p.{d.metadata.get('page', 0) + 1}")
                st.text(d.page_content[:300])
    st.session_state.messages.append({"role": "assistant", "content": answer})