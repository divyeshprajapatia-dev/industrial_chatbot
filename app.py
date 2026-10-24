from pathlib import Path
import streamlit as st
from rag_pipeline import build_index, make_asker

st.set_page_config(page_title="Industrial Chatbot", page_icon="🏭")
st.title("🏭 Industrial Document Chatbot")

# ---- session state ----
ss = st.session_state
ss.setdefault("files", {})        # {filename: bytes}
ss.setdefault("seen_ids", set())  # uploader file ids already handled
ss.setdefault("store", None)      # Chroma store
ss.setdefault("ask", None)
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("dirty", False)


def drop_store():
    if ss.store is not None:
        try:
            ss.store.delete_collection()
        except Exception:
            pass
    ss.store, ss.ask = None, None


def clear_all():
    drop_store()
    ss.files = {}
    ss.seen_ids = set()
    ss.messages = []
    ss.dirty = False
    ss.uploader_key += 1  # resets the uploader widget


# ---- sidebar ----
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
    )
    for up in uploads or []:
        if up.file_id not in ss.seen_ids:
            ss.seen_ids.add(up.file_id)
            ss.files[up.name] = up.getvalue()
            ss.dirty = True

    for name in list(ss.files):
        c1, c2 = st.columns([4, 1])
        c1.caption(name)
        if c2.button("✕", key=f"rm_{name}"):
            del ss.files[name]
            ss.dirty = True
            st.rerun()

    if ss.files and st.button("🗑️ Clear everything / start fresh"):
        clear_all()
        st.rerun()

# ---- (re)build index when the file set changed ----
if ss.dirty:
    drop_store()
    if ss.files:
        with st.spinner("Indexing documents..."):
            try:
                ss.store = build_index(ss.files)
                ss.ask = make_asker(ss.store)
            except ValueError as e:
                st.error(str(e))
    ss.dirty = False

# ---- chat ----
if ss.ask is None:
    st.info("Upload one or more PDFs in the sidebar to start.")
    st.stop()

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about your documents"):
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            answer, docs = ss.ask(question)
        st.markdown(answer)
        with st.expander("Sources"):
            for d in docs:
                st.caption(f"{Path(d.metadata.get('source', '?')).name}, "
                           f"p.{d.metadata.get('page', 0) + 1}")
                st.text(d.page_content[:300])
    ss.messages.append({"role": "assistant", "content": answer})