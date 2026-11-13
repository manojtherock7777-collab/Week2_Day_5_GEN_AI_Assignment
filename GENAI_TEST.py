import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Setup
st.set_page_config(page_title="Local DocChat AI", page_icon="📄", layout="wide")
st.title("📄 Local Document Q&A (100% Offline RAG)")
st.caption("Powered by LangChain, Ollama & Streamlit — Zero API Keys")

# 2. Sidebar: Upload a TXT or PDF file
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# 3. Helper function to index the uploaded file (cached so it doesn't re-embed on every message)
@st.cache_resource(show_spinner="Reading and indexing your document...")
def build_retriever(file_bytes: bytes, file_name: str):
    suffix = os.path.splitext(file_name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3}), len(chunks)


# 4. Initialize Chat Messages in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Main App Logic
if uploaded_file is not None:
    retriever, num_chunks = build_retriever(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed '{uploaded_file.name}' into {num_chunks} chunks!")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the document context below.
    If the answer is not in the context, say "I couldn't find that in the uploaded document."

    Context:
    {context}

    Question: {question}
    """)

    llm = ChatOllama(model="gemma3:1b", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new question
    if user_question := st.chat_input("Ask a question about your uploaded file..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a `.txt` (like your `company_policy.txt`) or `.pdf` file in the left sidebar to begin!")