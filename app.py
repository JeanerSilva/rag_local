import os
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.embeddings import HuggingFaceBgeEmbeddings
import glob

# Configuração
MODEL_PATH = "./models/mistral.gguf"
DOCS_PATH = "./docs"
VECTORDB_PATH = "./vectordb"

# Cache do modelo
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        temperature=0.7,
        verbose=False,
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # leve e compatível

@st.cache_resource
def create_vectorstore():
    docs = []
    files = glob.glob(f"{DOCS_PATH}/*")

    for file in files:
        ext = os.path.splitext(file)[1]
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt":
            loader = TextLoader(file)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        raise Exception("Nenhum documento encontrado em ./docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORDB_PATH)

    return db

@st.cache_resource
def load_vectorstore():
    if os.path.exists(VECTORDB_PATH):
        embeddings = load_embeddings()
        return FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vectorstore()

# Carregamento
llm = load_llm()
db = load_vectorstore()

# Criar chain de RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", k=4),
    return_source_documents=False
)

# Interface
st.set_page_config(page_title="LLM RAG com Pastas", page_icon="🧠")
st.title("📂 RAG com Arquivos Locais + LLM Offline")

# Histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Entrada do usuário
with st.form("pergunta-form", clear_on_submit=True):
    user_input = st.text_input("Pergunte com base nos documentos:")
    submitted = st.form_submit_button("Enviar")

if submitted and user_input:
    resposta = qa_chain.run(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", resposta))

# Exibição estilo chat
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# Botão para limpar histórico
if st.button("🗑️ Limpar conversa"):
    st.session_state.chat_history = []

# Botão para download da última resposta
if st.session_state.chat_history:
    for role, msg in reversed(st.session_state.chat_history):
        if role == "bot":
            st.download_button("📥 Baixar última resposta", msg, file_name="resposta.txt")
            break
