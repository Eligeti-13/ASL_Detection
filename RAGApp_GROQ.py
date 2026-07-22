import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import os

st.set_page_config(page_title="Free RAG with LLaMA", layout="wide")

st.write("<h1 style='text-align: center;'>DocuRAG Intelligence</h1>", unsafe_allow_html=True)

st.sidebar.title("DocuFlow")
page = st.sidebar.radio(
    "Choose an option:",
    ["Dashboard", "Upload PDF", "Chat with PDF"])


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

@st.cache_resource
def load_accuracy_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_model():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

embeddings = load_embedding_model()
accuracy_model = load_accuracy_model()
llm = load_model()

# Dashboard
if page == "Dashboard":
    st.markdown("""DocuRAG Intelligence uses a Retrieval-Augmented Generation engine that
    fuses document search with AI reasoning. It extracts the right knowledge from your
    PDFs, injects it into your query, and produces answers that are evidence-driven,
    transparent, and reliable not guesses.""")

    st.image("RAG.png",width="stretch")

# Upload PDF
elif page == "Upload PDF":
    st.title("Upload PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        st.session_state["retriever"] = retriever
        st.success("PDF indexed successfully.")

# Chat with PDF
elif page == "Chat with PDF":
    st.title("Chat with Your PDF")

    if "retriever" not in st.session_state:
        st.warning("Please upload and index a PDF first.")
        st.stop()

    retriever = st.session_state["retriever"]

    query = st.text_input("Ask a question about the document")

    # Retrieve top-k chunks
    if query:
        docs = retriever.get_relevant_documents(query)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        print("---RETRIEVED CHUNKS---")
        for i, doc in enumerate(docs):
              print(f"Chunk {i+1}:")
              print(doc.page_content)
              print("---")

        #prompt for LLaMA
        prompt = f"""
            You are a document-based question answering system.

            Rules:
            - Answer ONLY from the context below.
            - Do NOT use outside knowledge.
            - If the answer is not present in the context, say:
              "Not found in the document."
            - When listing points, number them sequentially as 1, 2, 3, 4...
              with no skipped numbers, regardless of how they appear in the context.

            Context:
            {context_text}

            Question:
            {query}

            Answer:
            """

        response = llm.chat.completions.create(
              model="llama-3.1-8b-instant",
              messages=[
                  {"role": "user", "content": prompt}
              ],
              max_tokens=256,
              temperature=0
          )
        answer = response.choices[0].message.content.strip()

        # Accuracy calculation
        answer_emb = accuracy_model.encode(answer, convert_to_tensor=True)
        context_emb = accuracy_model.encode(context_text, convert_to_tensor=True)
        accuracy = float(util.cos_sim(answer_emb, context_emb))

        # generating answer and accuracy
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Accuracy")
        st.write(f"{accuracy:.2f}")
