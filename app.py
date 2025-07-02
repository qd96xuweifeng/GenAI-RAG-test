import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# --- Load Embeddings and Vectorstore ---
st.title("📘 Company Policy Assistant (RAG)")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# --- Initialize Retrieval-based QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- Streamlit Interface ---
question = st.text_input("Ask a question about company policy:")

if question:
    with st.spinner("Searching and generating answer..."):
        result = qa_chain(question)
        st.markdown("### 🧠 Answer:")
        st.write(result["result"])

        st.markdown("---")
        st.markdown("### 📄 Source Snippets:")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:**")
            st.write(doc.page_content)