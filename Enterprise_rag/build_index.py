from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from realtime_ingest import load_realtime_data


def build_index():

    # ---- Load PDFs ----
    pdf_loader = DirectoryLoader(
        "data/pdfs",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    pdf_docs = pdf_loader.load()

    print(f"Loaded {len(pdf_docs)} PDF pages")

    # ---- Load real-time API data ----
    realtime_docs = load_realtime_data()

    print(f"Loaded {len(realtime_docs)} realtime records")

    # ---- Merge all documents ----
    all_documents = pdf_docs + realtime_docs

    # ---- Chunking ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_documents)

    print(f"Total chunks: {len(chunks)}")

    # ---- Embeddings + FAISS ----
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vectorstore")

    print("Vector index created with PDF + realtime data.")


if __name__ == "__main__":
    build_index()
