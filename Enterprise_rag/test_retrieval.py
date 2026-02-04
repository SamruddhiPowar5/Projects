import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_cited_context(docs):
    blocks = []

    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "unknown")

        block = f"""
[Chunk {i}]
Source: {source}, page {page}
Content:
{d.page_content}
"""
        blocks.append(block)

    return "\n\n".join(blocks)


def check_citations(answer_text, num_chunks):
    matches = re.findall(r"\[Chunk\s*(\d+)\]", answer_text)

    if not matches:
        return False, []

    cited = [int(m) for m in matches]
    valid = [c for c in cited if 1 <= c <= num_chunks]

    return len(valid) > 0, valid


def main():

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    print("\nEnterprise RAG with grounded citations. Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Bye 👋")
            break

        search_query = rewrite_query(llm, query)
        print(f"\n🔎 Rewritten query: {search_query}\n")

        docs = retriever.get_relevant_documents(search_query)


        if not docs:
            print("\nI don't know based on the provided documents.\n")
            continue

        context = build_cited_context(docs)

        prompt = f"""
You are an assistant for answering questions over company documents.

Use ONLY the information in the context below.

When you use a fact, you MUST cite the chunk number in square brackets,
for example: [Chunk 1], [Chunk 2].

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer with citations:
"""

        response = llm.invoke(prompt)

        print("\nAnswer:\n")
        print(response.content)

        is_grounded, cited_chunks = check_citations(
            response.content,
            len(docs)
        )

        if not is_grounded:
            print("\n⚠️ Warning: Answer contains NO valid citations. It may be ungrounded.")
        else:
            print(f"\n✅ Citations detected for chunks: {sorted(set(cited_chunks))}")

        print("\nRetrieved chunks:\n")
        for i, d in enumerate(docs, start=1):
            print(
                f"[Chunk {i}] "
                f"{d.metadata.get('source_file')} "
                f"(page {d.metadata.get('page')})"
            )

        print("\n" + "-" * 70 + "\n")

def rewrite_query(llm, question):
    prompt = f"""
Rewrite the following question into a short search query
suitable for retrieving relevant sections from a scientific paper.

Question:
{question}

Search query:
"""
    return llm.invoke(prompt).content.strip()


if __name__ == "__main__":
    main()
