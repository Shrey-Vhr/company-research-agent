from langchain_text_splitters import RecursiveCharacterTextSplitter
from research_agent.rag.loader import load_reports


def chunk_documents(docs, chunk_size=3200, chunk_overlap=400):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=splitter.split_documents(docs)
    # The problem. Later, when you save chunks into Chroma, each chunk needs a unique name (an ID), so that running the build twice replaces chunks instead of duplicating them. The natural name is company + page, for example tcs-p142.

    # But some pages split into 2 or 3 chunks. Then all of them would be called tcs-p142, and they’d overwrite each other. So each chunk also needs its own number within its page
    previous_page = None
    counter = 0
    for c in chunks:
        this_page = (c.metadata["company"], c.metadata["pdf_page"])
        if this_page == previous_page:
            counter += 1        # same page as the last chunk -> next number
        else:
            counter = 1         # new page -> start again at 1
        c.metadata["chunk"] = counter
        previous_page = this_page
    
    return chunks


if __name__ == "__main__":
    docs = load_reports()
    chunks = chunk_documents(docs)
    for c in chunks[100:106]:
        print(c.metadata["company"], c.metadata["pdf_page"], c.metadata["chunk"])
    print("Total chunks:", len(chunks))

    from collections import Counter
    print("Per company:", Counter(c.metadata["company"] for c in chunks))
    print("Most chunks from one page:", max(c.metadata["chunk"] for c in chunks))

    sample = chunks[200]
    print(sample.metadata)
    print(sample.page_content[:300])
