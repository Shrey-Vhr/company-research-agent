from research_agent.config import COLLECTION_NAME
from research_agent.rag.store import get_store


# In Chroma, a lower score means a closer match, because it’s a distance.
def search(
    query: str,
    company: str | None = None,
    fy: str | None = None,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    store = get_store(collection_name)
    conditions = []
    if company:
        conditions.append({"company": company})
    if fy:
        conditions.append({"fy": fy})

    if len(conditions) == 0:
        where = None  # no filter: search all reports
    elif len(conditions) == 1:
        where = conditions[0]  # one field: use it directly
    else:
        where = {"$and": conditions}  # two fields: Chroma needs $and
    results = store.similarity_search_with_score(query, k=k, filter=where)
    output = []
    for doc, score in results:
        output.append(
            {
                "text": doc.page_content,
                "company": doc.metadata["company"],
                "fy": doc.metadata["fy"],
                "pdf_page": doc.metadata["pdf_page"],
                "score": round(score, 3),
            }
        )
    return output


if __name__ == "__main__":
    for r in search("number of employees", company="TCS"):
        print(
            r["score"], r["company"], r["pdf_page"], r["text"][:100].replace("\n", " ")
        )
    print("---")
    for r in search("number of employees", company="INFY", fy="FY2025-26"):
        print(r["score"], r["company"], r["pdf_page"])
    print("---")
    for r in search("number of employees"):
        print(r["score"], r["company"], r["pdf_page"])
