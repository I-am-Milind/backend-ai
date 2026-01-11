from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection("semantic_memory")

def store_fact(text: str):
    embedding = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def recall_fact(query: str) -> str | None:
    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )

    if results["documents"]:
        return results["documents"][0][0]

    return None
