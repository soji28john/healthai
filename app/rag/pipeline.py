import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

_index = None

def get_index():
    global _index
    if _index is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("medical_kb")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return _index

async def retrieve_context(query: str, top_k: int = 3) -> str:
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return "\n\n".join(n.get_content() for n in nodes)