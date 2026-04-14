from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Put your medical KB text files in app/rag/knowledge_base/
# Sources: WHO guidelines (public), MedlinePlus (public domain), home remedy databases

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("medical_kb")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("app/rag/knowledge_base/").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print(f"Indexed {len(documents)} documents into ChromaDB")