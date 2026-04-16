import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma.base import ChromaVectorStore
import chromadb

# Put your medical KB text files in app/rag/knowledge_base/
# Sources: WHO guidelines (public), MedlinePlus (public domain), home remedy databases
KB_DIR = "app/rag/knowledge_base"
CHROMA_DIR = "./chroma_db"

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embed_model


def build():
    print("Building medical knowledge base...")

    # create KB directory with sample files if empty
    os.makedirs(KB_DIR, exist_ok=True)
    sample_file = os.path.join(KB_DIR, "general_health.txt")

    if not os.listdir(KB_DIR):
        print("  No documents found — creating sample knowledge base file...")
        with open(sample_file, "w") as f:
            f.write("""Iron deficiency anaemia is common in vegetarians and women of childbearing age.
Iron-rich plant foods include lentils, tofu, spinach, pumpkin seeds, and fortified cereals.
Vitamin C enhances iron absorption when consumed together. Vitamin D deficiency is common in people with limited sun exposure.

Diabetes management involves monitoring blood glucose, reducing refined carbohydrates,
increasing fibre intake, and regular physical activity.

Hypertension can be managed through reduced sodium intake, the DASH diet,
regular aerobic exercise, and stress reduction techniques.

Mild headaches are commonly caused by dehydration, tension, or lack of sleep.
Drinking water, resting in a quiet dark room, and gentle neck stretches may help.

Vitamin D deficiency is common in people with limited sun exposure.
Food sources include fatty fish, egg yolks, and fortified dairy products.
""")
        print(f"  Sample file created at {sample_file}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("medical_kb")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(KB_DIR).load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    print(f"  Indexed {len(documents)} document(s) into ChromaDB at {CHROMA_DIR}")

if __name__ == "__main__":
    build()
