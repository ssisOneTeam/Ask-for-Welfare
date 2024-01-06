from langchain.vectorstores.chroma import Chroma
from embedding import EmbeddingLoader

embedding_model_path = "model/ko_sroberta_multitask_seed_777_lr_1e-5"
model_name = embedding_model_path.split("/")[-1]

STE = EmbeddingLoader.SentenceTransformerEmbedding
sentenceloader = STE(model_name=embedding_model_path, multi_process=True, encode_kwargs={'normalize_embeddings':True})
embedding_model = sentenceloader.load()

db = Chroma(persist_directory="./chroma", collection_name=model_name, embedding_function=embedding_model)
print(db._collection.name)
print(db._collection.count())