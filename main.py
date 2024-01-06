from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from embedding import EmbeddingLoader


## get Database from chroma
embedding_model_path = "model/ko_sroberta_multitask_seed_777_lr_1e-5"
model_name = embedding_model_path.split("/")[-1]

STE = EmbeddingLoader.SentenceTransformerEmbedding
sentenceloader = STE(model_name=embedding_model_path, encode_kwargs={'normalize_embeddings':True})
embedding_model = sentenceloader.load()

print(embedding_model)

print("Get collection from chroma . . . ")
db = Chroma(persist_directory="chroma", collection_name=model_name, embedding_function=embedding_model)
print(f"collection name : {db._collection.name}")
print(f"collection size : {db._collection.count()}")
print("Loading collection Complete . . . ")



print(db.similarity_search("안녕하세요. 20대 청년 취업에 대해서 알려주세요"))
