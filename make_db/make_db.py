from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from utility.util import extract_title 

#Document Object
from langchain.schema.document import Document

import torch
import os
import json
import re

# GPU or CPU Device Setting
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

list_of_db_work = os.listdir("DB_work_cln")

os.path.dirname(os.getcwd())

whole_list_of_documents = []

####### readme.md 제외하고 db Directory에서 file들 가져와서 UnstructuredMarkdownLoader에 넣고 Parsing해서 list에 저장(whole_list_of_documents)
###### 이것도 함수로 wrap
for directory in list_of_db_work :
    if directory == "readme.md" :
        continue
    if directory == ".DS_Store" :
        continue
    
    absolute_path= os.path.abspath(os.path.join('DB_work_cln/',directory))
    directory_loader = DirectoryLoader(path=absolute_path, glob="*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = directory_loader.load()

    whole_list_of_documents.extend(documents)


###### 이것도 함수로 wrap
with open("./utility/metadata.json", "r", encoding="utf-8") as file :
    meta_json = json.load(file)



###### 이것도 함수로 wrap
#### whole_list_of_documents <-> metadata.json 비교할 떄 필요없는 문자들 전부 날리고 key값으로 넣어서 metadata tag 가져와서 return
documents_edit_metadata = []

for document in whole_list_of_documents :
    document_title = extract_title(document=document)
    document_title_strip = re.sub(r'[^가-힣a-zA-Z0-9\·\•(),~\[\]ⅠⅡ·-]', '', document_title)
    
    meta_result = meta_json.get(document_title_strip)

    if meta_result is not None :
        document.metadata = {"tag" : meta_result, "title":document_title}

    documents_edit_metadata.append(document)


#Embedding
############# 이것도 함수로 따로 wrap하는게 나을듯?
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding = SentenceTransformerEmbeddings(model_name='./finetune-embedding/da_finetune', model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings':True})
# embedding = SentenceTransformerEmbeddings(model_name='BM-K/KoSimCSE-roberta', model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings':True})


#Chroma Settings
#이건 따로 script 빼서 하는게 관리에 좋을듯? 아님말고
#persistentClient 관리하려면 지우는거떄문에 둘이 묶는게 나을수도
import chromadb
from langchain.vectorstores import Chroma

#VectorDB settings
chroma_client = chromadb.PersistentClient()

collection_name = "vector_db"
collection = chroma_client.get_or_create_collection(collection_name)

vectorstore = Chroma(
    client= chroma_client,
    collection_name= collection_name,
    embedding_function= embedding,
    persist_directory="./chroma"
)

print("There are", vectorstore._collection.count(), "in the collection.")

#### Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

from langchain.retrievers import ParentDocumentRetriever

child_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore= vectorstore,
    docstore=store,
    child_splitter= child_splitter,
    parent_splitter= parent_splitter,
)


retriever.add_documents(documents=documents_edit_metadata)

vectorstore.persist()

print("There are", vectorstore._collection.count(), "in the collection.")

##### 일단 test
# result_1 = retriever.get_relevant_documents("근로장려금")

# print(result_1[0].metadata)
# result_metadata_sample = result_1[0].metadata

# print(result_metadata_sample['tag'].split(','))

# result_1


### Collection 싹 날려버리는 명령어. 일단 Chroma랑 통합해서 놔둘지 생각해야 함.
# vectorstore.delete_collection()
# vectorstore.persist()

# chroma_client.delete_collection(collection_name)