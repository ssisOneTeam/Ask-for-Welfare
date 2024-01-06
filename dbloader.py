"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성. 
    
    need url_table, metadata.json """

## modules
import os
import uuid
import re
import pandas as pd
import numpy as np
import json

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import TextSplitter, SentenceTransformersTokenTextSplitter
from langchain.schema.document import Document 

import time
import functools

#embeddingloading
from embedding import EmbeddingLoader
from langchain.vectorstores.chroma import Chroma

## wrapper for check time
def timecheck(func):
    """메소드 실행 시간을 측정하고 출력하는 데코레이터"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"run method {func.__name__} takes: {end_time - start_time:.4f}sec.")
        return result

    return wrapper

class BaseDBLoader:
    """markdownDB folder에서 불러온 다음에 폴더별로 내부에 있는 내용 Load해서 Split하고 저장함"""

    def __init__(self, path_db:str, path_metadata:str, path_url_table:str, text_splitter:TextSplitter=None, loader_cls=UnstructuredMarkdownLoader):
        # textsplitter config
        self.text_splitter = text_splitter
        # loaderclass config
        self.loader_cls = loader_cls
        # md 파일 담고 있는 전체 디렉터리 경로
        self.path_db = path_db
        # metadata path
        self.path_metadata = path_metadata
        # url table paths
        self.path_url_table = path_url_table
        # storage
        self.storage = []

        #log
        print("initializing Class Start.")

        return

    @timecheck
    def load(self, is_split=True, is_regex=True, show_progress=True, use_multithreading=True) -> list[Document]: ### mul 수정
        """ Get Directory Folder and documents -> parse, edit metadata -> langchain Document list. 
        
            args :
                is_split: whether split or not(text_splitter)
                is_regex: apply regex to edit document form. 
                show_progress: show progress -> from LangChain.
                use_multithreading: use multithread(cpu) -> from LangChain. """
        # document pre-processing
        for db_folder in os.listdir(self.path_db):
            db_folder_abs = os.path.join(self.path_db, db_folder)
            if not os.path.isdir(db_folder_abs): ## 절대경로가 폴더일 경우에만 작동, 다른 경우 pass (구조 체크를 위해 넣었음)
                continue

            directory_loader = DirectoryLoader(path=db_folder_abs, loader_cls=self.loader_cls, show_progress=show_progress, use_multithreading=use_multithreading)
            doc_list = directory_loader.load()

            if is_regex:
                doc_list = self._preprocess_document_list(doc_list)            
            if is_split:
                doc_list = self.text_splitter.split_documents(doc_list)
            self.storage.extend(doc_list)
        self.storage = self._process_document_metadata(self.storage)

        return self.storage

    def _preprocess_document_list(self, doc_list:list[Document]) -> list[Document]:
        """regex splitter (Document)"""
        regex = '([^가-힣0-9a-zA-Z.,·•%↓()\s\\\])'
        result = []
        for document in doc_list:
            sub_str = re.sub(pattern=regex, repl="", string=document.page_content)
            document.page_content = sub_str
            result.append(document)
        return result

    ##### metadata edit methods
    def _read_tag_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    
    def _read_url_table(self, file_path):
        return pd.read_csv(file_path, header=0, index_col=0)

    def _replace_metadata(self, metafilename:str, replacer:dict={"_":" ", "•":"·"})->str:
        for key, value in replacer.items():
            metafilename = metafilename.replace(key, value)
        return metafilename

    def _strip_replace_text(self, s:str)->str:
        regex = '([^가-힣0-9a-zA-Z.,·•%↓()\s\\\])'
        # regex = "([^가-힣0-9a-zA-Z])"
        s = re.sub(pattern=regex, repl="", string=s)
        s = self._replace_metadata(metafilename=s, replacer={" ":"", '•':'·', 'Ⅰ':'', 'Ⅱ':'', "_":""})
        return s
    
    def _get_category_from_source(self, source:str)->str:
        """get category from Document object metadata['source'] and parse directory(for category use.)"""
        parsed_source = source.split("\\")
        dir_source = parsed_source[-2]
        return self._replace_metadata(dir_source)
    
    def _process_document_metadata(self, documents:list[Document])->list:
        """
        wrapper method.
        get metadata edit internal methods and integrate all.
        """
        metadata_json = self._read_tag_file(self.path_metadata)
        url_table = self._read_url_table(self.path_url_table)

        for document in documents:
            #### get source from Document metadata
            meta_source = document.metadata["source"]
            meta_source_parsed = meta_source.split("\\")

            #### category
            document.metadata["category"] = self._get_category_from_source(meta_source)

            #### title
            meta_source_parsed_file_name = meta_source_parsed[-1]
            meta_source_parsed_get = meta_source_parsed_file_name[3:-3]
            result = self._replace_metadata(metafilename=meta_source_parsed_get)
            document.metadata["title"] = result
            
            #### tag
            title = document.page_content.split("\n")[0]
            title_parsed = self._strip_replace_text(title)
            document.metadata["tag"] = metadata_json[title_parsed]

            #### url
            result = url_table.loc[url_table["source"] == meta_source_parsed_file_name]["url"].values[0]

            if result is np.nan :
                document.metadata["url"] = ""
            else :
                document.metadata["url"] = result

        return documents
    
    ## get corpus(legacy)
    def get_corpus(self) -> dict:
        """self.storage가 존재한다면 dict로 결과 return함."""
        if not self.storage :
            raise ValueError("loader에 storage가 생성되지 않았습니다. load 함수를 실행하거나 storage를 확인하고 다시 실행하세요.")     
        return {str(uuid.uuid4()): doc.page_content for doc in self.storage}

##################################################################################################################################################

class TokenDBLoader(BaseDBLoader):
    """
        fix loading sequence

        before: document -> split -> extract title
        after: document -> extract title -> split 
    """
    
    @timecheck
    def load(self, is_regex=False, show_progress=True, use_multithreading=True) -> list[Document]:
        """ Get Directory Folder and documents -> parse, edit metadata -> langchain Document list. 
        
            args :
                is_regex: apply regex to edit document form. 
                show_progress: show progress -> from LangChain.
                use_multithreading: use multithread(cpu) -> from LangChain. """
        # document pre-processing
        for db_folder in os.listdir(self.path_db):
            db_folder_abs = os.path.join(self.path_db, db_folder)
            if not os.path.isdir(db_folder_abs): ## 절대경로가 폴더일 경우에만 작동, 다른 경우 pass (구조 체크를 위해 넣었음)
                continue
            
            ## load data initialize
            directory_loader = DirectoryLoader(path=db_folder_abs, loader_cls=self.loader_cls, show_progress=show_progress, use_multithreading=use_multithreading)
            doc_list = directory_loader.load()

            ## 여기에서 제목추출하고 보내는게 맞을듯... (제목 추출 순서 변경 완료..) 231215
            doc_list = self._process_document_metadata(doc_list)

            if is_regex:
                doc_list = self._preprocess_document_list(doc_list)

            self.storage.extend(doc_list)

        return self.storage
    
    def _strip_replace_text(self, s: str)->str:
        regex = '([^가-힣0-9a-zA-Z])'
        s = re.sub(pattern=regex, repl="", string=s)
        return s
    
##################################################################################################################################################

#240106 edit
class HuggingFaceModelTextSplitter:
    """ Get HuggingFace Embedding(by local) -> check & get max token sequence length -> split document by max_token_sequence_length """
    def __init__(self, path:str):
        self.path = path
        # get max_sequence_length from model path
        sentence_bert_config = "sentence_bert_config.json"
        config_path = os.path.join(self.path, sentence_bert_config)

        with open(config_path) as file :
            bert_config = json.load(file)
            self.max_seq_length = bert_config["max_seq_length"]

        print(f"max sequence from current model({self.path}) is {self.max_seq_length}.")

    @timecheck
    def split_documents(self, documents:list[Document])->list[Document]:
        """ Get langchain document object list and split by TokenTextSplitter 
            
            Args:
            documents : Langchain document object list
        
        """
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, tokens_per_chunk=self.max_seq_length, model_name=self.path)


        splitted_docs = text_splitter.split_documents(documents)
        print(f"Document splitted with SentenceTransformerTokenizer -> length <{len(splitted_docs)}>")
    
        return splitted_docs
        
## Making DB example
if __name__ == "__main__":
    #config
    embedding_model_path = "model/ko_sroberta_multitask_seed_777_lr_1e-5"
    #Chroma object 생성.
    chroma = Chroma()
    ### from_document method 이용해서 저장(document, embedding, persist_directory, collection_name)

    #run
    db = TokenDBLoader(path_db="data", path_metadata="metadata.json", path_url_table="url_table.csv").load()
    splitter = HuggingFaceModelTextSplitter(path=embedding_model_path)
    db_split_by_token = splitter.split_documents(db)

    # file_path = "documents.txt"

    # # 파일에 쓰기
    # with open(file_path, "w", encoding="utf-8") as file:
    #     for document in db_split_by_token:
    #         file.write(document.page_content + "\n")

    # print(f"{len(db_split_by_token)}개의 문서가 '{file_path}'에 저장되었습니다.")

    #chroma setup
    STE = EmbeddingLoader.SentenceTransformerEmbedding
    sentenceloader = STE(model_name=embedding_model_path, multi_process=True, encode_kwargs={'normalize_embeddings':True})
    embedding_model = sentenceloader.load()

    # set model name(cause collection name length limit)
    model_name = embedding_model_path.split("/")[-1]

    # save document with chunk - embedding calculate and save it to persist directory
    collection = chroma.from_documents(documents=db_split_by_token, embedding=embedding_model, collection_name=model_name, collection_metadata={"hnsw:space":"cosine"}, persist_directory="chroma")
    collection.persist()
    print("collection generate with ChromaDB complete.")
    print(collection._collection)

    ## chroma 저장까지는 잘 됨, loading 부분(main.py에서 문제가 있는거임)

# ## TODO
# ## db생성 빼놓기
# ## 밑에 있는 chroma 관련해서 client 참조하는 방식 사용하기 (메모리 절약, 시간 절약)

# ## 위에거 하면 생성에서 stream 해가지고 streamlit에서 flush하게 출력될 수 있도록 하기(시간 절약)