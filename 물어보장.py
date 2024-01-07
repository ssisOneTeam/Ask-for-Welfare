###### TODO
# Document URL, TITLE 답변에서 따와가지고 붙이기
# 답변 형식 Markdown처럼 정제된 형식으로 나오게 하기
# TTS STT 할 거면 붙이기(누가 개좋아해서 해야할듯 ㅡㅡ 굳이?)

import os
from apikey import OPENAI_API_KEY

import streamlit as st
from langchain.schema import ChatMessage
from langchain.chat_models.openai import ChatOpenAI
from callbacks import StreamlitCallbackHandler

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

#Embedding
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from embedding import EmbeddingLoader

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

##CONFIG
score_threshold = 0.2
search_k = 5

##TODO
#이거 하나로 묶기 혹은 함수화해서 runanable하게 만들면 될듯

## get DB from chroma
embedding_model_path = "model/ko_sroberta_multitask_seed_777_lr_1e-5"
model_name = embedding_model_path.split("/")[-1]

STE = EmbeddingLoader.SentenceTransformerEmbedding
sentenceloader = STE(model_name=embedding_model_path, encode_kwargs={'normalize_embeddings':True})
embedding_model = sentenceloader.load()

print("Get collection from chroma . . . ")
db = Chroma(persist_directory="chroma", collection_name=model_name, embedding_function=embedding_model)
print(f"collection name : {db._collection.name}")
print(f"collection size : {db._collection.count()}")
print("Loading collection Complete . . . ")

## set DB as retriever
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k':search_k, 'score_threshold':score_threshold})

## prompt
template = """
You are Kindest Wellfare Expert. Use the following pieces of context to answer the users question shortly.
# context : {context}

Given the following summaries of a long document and a question, create a final answer with references ("source_documents"), use "source_documents" in capital letters regardless of the number of sources.
But Don't say word of source_documents.
If you don't know the answer, just say that "죄송합니다. 답변할 수 있는 정보가 없습니다.", don't try to make up an answer. YOU MUST ANSWER IN Korean. YOU MUST RETURN DETAILED DESCRIPTION BASED ON THE question.

# question : {question}
# your answer based on context : 
"""
prompt_template = ChatPromptTemplate.from_template(template)

## Output_parser
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context":retriever, "question":RunnablePassthrough()}
)

## Streamlit
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 저는 복지 상담 서비스 물어보장입니다. 무엇을 도와드릴까요? 복지와 관련한 궁금한 정보 아무거나 물어보장 ~")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not os.environ.get("OPENAI_API_KEY"):
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        container = st.empty()
        stream_handler = StreamlitCallbackHandler(container)

        llm = ChatOpenAI(model="gpt-4-1106-preview", streaming=True, callbacks=[stream_handler])
        # LCEL
        chain = setup_and_retrieval | prompt_template | llm | output_parser
        response = chain.invoke(prompt)
        
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        container.markdown(response)