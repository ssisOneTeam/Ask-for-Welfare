# Import Module
import streamlit as st
import chromadb
import torch
import re
import os
import io
import utility

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from streamlit_chat import message # Chatbot UI
from audio_recorder_streamlit import audio_recorder # 음성녹음
from pydub import AudioSegment # 녹음 파일 저장
from openai import OpenAI # STT


# ====================================================================================================================
# Global Config
try:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # 환경변수에 OPENAI_API_KEY를 설정합니다.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print('APY_KEY 없음..')

score_threshold = 0.02
search_k = 5
llm_model = "gpt-4-1106-preview" # gpt-3.5-turbo / gpt-4-1106-preview
user_img = "https://freesvg.org/img/abstract-user-flat-4.png"
bot_img = "https://github.com/ash-hun/WelSSISKo/raw/main/assets/logo02.png"

st.set_page_config(
    page_title="물어보장",
    page_icon="👋",
)

# GPU or CPU Device Setting
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

st.title("물어보장")




# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


score_threshold = 0.2
search_k = 5
llm_model = "gpt-4-1106-preview" # gpt-3.5-turbo / gpt-4-1106-preview
user_img = "https://freesvg.org/img/abstract-user-flat-4.png"
bot_img = "https://github.com/ash-hun/WelSSISKo/raw/main/assets/logo02.png"


# GPU or CPU Device Setting
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 임베딩 모델 불러오기
try:
    model_dir = './model/ko_sroberta_multitask_seed_777_lr_1e-5' # 필요시 경로변경
    embedding = SentenceTransformerEmbeddings(model_name=model_dir, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings':True})
except:
    print('plz check Embedding Model')
    pass


# 크로마 DB 불러오기
try:
    chroma_client = chromadb.PersistentClient(path="./chroma")

    collection_name = "ko_sroberta_multitask_seed_777_lr_1e-5"

    vectorstore = Chroma(
        client= chroma_client,
        collection_name= collection_name,
        embedding_function= embedding,
        persist_directory="./chroma"
    )
except:
    print('plz check ChromaDB')
    pass

try:
    # 임계점 기반 : 적절한 threshold 값 선정이 필수임.
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': search_k ,'score_threshold': score_threshold})

    ## llm 프롬프팅
    # 검색된 문장 내에서만 대답하도록 하고 내용을 바꾸지 못하게 프롬프트 작성

    system_template="""Use the following pieces of context to answer the users question shortly.
    Given the following summaries of a long document and a question, create a final answer with references ("source_documents"), use "source_documents" in capital letters regardless of the number of sources.
    But Don't say word of source_documents.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    ----------------
    {context}

    You MUST answer in Korean"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

except:
    print('백단에서 에러')
    pass
    ## 이부분 필요한 부분마다 except 해서 예외처리 해주면 좋을 것 같음..





# ====================================================================================================================
# Define Function
def stt():
    ## (녹음) 마이크 버튼 두번 누르면 사용자 음성신호 mp3 형식으로 저장 - ./output.mp3
    audio_bytes = audio_recorder(text="")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    ## 녹음 완료되면 mp3 형식으로 저장하고 SST 함수 이용하여 text 변환
    if audio_bytes is not None:
        ## mp3 형식으로 저장
        audio_segmant = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Export the audio file
        audio_segmant.export('./audio/output.mp3', format='mp3')

        # mp3 파일 불러와서 STT 적용
        client = OpenAI()
        sst_text = utility.STT("./audio/output.mp3", client)

        clean_text = sst_text.replace("\n", "")

        # 유저 input창에 텍스트 심어줌.
        js = f"""
        <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
        var chatInput = parent.document.querySelector('textarea[data-testid="stChatInput"]');
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeInputValueSetter.call(chatInput, "{clean_text}");
        var event = new Event('input', {{ bubbles: true}});
        chatInput.dispatchEvent(event);
        }}
        insertText({len(st.session_state['generated'])});
        </script>
        """
        st.components.v1.html(js)

        audio_bytes = None 

def tts(): # TTS 기능
    try:
        # st.sidebar.write(final_response)
        utility.chat_output_value(final_response)
        audio_file = open('./output.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.sidebar.audio(audio_bytes, format='audio/mp3')
    except:
        st.sidebar.write('최근 답변된 내용이 없습니다. 질문을 먼저 해주세요.')
        audio_file = open('./data/audio/output_error.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.sidebar.audio(audio_bytes, format='audio/mp3')

def llm_chatbot(question):
    """ llm_chatbot

    사용자가 쿼리(question)를 입력하면 LangChain을 통해 embedding 모델을 거쳐 Vector DB에 들어간 문서를 Retriever하여
    관련성이 깊은 문서를 찾는다. 이때, 찾아낸 결과(문서 개수)에 따라 서로 다른 처리를 이행한다.

    Args:
        question (str): _description_

    Returns:
        _type_: _description_
    """
    query = question
    result = chain(query)

    # 문서 검색결과에 따라 다른 처리
    if len(result['source_documents']) > 0: # 문서 하나라도 검색된 경우
        # title 반환을 위한 코드 
        lst = []
        for i in range(len(result['source_documents'])):
            try:
                # 시도: metadata['title']에 접근
                title_link = "[" + result['source_documents'][i].metadata['title'] + "](" + result['source_documents'][i].metadata['url'] + ")"
                #title_link = "www.naver.com"
                lst.append(title_link)
            except KeyError:
                # 예외 처리: 'title' 키가 없을 경우
                continue
        return(result['result'], lst)
    else: # 검색된 문서가 없는경우
        return ((f"'{result['query']}' 에 대한 내용은 문서에 없습니다."), '')

def set_list(docs):
    """ set_list
    문서 내용이 중복될 경우 제거한다.

    Args:
        docs (_type_) : None check duplicate data

    Returns:
        unique_list (list) : Delete duplicate data
    """

    unique_list = []
    seen = set()

    for item in docs:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

def modeloutput(prompt):
    """ modeloutput

    실제 출력될 챗봇내용을 정제한다.

    Args:
        prompt (str): Output prompt

    Returns:
        str : Transform output prompt
    """
    prompt, docs = llm_chatbot(prompt)
    prompt = re.sub(r'\[source_documents\]|\(source_documents\)|source_documents', '', prompt)
    # 리스트를 문자열로 변환 후 연결
    if len(docs) == 0:
        return (f"{prompt}", f"비슷한 의미의 단어를 사용하여 재검색 해보시거나, 'aaa@aaa.com'을 통해 문의 바랍니다.")
    else: 
        joined_docs = ', '.join(map(str, set_list(docs)))
        return (f"{prompt}", f"이와 관련된 복지제도는 **{joined_docs}** 등이 있습니다.")

# Main Contents
# $ streamlit run prototype.py
if __name__ == "__main__":
    # =================================================================
    ## Setting Styling
    with open('./css.css', 'r', encoding='utf-8') as file:
        css = file.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    # =================================================================
    # Layout Grid
    col1, col2= st.sidebar.columns(2)
    
    utility.add_logo()
    st.markdown(f"Version 0.2 / LLM : {llm_model}")
    # =================================================================
    with st.sidebar.container():
        with col1:
            # st.sidebar.button("🎤", on_click=stt) #🎤
            ## (녹음) 마이크 버튼 두번 누르면 사용자 음성신호 mp3 형식으로 저장 - ./output.mp3
            audio_bytes = audio_recorder(text="")

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

            ## 녹음 완료되면 mp3 형식으로 저장하고 SST 함수 이용하여 text 변환
            if audio_bytes is not None:
                ## mp3 형식으로 저장
                audio_segmant = AudioSegment.from_file(io.BytesIO(audio_bytes))
                # Export the audio file
                audio_segmant.export('./audio/output.mp3', format='mp3')

                # mp3 파일 불러와서 STT 적용
                client = OpenAI()
                sst_text = utility.STT("./audio/output.mp3", client)

                clean_text = sst_text.replace("\n", "")

                # 유저 input창에 텍스트 심어줌.
                js = f"""
                <script>
                function insertText(dummy_var_to_force_repeat_execution) {{
                var chatInput = parent.document.querySelector('textarea[data-testid="stChatInput"]');
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, "{clean_text}");
                var event = new Event('input', {{ bubbles: true}});
                chatInput.dispatchEvent(event);
                }}
                insertText({len(st.session_state['generated'])});
                </script>
                """
                st.components.v1.html(js)

                audio_bytes = None 
        with col2:
            st.sidebar.button("🎧", on_click=tts) # 🔈


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

            



if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="저는 '나에게 힘이되는 복지서비스 2023' 책자를 기반으로 복지정책에 대해 알려드리는 프로그램입니다. 궁금하신 복지 정책에 대해 질문해주세요.")]

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
        stream_handler = StreamHandler(container)
        llm = ChatOpenAI(model_name=llm_model, temperature=0, streaming=True, callbacks=[stream_handler])

        # 
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever = retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        result = chain(prompt)
        st.session_state.messages.append(ChatMessage(role="assistant", content=result['result']))

        lst = []
        for i in range(len(result['source_documents'])):
            try:
                # 시도: metadata['title']에 접근
                title_link = "[" + result['source_documents'][i].metadata['title'] + "](" + result['source_documents'][i].metadata['url'] + ")"
                lst.append(title_link)
            except KeyError:
                # 예외 처리: 'title' 키가 없을 경우
                lst.append('관련된 문서가 없습니다.')
        joined_docs = ', '.join(map(str, lst))
        docs = f"**{joined_docs}** 등이 있습니다."

        final_response = f"""
        {result['result']}  
        {"**이와 관련된 복지제도는**"}
        {docs} 
        """
        container.markdown(final_response)