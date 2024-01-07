from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
import os

from apikey import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

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
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
        container.markdown(response.content)