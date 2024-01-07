from langchain.callbacks.base import BaseCallbackHandler

## streamlit에서 response 있을 경우 stream 하는 Callback class
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)