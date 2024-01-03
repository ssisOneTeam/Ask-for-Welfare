"""
    property : sentence_transformers (pip install transformers)

    embedding 쉽게 받아오려고 쓰는거.
    모델 이름은 기본으로 저장된 BAAI/bge-base-en-v1.5 말고 다른거 쓰고싶으면 HuggingFace에서 알아서 찾아서 download에 인수로 넣을 것.

"""

import os


class EmbeddingDownLoader:
    # model 명 기본으로 지정되어 있음. 다른거 받으려면 변경해야 함.
    def __init__(self, model:str='model_name_here', path=None,) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("package not found. try install sentence_transformers."
                              "try following command : pip install sentence-transformers")
        
        self.model = model
        self.path = path
        return
        
    def download(self, model_name: str = None) -> None:
        from sentence_transformers import SentenceTransformer

        # 경로 지정 안해놨으면 현재 스크립트 실행되는 곳에 모델명으로 파일 생성되게 만듦.
        if self.path is None:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model)
        
        # 다운로드 함수
        downloader = SentenceTransformer(model_name_or_path=self.model)
        os.makedirs(self.path)
        downloader.save(self.path)

        print(f'model {self.model} downloaded at path {self.path}.')
        return


if __name__ == "__main__":
    # print("your default model is 'BAAI/bge-base-en-v1.5'. ")
    # model_name_here = input("If you want download any other model, please write here: ")
    loader = EmbeddingDownLoader(model="BM-K/KoSimCSE-roberta-multitask")
    loader.download()
