from langchain.schema.document import Document
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

""" dependency : pypdf, langchain, unstructured. """

class Extract_Text_From_PDF():
    def __init__(self, file_path):
        """ Get File From Local and Wrap with pypdf.PdfReader. """

        try:
            import pypdf
        except ImportError:      
            print(""" Error Occured. Try pip install pypdf. """)
            exit(0)

        self.file = pypdf.PdfReader(file_path)

    def extract_text(self, start_page=1, end_page=None, path="./result.txt"):
        """ 객체에 추출한 텍스트 담아서 path에 지정된 파일에 저장. start page부터 end page까지. """

        text = ""
        for i in range(start_page-1, end_page):
            text += f'\n{"=" * 60}\nStart of Page {i+1}.\n{"="*60}\n\n'
            text += self.file.pages[i].extract_text()
            text += f'\n\n{"="*60}\nEnd of Page {i+1}.\n{"="*60}\n'

        with open(path, "w", encoding='utf-8') as file:
            file.write(text)

def extract_title(document:Document)->str:
    """ get DocumentObject from langchain.schema.document.Document and split to get title."""
    sentences = document.page_content.split('\n')
    title = sentences[0]
    
    return title


