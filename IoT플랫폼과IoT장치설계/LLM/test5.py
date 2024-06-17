#pip install langchain_openai

#pip install chromadb



from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


url = 'https://ko.wikipedia.org/wiki/LLaMA'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
#print(docs[0].page_content[5000:6000])


#text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
text_splitter = CharacterTextSplitter(
    separator = ' ',
    chunk_size=100,
    chunk_overlap=50,
    length_function = len,
)
splits = text_splitter.split_documents(docs)

os.environ["OPENAI_API_KEY"] = "___________________본인 OPENAI API Key __________________"
api_key = os.getenv("OPENAI_API_KEY")

vectorstore = Chroma.from_documents(documents = splits,
                                    embedding=OpenAIEmbeddings(openai_api_key=api_key))

docs = vectorstore.similarity_search("유사도")
print(len(docs))
print(docs[0].page_content)











#pdf_filepath = '000660_SK_2023.pdf'
#loader = PyPDFLoader(pdf_filepath)
#pages = loader.load()

#print(len(pages))
#print(pages[10])