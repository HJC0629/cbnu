#pip install BeautifulSoup4

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

url = 'https://namu.wiki/w/LLaMA'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
#print(docs[0].page_content[5000:6000])