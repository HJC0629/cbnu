#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# model
#llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatOllama(model="test_gguf:latest")

prompt = ChatPromptTemplate.from_template("{topic}에 대하여 설명해줘")

chain = prompt | llm | StrOutputParser()

# chain 실행
print(chain.invoke({"topic" : "deep learning"}))