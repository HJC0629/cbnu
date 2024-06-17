from langchain_community.llms import Ollama

# model
llm = Ollama(model="test_gguf:latest")

# chain 실행
response = llm.invoke("지구의 자전 주기는?")

print(response)