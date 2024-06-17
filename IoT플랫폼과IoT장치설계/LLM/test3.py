#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import CommaSeparatedListOutputParser


# model
#llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatOllama(model="test_gguf:latest")

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

#prompt = ChatPromptTemplate.from_template("{topic}에 대하여 설명해줘")
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser

print(chain.invoke({"subject": "popular Korean cusine"}))