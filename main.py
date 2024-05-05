__import__('pysqlite3')
import sys
sys.modeules['sqlite3']=sys.mdules.pop('pysqlite3')

#from dotenv import load_dotenv
#load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

st.title('ChatPDF')
st.write("---")

uploaded_file = st.file_uploader('Choose a file(Only PDF)',type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    pass 
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False
    )

    texts=text_splitter.split_documents(pages)

    embeddings_model=OpenAIEmbeddings()

    db=Chroma.from_documents(texts,embeddings_model)#, persist_directory="./chroma_db")

    st.header("PDF에게 질문해보세요!!")
    questions = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        with st.spinner('돌아가는 중...'):
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain.invoke({'query':questions})
            st.write(result["result"])


#retriever_from_llm = MultiQueryRetriever.from_llm(
#    retriever=db.as_retriever(), llm=llm
#)
#docs = retriever_from_llm.invoke(questions)
