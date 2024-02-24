import streamlit
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from langchain.document_loaders.image import UnstructuredImageLoader
from streamlit_modal import Modal

audioExts = ["wav", "wave", "flac", "mp3", "ogg"]
imgExts = ["png", "jpg", "jpeg"]


class Program:
    def __init__(self, uploadedFile, question, file_extension, st: streamlit):
        self.st = st
        with get_openai_callback() as cb:
            if file_extension == "pdf":
                self.UsePDF(question, uploadedFile)  # If the file is a PDF file
            elif file_extension in imgExts:
                self.UseIMG(question, uploadedFile)  # If the file is a Image file
            elif file_extension == "csv":
                self.UseCSV(question, uploadedFile)  # If the file is a CSV file
            else:
                st.markdown("You haven't uploaded a valid file.")
            st.markdown(f"Total Spent Tokens : {cb.total_tokens}")
            st.markdown(f"Total Cost : ${cb.total_cost}")

    def FaissAndQuery(self,text, query):  
        text_splitter = CharacterTextSplitter(  # Text splitting
            separator="\n", chunk_size=300, chunk_overlap=200, length_function=len
        )
        knowledge_base = FAISS.from_texts(
            text_splitter.split_text(text), OpenAIEmbeddings()
        )  # FAISS
        docs = knowledge_base.max_marginal_relevance_search(
            query
        )  # Processing the query received as a parameter
        chain = load_qa_chain(OpenAI(verbose=False), chain_type="stuff")
        response = chain.run(
            input_documents=docs, question=query
        )  # Uploading the file and executing the query
        return response  # Obtaining the response
    
    def UsePDF(self,question, uploadedFile):
        loader = PyPDFLoader(uploadedFile)  
        pages = loader.load_and_split() 
        vectorstore = FAISS.from_documents(
            pages, OpenAIEmbeddings()
        ) 
        chain = RetrievalQA.from_chain_type(
            ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        result = chain({"query": question}) 
        print(result)
        self.st.markdown(result["result"]) 
        self.st.markdown(
            f"Expression Occurring in the File : \n\n {result['source_documents'][0].dict()['page_content']}"
        )  

    def UseIMG(self,question, uploadedFile):
        loader = UnstructuredImageLoader(file_path=uploadedFile) 
        data = loader.load()[0].dict()["page_content"] 
        self.st.markdown(
            f"Answer to the Question : \n\n{self.FaissAndQuery(data, question)}"
        ) 
        self.st.markdown(f"The Text in the Image : \n\n{data}")  

    def UseCSV(self,question, uploadedFile):
        loader = CSVLoader(file_path=uploadedFile, encoding="utf-8")  
        data = loader.load()
        vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())  
        chain = RetrievalQA.from_chain_type(
            ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        result = chain({"query": question}) 
        self.st.markdown(result["result"]) 
        modal = Modal("Expression Occurring in the File", key="csv")
        open_modal = self.st.button(f"File Name : {os.path.basename(uploadedFile)}")
        if open_modal:
            modal.open()
        if modal.is_open():
            with modal.container():
                self.st.write(result["source_documents"][0].dict()["page_content"])
