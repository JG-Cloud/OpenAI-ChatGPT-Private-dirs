# # python standard mods
import os
import sys
import platform

# suppress python warning messages
import warnings
warnings.filterwarnings("ignore")


### dependencies
from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

## Import prompt templates (to start the query question if the user enters a single word or phrase - (i.e. how can I x (get resource from xyz))
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

## Import data loaders for a specific doc or Directory
from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredPDFLoader

## Import embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

## Import Index with vector DBS/stores
from langchain.vectorstores import Chroma

## Import character/text splitter
from langchain.text_splitter import CharacterTextSplitter

# Web framework for hosting app
import streamlit as st


# # vars
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if 'mac' in platform.platform():
    test_dir = '/Users/jatingandhi/Downloads/test_dir'
else:
    test_dir = '/mnt/c/Users/jgandhi/OneDrive - Ensono/Downloads/test_dir'

# embeddings type for indexing
embeddings = OpenAIEmbeddings()

# chromaDB path
chromadb_path = "./chromadb"



# import data_loader.py / vectostore
# db = data_loader.data_loader()

db = Chroma(
    persist_directory=f"{chromadb_path}", 
    embedding_function=embeddings
)


##  expose this index in a retriever interface.
retriever = db.as_retriever()


# qa chain - (langchain)
def get_qa_chain():
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever
    )
    
    return qa


####### NOTES / TBC

# For chat history
# from langchain.chains import ConversationalRetrievalChain


################### OLD/DEBUG

## Moved to data_loader.py for testing
# if __name__ == "__main__":
#     def langchain_config():
#         # Doc loader - load on app startup only
#         dir_loader = DirectoryLoader(f"{test_dir}", show_progress=True, use_multithreading=True)
#         docs = dir_loader.load()

#         # # text_splitter
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         texts = text_splitter.split_documents(docs)

#         # # create/load vectorstore to use as index
#         # Guidance: https://python.langchain.com/docs/modules/data_connection/retrievers/# vectorstore db persistence TBC
#         # Enable to save to disk & reuse the model (for repeated queries on the same data)
#         PERSIST = True

#         if PERSIST and os.path.exists(f"{chromadb_path}"):
#             db = Chroma(
#                 persist_directory=f"{chromadb_path}", 
#                 embedding_function=embeddings
#             )
#         else:
#             # load split text into vectorstore, as set embedding
#             db = Chroma.from_documents(
#                 texts, 
#                 embedding=embeddings, 
#                 persist_directory='chromadb'
#             )
#             db.persist()
#
#         # #  expose this index in a retriever interface.
#         retriever = db.as_retriever()
#        
#         return retriever
#
#    langchain_config()

# ### streamlit webapp config/framework
# st.title("🦜️🔗 Jatin's data sourcing AI app")
# query_prompt = st.text_input('What are you looking for?', placeholder='Search inside my docs...')

# if query_prompt:        
#     # # qa chain - (langchain)
#     qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

#     st.write(qa.run(query_prompt)) 


# # query / response
# query = "What is my exam score"
# qa.run(query)


# # debugging docs/num of words
# print(docs)
# print(len(docs))

# for doc in docs:
#     num_words = len(doc.page_content.split(' '))

# print(num_words)


