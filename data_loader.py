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

## Import data loaders for a specific doc or Directory
from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredPDFLoader

## Import embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

## Import Index with vector DBS/stores
from langchain.vectorstores import Chroma

## Import character/text splitter
from langchain.text_splitter import CharacterTextSplitter


# Import pickle - to load/dump the ChromaDB/vectorstore
import pickle


## vars
# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

if 'mac' in platform.platform():
    test_dir = '/Users/jatingandhi/Downloads/test_dir'
else:
    test_dir = '/mnt/c/Users/jgandhi/OneDrive - Ensono/Downloads/test_dir'

# embeddings type for indexing
embeddings = OpenAIEmbeddings()

# chromaDB path
chromadb_path = "./chromadb"




# body

def data_loader():
    # Doc loader - load on app startup only
    dir_loader = DirectoryLoader(f"{test_dir}", show_progress=True, use_multithreading=True)
    docs = dir_loader.load()    
    
    # # text_splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)    
    
    # # create/load vectorstore to use as index
    # Guidance: https://python.langchain.com/docs/modules/data_connection/retrievers/# vectorstore db persistence TBC
    if PERSIST and os.path.exists(f"{chromadb_path}"):
        db = Chroma(
            persist_directory=f"{chromadb_path}", 
            embedding_function=embeddings
        )
        db.persist()
    else:
        # load split text into vectorstore, as set embedding
        db = Chroma.from_documents(
            texts, 
            embedding=embeddings, 
            persist_directory='chromadb'
        )
        db.persist()    
        
        
    return db


if __name__ == "__main__":
    data_loader()
