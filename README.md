# File structure

## data_loader.py 
Should be run independently when creating vectorstore indexes for the first time or when new files have been pushed to the documents directory

Variable - 
PERSIST set to TRUE will indicate that a vectorstore with indexes already exists
Note; if set to false, new vectorstore and indexes will be created


## query_data.py
Loads the existing vectorstore to query data from

Creates 'retriever' var to expose vectorstore indexes in a retriever interface.

Uses 'RetrievalQA' API to send arguments such as user input to OpenAI API/LLM for a formatted response

NOTE: ConversationalRetrievalChain to be worked on next


## webframework.py
This is the file which will expose/run the web frontend using streamlit. To start app, run;

```$ streamlit run webframework.py```