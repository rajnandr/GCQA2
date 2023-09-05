
import os
import requests
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.vectorstores import FAISS
from tenacity import  (    retry,     stop_after_attempt,     wait_random_exponential, )
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st


os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-03-15-preview'
os.environ['OPENAI_API_KEY'] = "052fc719df9e4771838f3295b2ef82a3"
os.environ['OPENAI_API_BASE'] = "https://openaistudio255.openai.azure.com/"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return FAISS.from_documents(**kwargs)

embeddingsAImodel = OpenAIEmbeddings(deployment="textembedding", model="text-embedding-ada-002",
                                     chunk_size=1,    max_retries=10,  maxConcurrency = 2 ,   show_progress_bar=True)

llm = AzureChatOpenAI(model_name="gpt-35-turbo", deployment_name="esujnand", temperature=0)


myTextSplitter = CharacterTextSplitter(
    separator = "Topic:",
    chunk_size = 100,
    chunk_overlap = 20,
    keep_separator = True,
    length_function = len,
    add_start_index = True
)



file_url =  "https://rajnandr.github.io/knowledge.txt"

response = requests.get(file_url)

aFAISSindex = None

try:
    aFAISSindex = FAISS.load_local(folder_path="Je", embeddings=embeddingsAImodel )
    #st.write("loaded FAISS")
except:
    aFAISSindex = None
    st.write("FAISS not found")

if aFAISSindex == None : 
    if (response.status_code):
      inputdata = response.text
      listofDocumentchunks = myTextSplitter.create_documents([inputdata])
      avectorstoreDBwithdata = completion_with_backoff(documents = listofDocumentchunks, embedding= embeddingsAImodel)
      avectorstoreDBwithdata.save_local("Je")
      aFAISSindex = FAISS.load_local(folder_path="Je", embeddings=embeddingsAImodel )




userquestion = "how can we reduce our carbon footprint to align with Carbon Disclosure Project requirements?. Topic: CDP (Carbon Disclosure Project)"

querystringdict = st.experimental_get_query_params()

userqueryquestion = querystringdict['userquestion'] 
userquestion = userqueryquestion[0]
st.write(userquestion)

listofMatchs = aFAISSindex.similarity_search_with_score(userquestion)
if listofMatchs[0][1] > 0.2 : 
  isQuestionoutofscope = True 
else:
  isQuestionoutofscope = False 


queryvector = embeddingsAImodel.embed_query( userquestion )
listofMatchingDocumentchunks = aFAISSindex.similarity_search_by_vector(queryvector)
listofMatchingDocumentchunksMOSTAccurate = listofMatchingDocumentchunks[0:1]


chain = load_qa_chain(llm, chain_type="stuff")
answer = chain({"input_documents": listofMatchingDocumentchunksMOSTAccurate, "question": userquestion},return_only_outputs=False)
print(answer['output_text'])

if isQuestionoutofscope==False:
    st.write(answer['output_text'])
else:
    st.write("This question is not relavent to topic")
    
