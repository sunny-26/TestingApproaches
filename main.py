from SPARQLWrapper import SPARQLWrapper, JSON
from llama_index.core import download_loader
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex,ServiceContext,Settings,StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import (
                         ServiceContext,
                         KnowledgeGraphIndex)
from llama_index.core.graph_stores import SimpleGraphStore
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.llms import LlamaCpp
# Callbacks support token-wise streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
def setup_llama_model(model_repo, model_file):
    # Callbacks support token-wise streaming

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file
    )

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.0,
        n_batch=4096,
        n_ctx=4096,
        #top_p=1,
        callback_manager=callback_manager,
        verbose=True  # Verbose is required to pass to the callback manager
    )
    return llm

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
GENERATIVE_AI_MODEL_REPO = "TheBloke/Mistral-7B-v0.1-GGUF"
GENERATIVE_AI_MODEL_FILE = "./mistral-7b-v0.1.Q4_K_M.gguf"
llm = setup_llama_model(GENERATIVE_AI_MODEL_REPO,GENERATIVE_AI_MODEL_FILE)

def generateResponse(prompt, URL):
     #create a documents

     documents=SimpleWebPageReader(html_to_text=True).load_data([URL])
     index = VectorStoreIndex.from_documents(documents, embed_model='local')

     #retrieve data
     query_engine = index.as_query_engine(llm=llm)
     response = query_engine.query(prompt)

     return response


#Factuality


promt1="What medical conditions and procedures are included in the field of cardiology, and how do pediatric cardiologists differ from adult cardiologists in terms of training and specialization?"
URL1 = "https://en.wikipedia.org/wiki/Cardiology"

promt2="Who was the mother of the king, William III of the Netherlands?"
URL2 = "https://en.wikipedia.org/wiki/William_III_of_the_Netherlands"
promt3="What is Luca Pacioli known for in the field of accounting and bookkeeping?"
URL3 = "https://en.wikipedia.org/wiki/Luca Pacioli"
promt4="What is the estimated value of Canada's natural resources in 2019 and how does it contribute to its status as an energy superpower?"
URL4 = "https://en.wikipedia.org/wiki/Canadian_Natural_Resources"
promt5="What is Ian Ashley Murdock known for in the field of software engineering?"
URL5 = "https://en.wikipedia.org/wiki/Ian Ashley Murdock"
"""response1=generateResponse(promt1,URL1)
print("Q1:"+response1+"\n")
response2=generateResponse(promt2,URL2)
response3=generateResponse(promt3,URL3)
response4=generateResponse(promt4,URL4)"""
response5=generateResponse(promt5,URL5)
"""print("Q1:"+response1+"\n")
print("Q2:"+response2+"\n")
print("Q3:"+response3+"\n")
print("Q4:"+response4+"\n")
print("Q5:"+response5+"\n")"""





# Set up the DBpedia SPARQL endpoint
#sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# Define your SPARQL query
query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>

SELECT ?concept ?label ?description
WHERE {
  ?concept a dbo:MedicalSpecialty ;
           rdfs:label ?label .
  OPTIONAL { ?concept dbo:abstract ?description }
  FILTER (lang(?label) = 'en' && regex(?label, "cardiology|pediatric cardiology", "i"))
}
LIMIT 20
"""
"""
# Set the query string
sparql.setQuery(query)

# Set the return format to JSON
sparql.setReturnFormat(JSON)

# Execute the query and parse the results
results = sparql.query().convert()

# Print the results
node_FromGraph_tups = []
for result in results["results"]["bindings"]:
    concept = result["concept"]["value"]
    label = result["label"]["value"]
    description = result.get("description", {}).get("value", "No description available")
    node_FromGraph_tups.append((concept, label, description))

# Print the results
for node_tuple in node_FromGraph_tups:
    print(node_tuple)



ENDPOINT="https://dbpedia.org/sparql"
GRAPH = 'http://dbpedia.org/resource/'
BASE_URI = 'https://dbpedia.org/page/Cardiology'

graph_store = SimpleGraphStore()
for tup in node_FromGraph_tups:
        subject, predicate, obj = tup
        graph_store.upsert_triplet(subject, predicate, obj)


storage_context = StorageContext.from_defaults(graph_store=graph_store)
service_context = ServiceContext.from_defaults(llm=llm, embed_model='local')

index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context,
    )
query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )
question="What is cardiology"
response = query_engine.query(
        question,

    )
"""


