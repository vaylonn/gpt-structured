import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import download_loader
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    ServiceContext,
    StorageContext,
)
from dotenv import load_dotenv
# Chargement des variables d'environnement depuis le fichier .env
load_dotenv('./.env')

# Configuration de l'API OpenAI
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.environ["OPENAI_API_BASE"] = 
openai.api_key = os.environ["OPENAI_API_KEY"] = 


#set context window
context_window = 2048

#set number of output tokens
num_output = 512

# Initialisation de l'objet AzureOpenAI
# test1 représente le nom de déployment model sur Azure (le nom du modèle gpt35turbo)
llm = AzureChatOpenAI(deployment_name="default", temperature=0.1, max_tokens=num_output, openai_api_version=openai.api_version, model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})

from llama_index.llm_predictor import StructuredLLMPredictor
llm_predictor = StructuredLLMPredictor(llm=llm)

# Initialisation de l'objet LangchainEmbedding pour l'indexation des documents à partir ici du modèle ada-002 nommé ada-test dans Azureembedding_llm = LangchainEmbedding(
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="learning",
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

# Chargement des documents à partir du répertoire './data'
documents = SimpleDirectoryReader('./data').load_data()
#permet de vérifier que les docs ont bien étés chargés
print('Document ID:', documents[0].doc_id)
print((f"Loaded doc n°1 with {len(documents)} pages"))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    context_window=context_window,
    num_output=num_output,
)

# Création de l'index à partir des documents et du service_context
# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# index.storage_context.persist(persist_dir="./storage")
# print((f"Finished building doc n°1 index with {len(index.docstore.docs)} nodes"))

# Chargement de l'index à partir du stockage (commenté pour le moment)
from llama_index import load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
print((f"Finished loading doc n°1 index from storage with {len(index.docstore.docs)} nodes"))


from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# define output schema
response_schemas = [
    ResponseSchema(name="Education", description="Describes the author's educational experience/background."),
    ResponseSchema(name="Work", description="Describes the author's work experience/background."),
    ResponseSchema(name="Languages", description="Describes the author's programmation languages and skills.")
]


from llama_index.output_parsers import LangchainOutputParser
# define output parser
lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

# Prompt de base du chatbot
# from llama_index import Prompt
template = (
    "Tu trouveras ci-dessous des informations contextuelles. \n"
    "---------------------\n"
    "{context_str}"
   "\n---------------------\n"
    "Tu es un assistant technique, l'utilisateur vas te poser des qusetions sur des informations spécifiques."
    "D'après le contexte, réponds à la question en ne donnant uniquement que la valeur de l'information demandée ou si il y a plusieurs réponses, séparent les d'une virgule. Ne fais pas de phrases. Réponds donc à la question:{query_str}\n"
    "Il se peut que l'utilisateur te pose des questions sur des parties spécifiques du document. Essaye de les retrouver et de répondre à la question"
    "Si la question n'a rien à voir avec les documents, réponds simplement : 'Je suis désolé, je n'ai pas pu trouver la réponse dans les documents que vous m'avez donné.'"
)
# qa_template = Prompt(template)

from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
# format each prompt with output parser instructions
fmt_qa_tmpl = output_parser.format(template)
fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

# Requête envoyée au modèle
# query = "c'est quoi microsoft ?"
# query_engine = index.as_query_engine(similarity_top_k=3, text_qa_template=qa_template)
# answer = query_engine.query(query)

# query index
query_engine = index.as_query_engine(
    service_context=ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    ),
    text_qa_template=qa_prompt, 
    refine_template=refine_prompt, 
)
response = query_engine.query(
    "What are a few things the author did growing up?", 
)
print(str(response))

# Affichage des résultats
#print(answer.get_formatted_sources())
# print('query was:', query)
# print('answer was:', answer)

# Documents source
# for node in answer.source_nodes:
#     print('-----')
#     text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
#     print(f"Text:\t {text_fmt} ...")
#     print(f'Metadata:\t {node.node.extra_info}')
#     print(f'Score:\t {node.score:.3f}')

# {
#     "languages": [
#     {
#         "Type" : "Python",
#         "years" : "3"
#     },
#     {
#         "Type" : "java",
#         "years" : "1"
#     },
#     ]
# }