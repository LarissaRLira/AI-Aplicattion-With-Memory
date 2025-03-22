# Importação das bibliotecas necessárias
import os
from dotenv import load_dotenv, find_dotenv # Biblioteca para carregar variáveis de ambiente
from langchain_groq import ChatGroq  # Integração do LangChain com Groq
from langchain_community.chat_message_histories import ChatMessageHistory  # Histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory  # Classe base para histórico
from langchain_core.runnables.history import RunnableWithMessageHistory  # Permite gerenciar histórico dinamicamente
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Criação de templates para prompts
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages  # Manipulação de mensagens
from langchain_core.runnables import RunnablePassthrough  # Para criar fluxos de execução reutilizáveis
from operator import itemgetter  # Facilita a extração de valores de dicionários

#Carregar as variáveis de ambiente do arquivo .env (para proteger as credenciais)
load_dotenv(find_dotenv())

#Obter a chave da API do GROQ armazenada no arquivo .env 
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Inicializa o modelo de IA utilizando a API do Groq
model = ChatGroq(
    model="gemma2-9b-it", 
    groq_api_key=GROQ_API_KEY
    )

#Dicionário para armazenar o histórico de mensagens
store ={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recupera ou cria um histórico de mensagens para uma determinada sessão.
    Isso permite manter um contexto contínuo para diferentes usuários ou interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#Criar um gerenciador de histórico qeu conecta o modelo ao armazenamento de mensagens
with_message_history = RunnableWithMessageHistory(model, get_session_history)

#Configuração da sessão (identificador único para cada chat/usuário)
config = {"configurable":{"session_id":"chat1"}}

#Exemplo de interação inicial do usuário
response = with_message_history.invoke(
    [HumanMessage(content= "Oie, meu nome é Lari e eu sou engenheira de dados")],
    config=config
)

# Exibir a resposta do modelo 
print("Resposta do modelo:", response.content)

# Criação de um prompt template para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente útil. Responda todas as perguntas com precisão no idioma."),
         MessagesPlaceholder(variable_name="messages")  # Permite adicionar mensagens dinamicamente
    ]
)

# Conecta o modelo ao template de prompt
chain = prompt | model

# Exemplo de interação usando o template
response = chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Lari")]})


# Gerenciamento da memória do chatbot
trimmer = trim_messages(
    max_tokens=45,  # Define um limite máximo de tokens para evitar ultrapassar o contexto
    strategy="last",  # Mantém as últimas mensagens mais recentes
    token_counter=model,  # Usa o modelo para contar os tokens
    include_system=True,  # Inclui a mensagem do sistema no histórico
    allow_partial=False,  # Evita que mensagens fiquem cortadas
    start_on="human"  # Começa a contagem com mensagens humanas
)

#Exemplo ce histórico de messagens
messages = [   SystemMessage(
    content="Você é um bom assistente"),
    HumanMessage(content="Oi! Meu nome é Bob"),
    AIMessage(content="Oi, Bob! Como posso te ajudar?"),
    HumanMessage(content="Eu gosto de sorvete de baunilha"),
]


#Aplicar o limitador de memória ao histórico
response = trimmer.invoke(messages)

# Criando um pipeline de execução para otimizar a passagem de informações
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)  # Aplica a otimização do histórico
    | prompt  # Passa a entrada pelo template de prompt
    | model  # Envia para o modelo
)

# Exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="Qual sorvete eu gosto?")],
    }
)

# Exibe a resposta final do modelo
print("Resposta final:", response.content)
