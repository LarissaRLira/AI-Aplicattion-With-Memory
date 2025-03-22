# 🤖 ChatBot com LangChain e Groq

## 📌 Descrição
Este projeto implementa um chatbot interativo utilizando **LangChain** e **Groq** para processamento de linguagem natural. Ele permite manter um histórico de conversações, otimizar o gerenciamento de memória e estruturar prompts de forma eficiente.

## 🚀 Tecnologias Utilizadas
- **Python** 🐍
- **LangChain** (para criação de fluxos conversacionais)
- **Groq API** (modelo de IA para resposta ao usuário)
- **dotenv** (para gestão de variáveis de ambiente)

## 📂 Estrutura do Código

```python
# Importação das bibliotecas necessárias
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq  
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Inicializa o modelo de IA
model = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

# Armazenamento do histórico
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Criar gerenciador de histórico
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Configurar sessão
config = {"configurable": {"session_id": "chat1"}}

# Exemplo de interação inicial
response = with_message_history.invoke([
    HumanMessage(content="Oie, meu nome é Lari e eu sou engenheira de dados")
], config=config)

print("Resposta do modelo:", response.content)

# Criação de um template de prompt para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda todas as perguntas com precisão no idioma."),
    MessagesPlaceholder(variable_name="messages")  # Permite adicionar mensagens dinamicamente
])

# Conecta o modelo ao template de prompt
chain = prompt | model

# Exemplo de interação usando o template
response = chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Lari")]})
print("Resposta do modelo:", response.content)

# Gerenciamento da memória do chatbot
trimmer = trim_messages(
    max_tokens=45,  # Define um limite máximo de tokens para evitar ultrapassar o contexto
    strategy="last",  # Mantém as últimas mensagens mais recentes
    token_counter=model,  # Usa o modelo para contar os tokens
    include_system=True,  # Inclui a mensagem do sistema no histórico
    allow_partial=False,  # Evita que mensagens fiquem cortadas
    start_on="human"  # Começa a contagem com mensagens humanas
)

# Exemplo de histórico de mensagens
messages = [
    SystemMessage(content="Você é um bom assistente"),
    HumanMessage(content="Oi! Meu nome é Bob"),
    AIMessage(content="Oi, Bob! Como posso te ajudar?"),
    HumanMessage(content="Eu gosto de sorvete de baunilha"),
]

# Aplicar o limitador de memória ao histórico
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
```

## 🛠️ Funcionalidades
✅ Integração com a **API do Groq** para respostas inteligentes.  
✅ **Histórico de conversação** persistente por sessão.  
✅ **Gestão de memória** para otimizar histórico.  
✅ **Criação de prompts dinâmicos** para interações estruturadas.  
✅ **Pipeline de execução** para melhor desempenho.  

## 🔧 Como Executar
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-repositorio/chatbot-langchain-groq.git
   ```
2. Acesse o diretório do projeto:
   ```bash
   cd chatbot-langchain-groq
   ```
3. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Linux/Mac
   venv\Scripts\activate     # Para Windows
   ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
5. Crie um arquivo `.env` e adicione sua chave da API do Groq:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
6. Execute o código:
   ```bash
   python chatbot.py
   ```

## 📌 Exemplo de Uso
```python
response = chain.invoke({
    "messages": [HumanMessage(content="Oi, meu nome é Lari")]
})
print("Resposta do modelo:", response.content)
```

## 🎯 Contribuição
Sinta-se à vontade para contribuir! Abra um **issue** ou faça um **pull request**. 💡

Larissa Souza 

