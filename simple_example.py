from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# Caminho para seu modelo .gguf (dentro do volume montado)
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Instancia o LLM local
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    temperature=0.7,
    verbose=True,
)

# Cria prompt
prompt = PromptTemplate.from_template("Pergunta: {pergunta}\nResposta:")

# Encadeia o prompt com o modelo
chain = prompt | llm

# Loop interativo simples
print("🤖 LLM Local (LangChain + llama-cpp-python)")
while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ("sair", "exit", "quit"):
        print("Encerrando...")
        break
    resposta = chain.invoke({"pergunta": pergunta})
    print(f"Bot: {resposta}")
