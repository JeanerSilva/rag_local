# Base Python com ferramentas de build
FROM python:3.10-slim

# Evita bytecode e mantém logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala compiladores e dependências necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código do app
COPY . .

# Comando padrão: iniciar Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
