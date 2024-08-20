#FROM python:3.10-slim-bullseye
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
run pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte e arquivos de configuração
COPY . .

# Copiar e configurar o arquivo .env
COPY .env .env

# Definir variáveis de ambiente
ENV $(cat .env | xargs)

# Expõe as portas utilizadas pela aplicação

RUN chmod +x entrypoint.sh

EXPOSE 8501

CMD ["./entrypoint.sh"]
# Configura o comando de entrada
#CMD ["streamlit", "run", "main.py"]