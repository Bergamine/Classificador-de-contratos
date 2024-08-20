import streamlit as st
import langchain
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import pandas as pd
import numpy as np
from pandas_gbq import to_gbq
from google.oauth2 import service_account
import csv
import matplotlib.pyplot as plt
from openai import AzureOpenAI
from dotenv import load_dotenv
import math
import json
from datetime import datetime

load_dotenv()

langchain.verbose = False
agora = datetime.now()
formated_datetime = agora.strftime("%Y%m%d_%H%M%S")


REGION = os.environ['REGION']
CHAT_COMPLETIONS_DEPLOYMENT_NAME = os.environ['CHAT_COMPLETIONS_DEPLOYMENT_NAME']
AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY']

client = AzureOpenAI(
  azure_endpoint = AZURE_OPENAI_ENDPOINT,
  api_key=AZURE_OPENAI_API_KEY,
  api_version="2024-02-01"
)



home_dir = os.path.expanduser("~")
downloads_dir = os.path.join(home_dir, "Downloads")
Logo = "app/images/MBA.png"
MBA_Logo = "app/images/MBA.png"
state = st.session_state.text = ""
#get clients names 


##Titulo da Aba
st.set_page_config(page_title="Classificador de contratos", page_icon=Logo)

df = None
##Titulo da aplicação
st.image("app/images/MBA.png", width=100)
st.title("Contract Classifier")
#st.subheader("Banking's Score Classifier")
prompts_path = os.path.join(os.getcwd(), "app/prompts")
prompt_files = os.listdir(prompts_path)
txt = None
download_message = None
option = None
is_download_on = None
end_message = None
next_form = None
end_message = None

# Dicionário com os dados
data_teste = {}

# Criar o DataFrame
df_classificado = pd.DataFrame(data_teste)

char_break_line = ['\n', '\r', '\r\n', '\u2028', '\u2029']

def safe_json_loads(s):
    try:
        result = json.loads(s.replace('""', '"'))
        result["Classificacao"] = {}
        return result
    except json.JSONDecodeError as e:
        return {"error": str(e), "Classificacao": {}}


def verificar_palavras(texto):
    try:
        if 'Positi' in texto[:20]:
            return 'Positivo'
        elif 'Negati' in texto[:20]:
            return 'Negativo'
        else:
            return 'Not defined'
    except:
        return 'error'

def remove_line_break(text):
    for quebra in char_break_line:
        text = text.replace(quebra, '', regex=True)
    return text

def save_to_bigquery(df, table_id, credentials_file='app/keys/client_secret.json'):
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    table_full_id = f"ProjetoGCP.ClassificadorMBA.{table_id}"
    # Save DataFrame to BigQuery
    try:
        with st.spinner(f'Salvando dados em ProjetoGCP.ClassificadorMBA.{table_id}"'):
            to_gbq(df, table_full_id, project_id='ProjetoGCP', if_exists='replace', credentials=credentials)
            return st.success(f'Dados salvos com sucesso em "ProjetoGCP.ClassificadorMBA.{table_id}"')            
    except FileNotFoundError as e:
        return st.error(f"Error: Credentials file not found. {e}")
    except Exception as e:
        return st.error(f'Erro ao salvar dados: {e}')



def convert_def(df, type="csv"):
    if type == "csv":
        return df.to_csv(encoding='utf-16', index=False, sep=",").encode('utf-16')
    return df.to_excel(f"classify_{system_prompt[:-4]}.xlsx", index=False, engine='xlsxwriter')

with st.sidebar:
    st.image(Logo, width=30)
    anyvar = st.text_input("OpenAI API Key", type="password")

    #AZURE_OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    #AZURE_OPENAI_API_KEY="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"
    model = st.selectbox("Model", ["gpt_4o_20240513"])
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    prompts_text = st.selectbox("Consultar prompts", prompt_files, index=0)
    prompt_path = os.path.join(os.getcwd(), f"app/prompts/{prompts_text}")
    with open(prompt_path, "r") as f:
        prompt_text = f.read()   
    st.text_area("", prompt_text)


def gen_chain(system_prompt, input_text):    

    response = client.chat.completions.create(
        model=CHAT_COMPLETIONS_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
    )
    return response.choices[0].message.content

# Função para processar cada item "text" dentro do JSON
def process_comments(json_data):
    if "comentarios" in json_data:
        for comment in json_data["comentarios"]:
            text = comment.get("text", "")
            # Execute a ação desejada aqui. Por exemplo, imprimir o texto
            print(f"Processing comment: {text}")
            # Supondo que a ação seja modificar o texto de alguma forma:
            comment["text"] = text.upper()  # Exemplo de ação: transformar o texto em maiúsculas
    return json_data

def classify(df, column, isJson="Não"):
    progress = st.progress(0)
    output = "Output"
    classificados = 0
    erros = 0
    rows = len(df)
    bar_progress = 0
    response = None
    if isJson == "Não": 
        for index, row in df.iterrows():    
            print(f'Classificando: {classificados}/{rows}.')
            try: 
                response = gen_chain(txt, row[column])
                df.at[index, output] = response             
                classificados = classificados + 1
            except Exception as e:
                df.at[index, output] = e
                erros = erros + 1
            print(f'Output: {response}')
            bar_progress = classificados/(rows-1)
            try:
                print(f"bar_progress: {bar_progress}")
                progress.progress(bar_progress)
            except Exception as e:
                print(f'erro {e}')

    else:  #dado json
        for index, row in df.iterrows():
            print(f'Classificando: {classificados}/{rows}.')
            #json_data = json.dumps(row[column])
            json_data = json.loads(row[column])
            try:
                for comment in json_data["comentarios"]:
                    comentario = comment.get("text", "")
                    #response = "aqui vai a classificacao"
                    response = gen_chain(txt, comentario)
                    comment["Classificacao"] = response

            except Exception as e:
                df.at[index, output] = e
                erros = erros + 1

            #st.write(json_data["comentarios"])
            classificados = classificados + 1
            json_data = json.dumps(json_data["comentarios"])
            df.at[index, output] = json_data
            print(f'Output: {response}')
            bar_progress = classificados/(rows-1)

            try:
                print(f"bar_progress: {bar_progress}")
                progress.progress(bar_progress)
            except Exception as e:
                print(f'erro {e}')


    #st.data_editor(df)

    #df['Categoria'] = df['Output'].apply(verificar_palavras)

    st.success(f"{classificados} videos foram classificado sucesso!")
    if erros > 0:
        st.error(f"{erros} videos tiveram erro em sua classificação.")
    #bar_chart(df, "Categoria")


    return df

def bar_chart(df_classified, column, x_size=5, y_size=4):
    # Calcular os valores de contagem
    df_plot = df_classified[[column]].value_counts().reset_index(name='count')

    # Criar o gráfico de barras
    fig, ax = plt.subplots(figsize=(x_size, y_size))
    ax.bar(df_plot[column], df_plot['count'])
    ax.set_xlabel('Categoria')
    ax.set_ylabel('Contagem')

    # Adicionar labels/contagens nas barras
    for i in range(len(df_plot)):
        ax.text(df_plot[column][i], df_plot['count'][i] + 0.05, str(df_plot['count'][i]), ha='center', va='bottom')

    # Exibir o gráfico com Streamlit
    st.pyplot(fig)


def download_data(df, file):
    file=f"{file}_classificado.csv"
    with open(file, 'w') as f:
        f.write(df.to_csv())

    st.success(f"O arquivo {file} foi gerado com sucesso!")

with st.form("input_file_form"):
    #text = gen_chain("Adicione 30 ao numero digitado", "20")
    csv_input = st.file_uploader("Arquivo CSV de entrada", type='csv', key='csv_input')
    st.form_submit_button(label="importar")

    if csv_input is not None:
        df = pd.read_csv(csv_input, sep=',')
        df_sample = df.sample(n=10).reset_index().drop(columns=['index']).drop_duplicates(subset=['video_id'])
        df = df.reset_index().drop(columns=['index']).drop_duplicates(subset=['video_id'])
        st.write(df.head())
        st.success("Importado com sucesso")



if df is not None:
    with st.form('Script_form'):
        #path
        isSample = st.radio("Está utilizando uma base amostral? Se sim, será classificado uma amostra de 10 itens do csv", ["Sim", "Não"], index=0)
        if isSample == "Sim":
            df = df_sample

        path = os.path.join(os.getcwd(), "app/prompts")
        roteiros = os.listdir(path)
        roteiros.insert(0, "Selecione")
        system_prompt = st.selectbox("Escolha o script que será interpretado", roteiros, key="roteiro_column", index=0)
        st.form_submit_button(label="Selecionar")

        file_path = os.path.join(os.getcwd(), f"app/prompts/{system_prompt}")
        if system_prompt is not None and system_prompt != "Selecione":
            # Open the file and read the text
            with open(file_path, "r") as f:
                script = f.read()            

            txt = st.text_area('Roteiro de aprendizado', script,
                help="Script responsável por treinar o modelo.")
            script = st.form_submit_button(label="Atualizar roteiro")    
            if script:
                st.success("Atualizado com sucesso")

if txt is not None:
    with st.form("Input_form"):
        list_columns = list(df.columns)
        list_columns.insert(0, "Selecione")
        option = st.selectbox(
            'Página para ser classificada',
            list_columns, key='classification_column', index=0)

        isJsonData = st.radio("Resposta em formato json?", ["Sim", "Não"], index=1)

        selecionado = st.form_submit_button('Selecionar')

if option is not None and option != "Selecione":
    with st.form("Classification_form"):
        if isJsonData == "Sim": 
            new_column= f"{option}_json"
            df[new_column] = df[option].apply(safe_json_loads)
            st.write(df[new_column])   
        st.write(f"O algoritmo classificará o campo *'{option}'*, seguindo o script *'{system_prompt}'*. Deseja prosseguir?")
        pos_classify_action = st.radio("Salvar resultados no BigQuery?", ["Sim", "Não"], index=1)
        classificar = st.form_submit_button(label='Classificar')
        table_id = f"classify_{system_prompt[:-4]}_{formated_datetime}"

        if classificar:
            is_df_classificado = classify(df, option, isJsonData)                    
            if is_df_classificado is not None:                          

                df_classificado = is_df_classificado
                df_classificado = df_classificado.apply(remove_line_break)
                st.data_editor(df_classificado)
                print('Classificado com sucesso')
                if pos_classify_action == "Sim":                 
                    end_message = save_to_bigquery(df_classificado, table_id=table_id)
                    print(end_message)
                    is_download_on = True
                else:   
                    is_download_on = True
                ##############################estado funcional



# if next_form is not None:
#     with st.form("Save results"): 
#         st.write('Como deseja prosseguir?')
#         is_download_on = True
#         save_to_bq = st.form_submit_button(label='Save to BigQuery')
#         print(save_to_bq)
#         if save_to_bq:          
#             end_message = save_to_bigquery(df_classificado, table_id=table_id)


# ,
#     "MBA_USP_ESALQ": "Retorne como Sim ou Não se o video contiver alguma citação da palavra USP ESALQ".

if is_download_on is not None:
    file_type = "csv"

    download_csv_message = st.download_button(label="Download as CSV", \
        data = convert_def(df_classificado, type='csv'), file_name=f"classify_{system_prompt[:-4]}_{formated_datetime}.csv")
    if download_csv_message:
       st.success(f'Arquivo classify_{system_prompt[:-4]}_{formated_datetime}.csv baixado com sucesso')