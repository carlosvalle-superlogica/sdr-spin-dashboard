import os
import csv
import json
import time
import urllib.request
import io
import re
import traceback
from urllib.error import URLError
from groq import Groq

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets do GitHub!")

client = Groq(api_key=GROQ_KEY)

CSV_FILE = "dados_chamadas.csv"
PROMPT_FILE = "evaluation_prompt.txt"
CONSOLIDATED_FILE = "consolidated_data.json"

# ==========================================
# 2. FUNÇÕES AUXILIARES DE SUPORTE
# ==========================================
def load_prompt():
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Aviso: Arquivo de prompt não encontrado. Usando prompt padrão. Erro: {e}")
        return "Atue como auditor de SPIN Selling e avalie esta transcrição."

def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def safe_float(value, default=0.0):
    try:
        val = float(value)
        return val if 0 <= val <= 10 else default
    except:
        return default

# ==========================================
# 3. NÚCLEO DE PROCESSAMENTO DE LIGAÇÕES
# ==========================================
def process_all_calls():
    if not os.path.exists(CSV_FILE):
        print(f"Erro: Arquivo {CSV_FILE} não encontrado!")
        return

    print(f"✅ Arquivo de dados encontrado: {CSV_FILE}")
    prompt_content = load_prompt()
    
    db = {}
    if os.path.exists(CONSOLIDATED_FILE):
        try:
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except json.JSONDecodeError:
            db = {}

    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        delimiter = ';' if ';' in sample else ','
    
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        linhas_processadas = 0
        
        for row in reader:
            call_id = row.get("ID do objeto", "").strip()
            audio_url = row.get("URL de gravação", "").strip()
            result = row.get("Resultado da chamada", "").strip()
            
            sdr_name = row.get("Atividade atribuída a", "").strip() or "SDR Não Identificado"
            date_str = row.get("Data da atividade", "").strip() or "Data Indisponível"
            duration = row.get("Duração da chamada (HH:mm:ss)", "").strip() or "00:00"
            title = row.get("Título da chamada", "").strip() or "Chamada de Vendas"

            if not call_id or not audio_url or not audio_url.startswith("http"):
                continue
            
            if result.lower() not in ["ligação atendida", "connected", "atendida"]:
                continue

            if call_id in db:
                continue

            print(f"-> [NOVA LIGAÇÃO] Iniciando ID {call_id} | SDR: {sdr_name}...")
            
            try:
                req = urllib.request.Request(
                    audio_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                
                with urllib.request.urlopen(req, timeout=45) as response:
                    content_length = response.getheader('Content-Length')
                    if content_length and int(content_length) > 24 * 1024 * 1024:
                        print(f"⚠️ Ignorado: Chamada {call_id} descartada por tamanho excessivo.")
                        continue
                    
                    buffer = io.BytesIO()
                    bytes_read = 0
                    max_bytes = 24 * 1024 * 1024
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_read += len(chunk)
                        if bytes_read > max_bytes:
                            break
                        buffer.write(chunk)
                    
                    if bytes_read > max_bytes:
                        continue
                    
                    audio_bytes = buffer.getvalue()

                # Transcrição Whisper
                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="json"
                )
                texto_ligacao = transcription.text

                if len(texto_ligacao.strip()) < 10:
                    continue

                # Instruções rígidas para as Notas SPIN e Sugestões obrigatórias menores que 10
                prompt_sistema = (
                    f"{prompt_content}\n\n"
                    "REGRA CRÍTICA DE EXECUÇÃO:\n"
                    "1. Responda estritamente com um objeto JSON válido.\n"
                    "2. Se a 'nota_spin' global for MENOR QUE 10.0, você DEVE obrigatoriamente preencher o campo 'sugestoes' detalhando onde foi o erro/falha do SDR na ligação e qual a sugestão prática de melhoria.\n"
                    "3. Forneça notas de 0.0 a 10.0 para cada pilar do método SPIN nos campos descritos.\n\n"
                    "ESTRUTURA OBRIGATÓRIA DO JSON:\n"
                    "{\n"
                    "  \"nota_spin\": 7.5,\n"
                    "  \"nota_s\": 8.0,\n"
                    "  \"nota_p\": 7.0,\n"
                    "  \"nota_i\": 6.5,\n"
                    "  \"nota_n\": 8.5,\n"
                    "  \"avaliacao\": \"Sua análise geral aqui...\",\n"
                    "  \"sugestoes\": \"[OBRIGATÓRIO PARA NOTAS < 10] O SDR errou em X... Sugerimos corrigir para Y...\"\n"
                    "}"
                )

                # ALTERADO PARA O MODELO INSTANT COM LIMITE DE 500K TOKENS DIÁRIOS
                chat_completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": f"Transcrição:\n\n{texto_ligacao}"}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                raw_content = chat_completion.choices[0].message.content
                clean_text = clean_json_response(raw_content)
                analysis_data = json.loads(clean_text)
                
                nota_global = safe_float(analysis_data.get("nota_spin"), default=7.0)

                db[call_id] = {
                    "id": call_id,
                    "sdr": sdr_name,
                    "data": date_str,
                    "titulo": title,
                    "duracao": duration,
                    "audio_url": audio_url,
                    "nota_spin": nota_global,
                    "nota_s": safe_float(analysis_data.get("nota_s"), default=nota_global),
                    "nota_p": safe_float(analysis_data.get("nota_p"), default=nota_global),
                    "nota_i": safe_float(analysis_data.get("nota_i"), default=nota_global),
                    "nota_n": safe_float(analysis_data.get("nota_n"), default=nota_global),
                    "avaliacao": analysis_data.get("avaliacao", "Análise de conformidade executada."),
                    "sugestoes": analysis_data.get("sugestoes", "Nenhuma falha crítica detectada."),
                    "transcricao": texto_ligacao
                }
                
                linhas_processadas += 1
                
                tmp_file = CONSOLIDATED_FILE + ".tmp"
                with open(tmp_file, 'w', encoding='utf-8') as sf:
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                os.replace(tmp_file, CONSOLIDATED_FILE)

                print(f"✅ Sucesso: Chamada {call_id} salva no JSON (Nota: {nota_global})")
                time.sleep(2)

            except Exception as e:
                if "429" in str(e) or "rate_limit_exceeded" in str(e):
                    print("\n🛑 Limite de requisições temporárias atingido no Groq. Salvando progresso.")
                    break
                else:
                    print(f"Erro no processamento do ID {call_id}: {e}")
                    time.sleep(2)
                    continue

    print(f"\n✅ EXECUÇÃO CONCLUÍDA: {linhas_processadas} novas chamadas processadas.")

if __name__ == "__main__":
    process_all_calls()
