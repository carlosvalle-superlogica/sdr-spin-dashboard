import os
import csv
import json
import time
import urllib.request
import io
import re
import traceback
from groq import Groq

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets!")

client = Groq(api_key=GROQ_KEY)
CSV_FILE = "dados_chamadas.csv"
CONSOLIDATED_FILE = "consolidated_data.json"
PORTAL_ID = "20131994"

# Modelos ativos oficiais estáveis da API Groq
MODELO_RAPIDO = "llama-3.1-8b-instant"
MODELO_PARERES = "llama-3.3-70b-versatile"

# ==========================================
# 2. SISTEMAS DE SEGURANÇA E MATEMÁTICA RECALIBRADA
# ==========================================
def clean_json(text):
    """Garante a limpeza e extração apenas do objeto JSON retornado pelas APIs de LLM."""
    text = text.strip()
    
    if text.startswith("```json"): 
        text = text[7:]
    elif text.startswith("```"): 
        text = text[3:]
    
    if text.endswith("```"): 
        text = text[:-3]
    
    text = text.strip()
    
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: 
            return match.group(0)
    except Exception: 
        pass
    
    return text

def safe_float(val, default=0.0):
    try: 
        return float(val)
    except Exception: 
        return default

def calcular_segundos(duracao_str):
    """Converte strings de duração formatadas (HH:mm:ss ou mm:ss) em segundos totais."""
    try:
        partes = duracao_str.split(':')
        if len(partes) == 3: 
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        if len(partes) == 2: 
            return int(partes[0]) * 60 + int(partes[1])
    except Exception: 
        pass
    
    return 1

def calcular_nota_operacional(op_data, erro_fatal):
    """
    MATEMÁTICA ADITIVA RÍGIDA (A BUSCA PELO 10.0):
    O SDR começa com 0.0. Para tirar 10.0, precisa de 17 "Sim".
    Qualquer "N/A" soma 0.0 (logo, o teto da nota diminui naturalmente de forma justa).
    Qualquer "Não" aplica penalidade real por erro ou oportunidade desperdiçada.
    """
    nota = 0.0

    chaves_criticas = ['sla', 'passos_ro', 'gestao']  
    # 3 itens (1.0 cada = 3.0 max)
    
    chaves_estrategicas = ['spin', 'dor', 'validacao', 'objecoes', 'produto', 'escuta', 'compreensao'] 
    # 7 itens (0.7 cada = 4.9 max)
    
    chaves_formais = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza', 'gatilhos'] 
    # 7 itens (0.3 cada = 2.1 max)

    # --- TIER 1: CRÍTICOS ---
    for k in chaves_criticas:
        r = op_data.get(k, {}).get('r')
        if r == 'Sim': 
            nota += 1.0
        elif r == 'Não': 
            nota -= 1.0 # Penalidade grave

    # --- TIER 2: ESTRATÉGICOS ---
    for k in chaves_estrategicas:
        r = op_data.get(k, {}).get('r')
        if r == 'Sim': 
            nota += 0.7
        elif r == 'Não': 
            nota -= 0.5 # Penalidade média

    # --- TIER 3: FORMAIS ---
    for k in chaves_formais:
        r = op_data.get(k, {}).get('r')
        if r == 'Sim': 
            nota += 0.3
        elif r == 'Não': 
            nota -= 0.2 # Penalidade leve

    # --- ERRO FATAL ---
    if erro_fatal:
        nota -= 4.0

    # CLAMPEAMENTO SEGURO
    return min(max(nota, 0.0), 10.0)

def executar_chat_com_retentativa(model, messages, response_format, max_retries=6):
    """Executa chamadas à API do Groq controlando de forma inteligente erros de Rate Limit (429)."""
    base_delay = 15  
    
    for attempt in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model=model, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.1
            )
            return chat
            
        except Exception as e:
            err_msg = str(e).lower()
            
            # Captura qualquer erro de limite de requisição ou 429
            if "429" in err_msg or "rate" in err_msg or "too many" in err_msg:
                match = re.search(r"try again in ([0-9.]+)(s|ms)", err_msg)
                
                if match:
                    wait_time = float(match.group(1))
                    if match.group(2) == "ms": 
                        wait_time = wait_time / 1000.0
                else:
                    wait_time = base_delay * (attempt + 1)
                
                wait_time += 5.0 # Margem de segurança
                print(f"   ⚠️ [RATE LIMIT] Limite da API atingido. Aguardando {wait_time:.1f}s (Tentativa {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                
            else:
                raise e
                
    raise RuntimeError(f"Erro: Falha persistente na API da Groq após {max_retries} tentativas.")

# ==========================================
# 3. PIPELINE DE EXECUÇÃO MULTIAGENTE
# ==========================================
def process_all_calls():
    
    if not os.path.exists(CSV_FILE):
        print(f"Erro: Ficheiro {CSV_FILE} não encontrado.")
        return
        
    db = {}
    
    if os.path.exists(CONSOLIDATED_FILE):
        try:
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f: 
                db = json.load(f)
        except Exception: 
            db = {}

    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        delimiter = ';' if ';' in sample else ','
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            call_id = row.get("ID do objeto", "").strip()
            audio_url = row.get("URL de gravação", "").strip()
            result = row.get("Resultado da chamada", "").strip()
            sdr_name = row.get("Atividade atribuída a", "").strip() or "SDR"
            date_str = row.get("Data da atividade", "").strip()
            duration = row.get("Duração da chamada (HH:mm:ss)", "").strip() or "00:00"
            title = row.get("Título da chamada", "").strip()
            
            # Recupera IDs associados e constrói dinamicamente a URL do HubSpot
            deal_id = row.get("Associated Deal IDs", "").strip()
            deal_url = ""
            if deal_id:
                primeiro_id = deal_id.split(',')[0].strip()
                deal_url = f"[https://app.hubspot.com/contacts/](https://app.hubspot.com/contacts/){PORTAL_ID}/deal/{primeiro_id}/"

            if not call_id or not audio_url.startswith("http") or result.lower() not in ["ligação atendida", "connected", "atendida"] or call_id in db:
                continue

            print(f"\n=======================================================")
            print(f"🔥 INICIANDO AUDITORIA | ID: {call_id} | SDR: {sdr_name}")
            print(f"=======================================================")
            
            txt_verif = (title + " " + json.dumps(row)).lower()
            produto_detectado = "CRM" if any(p in txt_verif for p in ["crm", "creci", "corretor"]) else "ERP"

            # Trava de Segurança Isolada para Download de Áudio (Timeout de 10s)
            try:
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response: 
                    audio_bytes = response.read()
            except Exception as e:
                print(f"   ⚠️ [TIMEOUT/ERRO DOWNLOAD] Servidor de áudio falhou ou demorou muito: {e}. Pulando...")
                continue

            try:
                # Prevenção Ativa Contra Loop de Arquivos Enormes na Groq: Tolerância máxima de 20MB.
                tamanho_mb = len(audio_bytes) / (1024 * 1024)
                if tamanho_mb > 20.0:
                    print(f"   ⚠️ [PULANDO CHAMADA] O arquivo possui {tamanho_mb:.2f} MB excedendo o teto seguro de 20MB da API.")
                    continue

                # Transcrição com Whisper-Large-V3
                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)), 
                    model="whisper-large-v3", 
                    response_format="json"
                )
                
                texto = transcription.text
                
                if len(texto) < 10: 
                    print("Chamada ignorada: Áudio sem conteúdo legível ou muito curto.")
                    continue

                segundos = calcular_segundos(duration)
                wps = round(len(texto.split()) / segundos, 2) if segundos > 0 else 0.0

                # --------------------------------------------------
                # AGENTE 1: CONFORMIDADE CALIBRADA (SIM, NÃO E N/A)
                # --------------------------------------------------
                print(" -> Agente 1: Analisando Conformidade e Adaptação...")
                prompt_agente1 = f"""
                Você é o Agente 1: Auditor Comercial Inteligente. Avalie o SDR no produto {produto_detectado}.
                
                MUITO IMPORTANTE - REGRA DO SIM, NÃO E N/A:
                - SIM (Objetivo Atingido): O SDR executou a técnica ativamente OU o lead entregou a informação de bandeja e o SDR teve a maturidade de não repetir a pergunta.
                - NÃO (Oportunidade Desperdiçada): O cenário/oportunidade existiu na conversa, mas o SDR falhou, ignorou a dor, atropelou o cliente, leu script como robô ou quebrou o processo.
                - N/A (Cenário Inexistente): A oportunidade técnica NUNCA se materializou (ex: lead não apresentou objeções, o lead monopolizou a fala sozinho, ou o lead foi desqualificado e a chamada foi abortada antes da agenda).

                DIRETRIZES DE AUDITORIA POR ITEM (LEIA E JULGUE COM ATENÇÃO):
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: SIM se o SDR ouviu e adaptou. NÃO se interrompeu ou ignorou algo para ler o script passivamente. N/A se a ligação foi 100% um monólogo do lead.
                - validacao: SIM se o lead expôs dor e o SDR demonstrou empatia. NÃO se o lead desabafou e o SDR mudou de assunto secamente. N/A se o lead NÃO expôs nenhuma dor na chamada.
                - compreensao: SIM se o SDR usou inteligentemente informações já ditas. NÃO se o SDR perguntou de novo algo que o lead já havia respondido. N/A se a chamada caiu antes de poder avaliar a memória.
                - objecoes: SIM se contornou barreiras. NÃO se o lead trouxe objeção e o SDR aceitou facilmente ou desistiu. N/A se o lead concordou com tudo e NÃO apresentou objeção alguma.

                [2. COMUNICAÇÃO E POSTURA B2B]
                - linguagem: SIM se manteve postura formal. NÃO se usou diminutivos infantis (sisteminha, minutinho, propostinha). N/A se quase não há amostra de voz do SDR para avaliar.
                - receptividade: SIM se executou saudação acolhedora. NÃO se começou de forma ríspida ou confusa. N/A se a gravação já começou no meio da conversa.
                - rapport: SIM se aproveitou contexto para quebrar o gelo. NÃO se iniciou interrogatório seco. N/A se o lead atendeu apressado/agressivo matando a chance de rapport.
                - discurso: SIM se usou vocabulário técnico imobiliário correto. NÃO se falou bobagem técnica. N/A se a ligação abortou antes de entrar no tema do sistema.
                - compreensao_cliente: SIM se após explicar algo, perguntou se fez sentido. NÃO se fez um monólogo gigante sem checar entendimento. N/A se não houve explicação de produto/processo.
                - clareza: SIM se fez perguntas curtas e diretas. NÃO se fez perguntas confusas. N/A se o SDR quase não fez perguntas.

                [3. PROCESSO E QUALIFICAÇÃO]
                - sla: 
                  * {produto_detectado} CRM: Coletou Número de Corretores E Situação do CRECI?
                  * {produto_detectado} ERP: Coletou Quantidade de Contratos E Bancos operados?
                  (SIM se coletou ou se o lead já falou sozinho. NÃO se não descobriu. N/A se o lead foi desqualificado antes disso).
                - spin: SIM se fez investigação sequencial lógica. NÃO se virou um "panfleteiro" apresentando funcionalidades do nada. N/A se o lead já despejou o cenário e problemas todos sozinho.
                - dor: SIM se arrancou uma dor real. NÃO se o lead deu respostas rasas e o SDR não insistiu para descobrir o gargalo. N/A se o lead for irredutível e blindado afirmando estar tudo perfeito.
                - gestao: SIM se mapeou o decisor ou se o lead revelou. NÃO se agendou sem fazer ideia de quem decide. N/A se desqualificou antes dessa fase.
                - passos_ro: SIM se conseguiu a confirmação VERBAL CLARA de que o lead estará num COMPUTADOR na próxima reunião. NÃO se aceitou "vou ver pelo celular/carro". N/A se a ligação NÃO gerou agendamento de reunião.
                - produto: SIM se conectou a solução à dor de forma inteligente. NÃO se tentou empurrar agenda listando recursos inúteis pro cliente. N/A se não evoluiu para o pitch de agendamento.
                - gatilhos: SIM se gerou valor e urgência de agenda. NÃO se agendou de forma desleixada. N/A se a ligação NÃO gerou agendamento de reunião.

                REGRAS DE ERRO FATAL E JSON: 
                - Marque 'erro_fatal': true APENAS se o SDR quebrar o sigilo e passar preço ou agendar reunião com lead fora de perfil.
                - 🚨 NUNCA use aspas duplas (") dentro das suas frases de 'Evidência'. Use sempre aspas simples (').

                Retorne OBRIGATORIAMENTE o JSON preenchendo 'r' com 'Sim', 'Não' ou 'N/A':
                {{
                  "erro_fatal": false,
                  "operacional": {{
                    "escuta": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "validacao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "compreensao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "objecoes": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "linguagem": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "receptividade": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "rapport": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "discurso": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "compreensao_cliente": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "clareza": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "sla": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "spin": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "dor": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "gestao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "passos_ro": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "produto": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "gatilhos": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}
                  }}
                }}
                """
                chat1 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO, 
                    messages=[{"role": "system", "content": prompt_agente1}, {"role": "user", "content": texto}], 
                    response_format={"type": "json_object"}
                )
                res1 = json.loads(clean_json(chat1.choices[0].message.content))
                time.sleep(2)

                # --------------------------------------------------
                # AGENTE 2: SPIN COM AVALIAÇÃO JUSTA E FLEXÍVEL
                # --------------------------------------------------
                print(" -> Agente 2: Avaliando Notas de Metodologia SPIN...")
                prompt_agente2 = """
                Você é o Agente 2: Especialista em Metodologia de Vendas e Psicologia Comercial.
                
                INSTRUÇÕES DE NOTAS (SEJA JUSTO E FLEXÍVEL NAS NOTAS MÉDIAS):
                - Notas 9.0 a 10.0: Seja extremamente rigoroso. Só dê nota máxima se o SDR foi cirúrgico, tocou na ferida do cliente e gerou uma urgência inquestionável usando perguntas de Implicação e Necessidade maravilhosas.
                - Notas 5.0 a 8.5: SEJA FLEXÍVEL. Se o SDR tentou investigar, fez perguntas para identificar o problema e manteve a conversa fluindo de forma minimamente investigativa (mesmo que não tenha sido o SPIN perfeito dos livros), dê notas intermediárias boas para recompensar e validar o esforço técnico.
                - Notas 0.0 a 4.5: Use apenas se o SDR foi totalmente reativo, raso ou apenas leu perguntas engessadas de "Situação" como um robô, sem criar nenhum tipo de valor para a dor do cliente.

                🚨 REGRA DE FORMATAÇÃO: NUNCA use aspas duplas (") na sua justificativa, pois quebra o JSON. Use apenas aspas simples (').

                Responda estritamente neste formato JSON:
                {{
                  "spin_scores": {{"s": 5.0, "p": 6.5, "i": 4.0, "n": 3.0}},
                  "analise_autoridade": "Breve justificativa técnica avaliando a postura do vendedor usando aspas simples se precisar."
                }}
                """
                chat2 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO, 
                    messages=[{"role": "system", "content": prompt_agente2}, {"role": "user", "content": texto}], 
                    response_format={"type": "json_object"}
                )
                res2 = json.loads(clean_json(chat2.choices[0].message.content))
                
                # 🚨 RESPIRO ABSOLUTO DE 35 SEGUNDOS PARA ZERAR O RATE LIMIT 🚨
                print("   ⏳ Dando fôlego estratégico (35s) para a cota da IA limpar antes do modelo pesado...")
                time.sleep(35)

                # --------------------------------------------------
                # AGENTE 3: FEEDBACK TÁTICO, DIRETO E SEM DESCULPAS GENÉRICAS
                # --------------------------------------------------
                print(" -> Agente 3: Construindo Feedback Técnico Estruturado...")
                contexto_sintese = f"Resultados Agente 1: {json.dumps(res1)}\nResultados Agente 2: {json.dumps(res2)}"
                prompt_agente3 = """
                Você é o Diretor de Enablement. Sua missão é dar feedback técnico para o vendedor de forma absurdamente prática, útil e aplicável.

                🚨 REGRA DE OURO INQUEBRÁVEL (TOLERÂNCIA ZERO PARA FEEDBACK GENÉRICO E PALESTRAS DE IA):
                É EXPRESSAMENTE PROIBIDO usar palavras vazias e burocráticas como 'você não seguiu o playbook', 'você ignorou o roteiro', 'faltou sequência lógica' ou 'não seguiu as diretrizes'. 
                Se você apontar um erro, VOCÊ DEVE OBRIGATORIAMENTE FORNECER A FALA EXATA que o vendedor deveria ter usado no lugar, como um treinador entregando uma receita prática de vendas.

                🚨 REGRAS CRÍTICAS DE FORMATAÇÃO JSON (ANTI-ERRO):
                1. Os valores das chaves DO JSON DEVEM SER STRINGS (iniciar e terminar com aspas duplas).
                2. NUNCA use aspas duplas (") DENTRO do seu texto. Se precisar citar algo, use aspas simples (').
                3. NUNCA quebre a linha fisicamente. Para pular linhas e formatar os tópicos em Markdown, use OBRIGATORIAMENTE os caracteres literais \\n.

                Estruture SUA resposta OBRIGATORIAMENTE com estes tópicos em Markdown usando \\n:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Um resumo direto de 2 linhas sobre o controle de conversa e inteligência comercial demonstrada na ligação]

                ### 2. O QUE ERROU
                - [Aponte falhas REAIS encontradas na transcrição. Ex: 'No minuto 03:10, o cliente disse que perde horas, mas você não aprofundou.']

                ### 3. COMO DEVERIA TER FEITO (SCRIPT PRÁTICO)
                - [Forneça o texto exato em formato de fala. Ex: 'Em vez de mudar de assunto, pergunte: Cliente, como fica o seu repasse no final do mês?']
                *Aviso: Consulte a aba 'Playbooks SPIN' no menu lateral.*

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique de forma direta como esse erro esfria o lead.]

                Responda estritamente neste formato JSON:
                {{
                  "parecer_executivo": "### 1. PARECER E POSTURA CONSULTIVA\\nResumo aqui.\\n\\n### 2. O QUE ERROU\\nErro aqui.\\n\\n### 3. COMO DEVERIA TER FEITO\\nCorreção aqui.\\n\\n### 4. CAUSA E EFEITO\\nEfeito aqui.",
                  "plano_de_acao_curto": "Ação exata sem usar aspas duplas no meio do texto."
                }}
                """
                
                chat3 = executar_chat_com_retentativa(
                    model=MODELO_PARERES, 
                    messages=[
                        {"role": "system", "content": prompt_agente3}, 
                        {"role": "user", "content": f"Contexto Analítico: {contexto_sintese}\nTranscrição da Chamada: {texto}"}
                    ], 
                    response_format={"type": "json_object"}
                )
                res3 = json.loads(clean_json(chat3.choices[0].message.content))

                # --------------------------------------------------
                # 4. CONSOLIDAÇÃO DOS DADOS NO ARQUIVO
                # --------------------------------------------------
                s_spin = res2.get("spin_scores", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(res1.get("operacional", {}), res1.get("erro_fatal", False))
                
                urgencia = "SIM" if (nota_op <= 5.0 or nota_spin <= 5.0) else "NÃO"

                db[call_id] = {
                    "id": call_id, 
                    "sdr": sdr_name, 
                    "produto": produto_detectado, 
                    "data": date_str, 
                    "duracao": duration,
                    "wps": wps, 
                    "nota_spin": round(nota_spin, 1), 
                    "nota_op": round(nota_op, 1),
                    "urgencia": urgencia, 
                    "deal_url": deal_url, 
                    "audio_url": audio_url,
                    "notas_s_p_i_n": s_spin, 
                    "formulario": res1.get("operacional", {}),
                    "parecer": res3.get("parecer_executivo", ""), 
                    "sugestoes": res3.get("plano_de_acao_curto", ""),
                    "transcricao": texto
                }
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf: 
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Auditoria Finalizada com Sucesso! SPIN: {nota_spin:.1f} | Conformidade: {nota_op:.1f}")
                
                # Zera o fluxo final com mais uma pequena folga antes da próxima linha do CSV
                time.sleep(10)

            except Exception as e:
                print(f"❌ Erro na auditoria do ID {call_id}: {e}")
                traceback.print_exc()
                # Em caso de erro pesado, o robô dorme e recupera as forças
                time.sleep(30)

if __name__ == "__main__":
    process_all_calls()
