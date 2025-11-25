import os
import requests
import pandas as pd
from pypdf import PdfReader
from openai import OpenAI

# API keys
GEMINI_KEY = "---------"
OPENAI_KEY = "---------" 

# Modelli
GEMINI_MODEL = "models/gemini-flash-latest"
GPT_MODEL = "gpt-5.1"

# Endpoint Gemini
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"

# Client OpenAI
gpt_client = OpenAI(api_key=OPENAI_KEY)


def estrai_testo_da_pdf(path_pdf: str) -> str:
    """Estrae testo leggibile da PDF"""
    reader = PdfReader(path_pdf)
    testi = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(testi)


def prompt_revisore(testo: str) -> str:
    """Prompt comune per tutti i modelli"""
    return f"""
Sei un revisore scientifico.
Analizza il seguente testo e valuta se è accettabile dal punto di vista logico, concettuale e scientifico, ignorando completamente gli errori tipografici (spazi mancanti, accenti, simboli troncati, note †/‡/§ disallineate, ecc.).
Concentrati esclusivamente su eventuali inconsistenze logiche, errori metodologici, affermazioni non supportate, imprecisioni scientifiche o contraddizioni interne.

Rispondi esclusivamente nel seguente formato:

VERDETTO: ACCEPTED oppure REJECTED
MOTIVO: breve spiegazione (massimo 2 righe) centrata sugli aspetti logici/scientifici del testo analizzato.

Testo:
{testo}
"""


def analizza_con_gemini(testo: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt_revisore(testo)}]}]
    }
    resp = requests.post(GEMINI_URL, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Errore Gemini: {resp.status_code} - {resp.text}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def analizza_con_gpt(testo: str) -> str:
    risposta = gpt_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_revisore(testo)},
        ],
        temperature=0.0,
    )
    return risposta.choices[0].message.content.strip()


def parse_risposta(raw: str):
    """Estrae verdetto e motivazione dal testo del modello"""
    lines = raw.splitlines()
    verdict = "UNKNOWN"
    reason = ""
    for line in lines:
        if "ACCEPTED" in line.upper():
            verdict = "ACCEPTED"
        elif "REJECTED" in line.upper():
            verdict = "REJECTED"
        elif line.strip().lower().startswith("motivo"):
            reason = line.split(":", 1)[-1].strip()
    if not reason:
        # se non trova una riga 'motivo', prende tutto dopo la prima riga
        reason = " ".join(lines[1:]).strip()
    return verdict, reason


def main():
    risultati = []
    papers_dir = "papers"

    for filename in os.listdir(papers_dir):
        if not (filename.endswith(".txt") or filename.endswith(".pdf")):
            continue

        path = os.path.join(papers_dir, filename)

        # lettura file
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                testo = f.read()
        else:
            print(f"Estraggo testo da PDF: {filename}...")
            testo = estrai_testo_da_pdf(path)

        print(f"Analizzo {filename} con Gemini...")
        out_gemini = analizza_con_gemini(testo)
        verdict_g, reason_g = parse_risposta(out_gemini)

        print(f"Analizzo {filename} con ChatGPT...")
        out_gpt = analizza_con_gpt(testo)
        verdict_gpt, reason_gpt = parse_risposta(out_gpt)

        risultati.extend([
            {"paper": filename, "LLM": "Gemini", "Verdetto": verdict_g, "Motivo": reason_g},
            {"paper": filename, "LLM": "ChatGPT", "Verdetto": verdict_gpt, "Motivo": reason_gpt},
        ])

    df = pd.DataFrame(risultati)
    df.to_excel("risultati_confronto.xlsx", index=False)
    print("Analisi completata! File creato: risultati_confronto.xlsx")


if __name__ == "__main__":
    main()
