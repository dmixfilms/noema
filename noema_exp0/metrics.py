"""Extração de resposta numérica (GSM8K) e agregação de métricas."""
import json
import re
import statistics

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_PARENTESES = re.compile(r"\([^)]*\)")


def _normalizar(s: str):
    s = s.replace(",", "").rstrip(".")  # 1,234 → 1234
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def extrair_gold(answer: str):
    """No GSM8K a resposta gold vem após '####'."""
    return _normalizar(answer.split("####")[-1].strip())


def extrair_resposta(texto: str):
    """Último número da geração de B, ignorando apartes entre parênteses —
    "a total of 3 bolts (2 blue and 1 white)" responde 3, não 1. A mesma
    régua vale para as três condições."""
    texto = _PARENTESES.sub(" ", texto)
    for n in reversed(_NUM.findall(texto)):
        v = _normalizar(n)
        if v is not None:
            return v
    return None


def acertou(resposta_gerada: str, gold) -> bool:
    r = extrair_resposta(resposta_gerada)
    return r is not None and gold is not None and abs(float(r) - float(gold)) < 1e-6


def ler_jsonl(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _media(valores):
    return statistics.mean(valores) if valores else 0.0


def resumir(linhas):
    """Agrega uma condição: acurácia (geral e sem os flagados), custos e latências."""
    validas = [l for l in linhas if not l.get("concluiu_antes_do_corte")]
    return {
        "n": len(linhas),
        "acuracia": _media([l["acertou"] for l in linhas]),
        "n_sem_flag": len(validas),
        "acuracia_sem_flag": _media([l["acertou"] for l in validas]),
        "tokens_texto_medio": _media([l["tokens_texto_A_para_B"] for l in linhas]),
        "bytes_cache_medio": _media([l["bytes_cache"] for l in linhas]),
        "latencia_handoff_media_s": _media([l["latencia_handoff_s"] for l in linhas]),
        "latencia_total_media_s": _media([l["latencia_total_s"] for l in linhas]),
    }
