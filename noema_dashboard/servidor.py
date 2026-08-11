"""Noema Dashboard — local web UI to run and inspect the experiments.

Usage:
    pip install fastapi uvicorn
    python servidor.py            # opens on http://localhost:7860

Everything runs locally: the dashboard talks to the same experiment scripts
(noema_exp*/) via subprocesses, and the live handoff playground drives the
model in-process through noema_exp0/nucleo.py.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parent
REPO = RAIZ.parent
sys.path.insert(0, str(REPO / "noema_exp0"))

MODELOS_SUGERIDOS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

EXPERIMENTOS = {
    "exp0_smoke": {"dir": "noema_exp0", "script": "run_experiment.py",
                   "args": ["--smoke"], "nome": "Exp 0 — smoke (3 problems)"},
    "exp0": {"dir": "noema_exp0", "script": "run_experiment.py",
             "args": ["--kl"], "nome": "Exp 0 — full run + KL fidelity"},
    "exp1": {"dir": "noema_exp1", "script": "run_exp1.py",
             "args": [], "nome": "Exp 1 — compression curves"},
    "exp1_kl": {"dir": "noema_exp1", "script": "fidelidade_exp1.py",
                "args": [], "nome": "Exp 1 — KL per configuration"},
    "exp05": {"dir": "noema_exp05", "script": "run_exp05.py",
              "args": [], "nome": "Exp 0.5 — continuous thought"},
    "exp2": {"dir": "noema_exp2", "script": "run_exp2.py",
             "args": [], "nome": "Exp 2 — interlingua (trains the adapter)"},
    "exp4_smoke": {"dir": "noema_exp4", "script": "run_exp4.py",
                   "args": ["--smoke"], "nome": "Exp 4 — pipeline smoke"},
    "exp4": {"dir": "noema_exp4", "script": "run_exp4.py",
             "args": [], "nome": "Exp 4 — 3-agent pipeline"},
}

IMAGENS = {
    "exp0": REPO / "noema_exp0" / "resultados" / "grafico.png",
    "exp1": REPO / "noema_exp1" / "resultados" / "curva.png",
    "exp4": REPO / "noema_exp4" / "resultados" / "esteira.png",
}

app = FastAPI(title="Noema Dashboard")
_job = {"proc": None, "nome": None, "log": RAIZ / "job.log", "inicio": None}
_demo_lock = threading.Lock()


def _modelo_atual():
    return os.environ.get("NOEMA_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def _job_rodando():
    return _job["proc"] is not None and _job["proc"].poll() is None


# ---------------------------------------------------------------- status
@app.get("/api/status")
def status():
    vram = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return {
        "cuda": torch.cuda.is_available(),
        "gpu": gpu, "vram_gb": vram,
        "modelo": _modelo_atual(),
        "modelos_sugeridos": MODELOS_SUGERIDOS,
        "experimentos": {k: v["nome"] for k, v in EXPERIMENTOS.items()},
        "rodando": _job_rodando(),
        "job_nome": _job["nome"] if _job_rodando() else None,
    }


# ------------------------------------------------------------- run jobs
class RodarReq(BaseModel):
    exp: str
    modelo: str | None = None
    n_problemas: int | None = None
    n_corte: int | None = None
    max_new_b: int | None = None


@app.post("/api/rodar")
def rodar(req: RodarReq):
    if _job_rodando():
        return JSONResponse({"erro": "a job is already running"}, status_code=409)
    if req.exp not in EXPERIMENTOS:
        return JSONResponse({"erro": "unknown experiment"}, status_code=400)

    e = EXPERIMENTOS[req.exp]
    env = dict(os.environ)
    if req.modelo:
        env["NOEMA_MODEL"] = req.modelo
        os.environ["NOEMA_MODEL"] = req.modelo
    if req.n_problemas:
        env["NOEMA_N_PROBLEMAS"] = str(req.n_problemas)
    if req.max_new_b:
        env["NOEMA_MAX_NEW_B"] = str(req.max_new_b)

    args = list(e["args"])
    if req.exp.startswith("exp0"):
        if req.n_problemas:
            args += ["--n-problemas", str(req.n_problemas)]
        if req.n_corte:
            args += ["--n-corte", str(req.n_corte)]

    log = open(_job["log"], "w", encoding="utf-8")
    _job["proc"] = subprocess.Popen(
        [sys.executable, "-u", str(REPO / e["dir"] / e["script"]), *args],
        cwd=REPO / e["dir"], env=env, stdout=log, stderr=subprocess.STDOUT)
    _job["nome"] = e["nome"]
    _job["inicio"] = time.time()
    return {"ok": True, "nome": e["nome"]}


@app.post("/api/parar")
def parar():
    if _job_rodando():
        _job["proc"].terminate()
        return {"ok": True}
    return {"ok": False}


@app.get("/api/log")
def log(offset: int = 0):
    texto = ""
    if _job["log"].exists():
        with open(_job["log"], encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            texto = f.read()
    rodando = _job_rodando()
    codigo = None if rodando or _job["proc"] is None else _job["proc"].returncode
    return {"texto": texto, "offset": offset + len(texto.encode("utf-8")),
            "rodando": rodando, "codigo": codigo,
            "duracao_s": round(time.time() - _job["inicio"], 1) if _job["inicio"] else 0}


# ------------------------------------------------------- live playground
class DemoReq(BaseModel):
    pergunta: str
    n_corte: int = 120
    max_new: int = 160
    modelo: str | None = None
    comparar_texto: bool = True


@app.post("/api/demo")
def demo(req: DemoReq):
    if _job_rodando():
        return JSONResponse(
            {"erro": "an experiment is running — wait for it (shared GPU)"},
            status_code=409)
    with _demo_lock:
        if req.modelo and req.modelo != _modelo_atual():
            os.environ["NOEMA_MODEL"] = req.modelo
        import config
        import nucleo
        if req.modelo and config.MODEL != req.modelo:
            config.MODEL = req.modelo
            nucleo._model = None
            nucleo._tok = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.manual_seed(config.SEED)
        tok, _ = nucleo.carregar_modelo()

        # Agent A thinks and is cut mid-reasoning
        t0 = time.time()
        cache, texto_parcial, concluiu, _ = nucleo.agente_A_pensa(
            req.pergunta, req.n_corte)
        t_pensar = time.time() - t0

        # the handoff artifact: serialized to disk, loaded back
        arq = RAIZ / "demo_cache.pt"
        t0 = time.time()
        nbytes = nucleo.serializar_cache(cache, str(arq))
        t_serializar = time.time() - t0
        t0 = time.time()
        cache_b = nucleo.carregar_cache(str(arq))
        t_carregar = time.time() - t0

        # Agent B: inherits the thought, receives ONLY the fixed suffix
        ids = tok(config.SUFIXO, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(config.DEVICE)
        t0 = time.time()
        resposta_L, _ = nucleo.gerar_greedy(ids, cache=cache_b, max_new=req.max_new)
        t_gerar_L = time.time() - t0
        arq.unlink(missing_ok=True)

        saida = {
            "texto_parcial": texto_parcial,
            "concluiu_antes_do_corte": concluiu,
            "resposta_latente": resposta_L,
            "bytes_cache": nbytes,
            "tokens_texto_L": 0,
            "latencias": {"A_pensar_s": round(t_pensar, 3),
                          "serializar_s": round(t_serializar, 3),
                          "handoff_s": round(t_carregar, 3),
                          "B_gerar_s": round(t_gerar_L, 3)},
        }

        if req.comparar_texto:
            msgs = [{"role": "user",
                     "content": req.pergunta + "\nPense passo a passo."}]
            ids_p = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt")
            ids_c = tok(texto_parcial + config.SUFIXO, return_tensors="pt",
                        add_special_tokens=False).input_ids
            ids_t = torch.cat([ids_p, ids_c], dim=-1).to(config.DEVICE)
            t0 = time.time()
            resposta_T, t_prefill = nucleo.gerar_greedy(ids_t, max_new=req.max_new)
            saida["resposta_textual"] = resposta_T
            saida["tokens_texto_T"] = len(tok(req.pergunta + texto_parcial,
                                              add_special_tokens=False).input_ids)
            saida["latencias"]["T_reprefill_s"] = round(t_prefill, 3)
            saida["latencias"]["T_total_s"] = round(time.time() - t0, 3)
        return saida


# ------------------------------------------------------------- results
@app.get("/api/resultados")
def resultados():
    import metrics
    out = {}
    exp0 = {}
    for cond in ("L", "T", "Z"):
        p = REPO / "noema_exp0" / "resultados" / f"resultados_{cond}.jsonl"
        if p.exists():
            exp0[cond] = metrics.resumir(metrics.ler_jsonl(p))
    if exp0:
        out["exp0"] = exp0
    p = REPO / "noema_exp1" / "resultados" / "curva.json"
    if p.exists():
        out["exp1"] = json.load(open(p, encoding="utf-8"))
    p = REPO / "noema_exp0" / "resultados" / "fidelidade_kl.json"
    if p.exists():
        out["exp0_kl"] = json.load(open(p, encoding="utf-8"))["kl_medio"]
    p = REPO / "noema_exp4" / "resultados" / "resumo.json"
    if p.exists():
        out["exp4"] = json.load(open(p, encoding="utf-8"))
    for k, img in IMAGENS.items():
        if img.exists():
            out.setdefault("imagens", []).append(k)
    return out


@app.get("/api/imagem/{nome}")
def imagem(nome: str):
    if nome in IMAGENS and IMAGENS[nome].exists():
        return FileResponse(IMAGENS[nome])
    return JSONResponse({"erro": "not found"}, status_code=404)


@app.get("/")
def index():
    return FileResponse(RAIZ / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n  Noema Dashboard →  http://localhost:7860\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
