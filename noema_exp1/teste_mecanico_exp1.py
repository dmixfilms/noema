"""Smoke test mecânico do Experimento 1, sem rede.

Valida: (1) roundtrip de quantização int8/int4 com erro pequeno; (2) truncamento
temporal e de camadas com formas corretas; (3) pipeline completo agente A →
compressão → reidratação → geração, com o modelo aleatório minúsculo do Exp 0.

Uso: python teste_mecanico_exp1.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
EXP0 = RAIZ.parent / "noema_exp0"
sys.path.insert(0, str(EXP0))

import wire  # noqa: E402


def testar_wire_unitario():
    torch.manual_seed(42)
    k = [torch.randn(1, 2, 30, 16) for _ in range(4)]
    v = [torch.randn(1, 2, 30, 16) for _ in range(4)]

    # roundtrip de quantização
    for modo, tol in [("fp16", 1e-3), ("int8", 2e-2), ("int4", 0.5)]:
        w = wire.comprimir(k, v, quant=modo)
        d = wire._dequantizar(w["k"][0], "cpu", torch.float32)
        erro = (d - k[0]).abs().max().item()
        assert erro < tol, f"{modo}: erro {erro} > {tol}"
        assert w["pos_offset"] == 30

    # janela temporal: 4 âncoras + últimos 8 → 12 tokens
    w = wire.comprimir(k, v, janela=8)
    assert w["k"][0]["q"].shape[2] == 12
    cache, pos = wire.reidratar(w, "cpu", torch.float32)
    assert pos == 30 and cache.get_seq_length() == 12

    # camadas: só as 2 últimas viajam; as 2 primeiras voltam zeradas
    w = wire.comprimir(k, v, camadas=2)
    assert w["transmitidas"] == [2, 3]
    cache, _ = wire.reidratar(w, "cpu", torch.float32)
    ks, _ = _kv(cache)
    assert ks[0].abs().sum() == 0 and ks[3].abs().sum() > 0

    print("[teste1] wire.py unitário OK (fp16/int8/int4, janela, camadas)")


def _kv(cache):
    layers = getattr(cache, "layers", None)
    if layers:
        try:
            return [l.keys for l in layers], [l.values for l in layers]
        except AttributeError:
            pass
    return list(cache.key_cache), list(cache.value_cache)


def testar_pipeline():
    sys.path.insert(0, str(EXP0))
    import teste_mecanico  # noqa: E402  (constrói modelo/problemas do Exp 0)

    teste_mecanico.construir_modelo_teste()
    teste_mecanico.preparar_problemas_fake()

    env = dict(os.environ, NOEMA_MODEL=str(teste_mecanico.DIR_MODELO))
    subprocess.run([sys.executable, str(EXP0 / "agente_a.py"), "--n-corte", "16"],
                   check=True, cwd=EXP0, env=env)
    subprocess.run([sys.executable, str(RAIZ / "run_exp1.py"),
                    "--configs", "base_fp16,int4,jan8_teste,cam1_teste"],
                   check=True, cwd=RAIZ, env=env)

    curva = json.load(open(RAIZ / "resultados" / "curva.json", encoding="utf-8"))
    tags = {p["tag"] for p in curva}
    assert {"base_fp16", "int4"} <= tags, tags
    # pontos _teste ficam fora da curva, mas os resultados devem existir
    for t in ["jan8_teste", "cam1_teste"]:
        assert (RAIZ / "resultados" / f"resultados_{t}.jsonl").exists(), t
    base = next(p for p in curva if p["tag"] == "base_fp16")
    int4 = next(p for p in curva if p["tag"] == "int4")
    assert 0 < int4["bytes_medio"] < base["bytes_medio"], "int4 não comprimiu"
    print(f"\n[teste1] OK — pipeline validado; int4 = "
          f"{int4['bytes_medio'] / base['bytes_medio'] * 100:.0f}% do baseline")


if __name__ == "__main__":
    testar_wire_unitario()
    testar_pipeline()
