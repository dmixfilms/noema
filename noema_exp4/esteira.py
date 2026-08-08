"""Definição da esteira de 3 agentes — Experimento 4.

Papéis diferentes sobre o MESMO checkpoint. O que muda entre os agentes é a
função, não os pesos. A instrução de papel é fixa e não carrega conteúdo do
problema (custo de protocolo, não de comunicação).

Lição da 1ª rodada (L 26% × T 42%): injetar a instrução crua no meio do fluxo
herdado tira o modelo da gramática de conversa e degrada a esteira latente.
Na via L a instrução agora entra como TURNO estruturado (fecha o turno do
assistente herdado → turno de usuário com a instrução → reabre o assistente),
tudo com tokens fixos do template.
"""

ETAPAS = [
    {
        "nome": "extrator",
        "instrucao": "Liste os dados numéricos do problema e o que se pede.",
        "max_tokens": 96,
        "prefixo_resposta": None,
    },
    {
        "nome": "calculador",
        "instrucao": "Agora faça as contas passo a passo com esses dados.",
        "max_tokens": 96,
        "rotulo_textual": "Dados extraídos",
        "prefixo_resposta": None,
    },
    {
        "nome": "verificador",
        "instrucao": "Confira o cálculo e dê o resultado final.",
        "max_tokens": None,  # usa config.MAX_NEW_B
        "rotulo_textual": "Cálculo",
        # entra já no turno do assistente, guiando a forma da resposta
        "prefixo_resposta": "Resposta final:",
    },
]
