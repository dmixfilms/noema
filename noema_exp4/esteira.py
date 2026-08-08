"""Definição da esteira de 3 agentes — Experimento 4.

Papéis diferentes sobre o MESMO checkpoint (como 90% dos sistemas multi-agente
reais são montados). O que muda entre os agentes é a função, não os pesos.

Na via latente, a instrução de papel é o único texto injetado — ela é fixa e
não carrega conteúdo do problema, então é contabilizada à parte (custo de
protocolo, não de comunicação).
"""

ETAPAS = [
    {
        "nome": "extrator",
        "instrucao_papel": "Liste os dados numéricos do problema e o que se pede.",
        "max_tokens": 96,
    },
    {
        "nome": "calculador",
        "instrucao_papel": "\n\nAgora faça as contas passo a passo com esses dados.",
        "max_tokens": 96,
        # rótulo usado só na via textual, ao remontar o contexto em linguagem
        "rotulo_textual": "Dados extraídos",
    },
    {
        "nome": "verificador",
        "instrucao_papel": "\n\nConfira o cálculo e dê o resultado.\nResposta final:",
        "max_tokens": None,  # usa config.MAX_NEW_B
        "rotulo_textual": "Cálculo",
    },
]
