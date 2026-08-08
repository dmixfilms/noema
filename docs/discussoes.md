# Discussões públicas — perguntas que o projeto recebeu e o que elas ensinam

*Registro das trocas técnicas após a divulgação (ago/2026). Cada pergunta boa
recebida vira uma entrada aqui; as recorrentes são promovidas ao FAQ do README.*

---

## 1. "O Ollama não poderia fazer isso?"

**Resposta curta:** não — API só de texto; sem acesso a `past_key_values`,
hidden states, `inputs_embeds` ou logits. O prompt cache do Ollama é otimização
invisível da própria sessão; o Noema usa o cache como meio de comunicação entre
agentes. Nuance: o llama.cpp (motor do Ollama) tem save/restore de slot que
reproduziria o handoff básico, mas nada da instrumentação científica.

**O que ensina:** a pergunta mais comum confunde *cache como otimização* com
*cache como canal*. A distinção virou o núcleo do FAQ.

## 2. "Não é o que os workflows do ChatGPT/Claude fazem?"

**Resposta curta:** não — workflows comerciais passam texto entre agentes;
as APIs não expõem o estado interno. O prompt caching delas é otimização de
servidor para prefixos idênticos, não estado transferível.

**O que ensina:** o público associa "multi-agente" a orquestração por API.
O contraste local × API precisa vir cedo em qualquer apresentação.

## 3. "Os grandes provedores já não fazem isso dentro dos servidores?"

**Resposta curta:** em parte sim — prefix caching e inferência desagregada
movem KV-cache entre servidores em produção (Mooncake, vLLM, NVIDIA Dynamo),
o que *valida* a premissa. Mas é otimização de identidade (mesmo prefixo, byte
a byte); os agentes deles seguem trocando texto. O Noema opera na camada
semântica (agente diferente herda e redireciona o raciocínio) e mede o canal.

**O que ensina:** a objeção mais sofisticada. A resposta certa não nega — situa:
infra ≠ canal semântico ≠ ciência do canal.

## 4. "Então é o mesmo que os loops paralelos/lineares da Anthropic?"

**Resposta curta:** os loops de agentes da Anthropic (paralelos ou lineares)
são exatamente o *baseline* do Noema: orquestrador manda prompt em texto,
subagente devolve relatório em texto, cada fronteira re-processa e perde
nuance. É a "condição T" dos experimentos. O texto tem a vantagem real de ser
auditável; o Noema mostra que no território local o custo dele é opcional.

**O que ensina:** confirmação vinda do próprio uso dos produtos — as pessoas
percebem o telefone sem fio quando perguntam "como você faz isso?" ao modelo.

## 5. "É mais seguro porque humanos não conseguem ler?"

**Resposta curta:** não — o decodificador (modelo aberto) é público; quem tem o
arquivo extrai tudo. Obscuridade ≠ segurança, e ainda cega auditoria. Proteção:
criptografia clássica, perímetro, camada de contratos (Exp 3).

**O que ensina:** a intuição "ilegível = seguro" é comum e perigosa; corrigi-la
proativamente protege a credibilidade do projeto.
