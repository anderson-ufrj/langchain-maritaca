# Cache Semântico

`MaritacaSemanticCache` substitui os caches de match exato do LangChain. Ele compara prompts por similaridade de cossenos sobre embeddings, então reformulações de perguntas já respondidas reutilizam a resposta em cache.

## Início rápido

```python
from langchain_core.globals import set_llm_cache
from langchain_maritaca import (
    ChatMaritaca,
    DeepInfraEmbeddings,
    MaritacaSemanticCache,
)

cache = MaritacaSemanticCache(
    embeddings=DeepInfraEmbeddings(),
    similarity_threshold=0.95,
    max_entries=1000,
)
set_llm_cache(cache)

modelo = ChatMaritaca()
modelo.invoke("Qual a capital do Brasil?")    # miss -> API
modelo.invoke("Qual é a capital do Brasil?")  # hit (cosseno >= 0.95)
```

## Isolamento por escopo

Entradas são agrupadas por `llm_string`, a representação serializada da configuração do modelo que o LangChain já passa para `BaseCache`. Duas chamadas com `temperature`, `model` ou tools diferentes nunca compartilham entradas — o cache protege de hits cruzados acidentais.

## Ajustando o threshold

- `0.95` (default) é estrito — apenas paráfrases próximas dão hit.
- `0.90` é um meio-termo comum para cargas do tipo FAQ.
- Valores abaixo de `0.85` correm risco de "respostas parecidas mas erradas".

Meça precisão no seu dataset antes de reduzir o threshold.

## Modos de falha

`fail_silently=True` (default) transforma falha no embedding em cache miss. A chamada à API prossegue e o pipeline segue funcionando. Use `fail_silently=False` se o cache for crítico (experimentos de custo em batch, replays de eval) e preferir falhar ruidosamente.

```python
cache = MaritacaSemanticCache(
    embeddings=DeepInfraEmbeddings(),
    similarity_threshold=0.92,
    fail_silently=False,  # levanta exceção em falhas de embedding
)
```

## Eviction LRU

O limite `max_entries` é aplicado por escopo de `llm_string`. Quando excedido, a entrada acessada há mais tempo é removida. Hits contam como acesso, então entradas frequentemente usadas permanecem vivas.

## Async

`alookup` e `aupdate` são suportados e delegam para as versões síncronas, mantendo a busca de similaridade baseada em numpy dentro da mesma thread (o que já é rápido para tamanhos típicos de bucket).

## O que este cache não é

- **Não é persistente** — entradas vivem em memória e somem ao reiniciar o processo.
- **Não é para compartilhamento multi-processo** — para cache distribuído, aguarde integração futura com vector store externo.
