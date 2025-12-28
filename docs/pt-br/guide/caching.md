# Cache

O ChatMaritaca suporta o mecanismo nativo de cache do LangChain, que pode reduzir significativamente os custos de API e melhorar os tempos de resposta para consultas repetidas.

## Por que Usar Cache?

- **Redução de Custos**: Evite pagar por chamadas de API duplicadas
- **Respostas Mais Rápidas**: Respostas em cache são retornadas instantaneamente
- **Proteção contra Rate Limit**: Menos chamadas de API significa menos chance de atingir limites

## Uso Básico

### Cache em Memória

A forma mais simples de habilitar o cache é com `InMemoryCache`:

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

# Habilitar cache globalmente
set_llm_cache(InMemoryCache())

model = ChatMaritaca()

# Primeira chamada - acessa a API
response1 = model.invoke("Qual é a capital do Brasil?")
print(response1.content)  # "A capital do Brasil é Brasília."

# Segunda chamada com mesma entrada - usa cache (instantâneo!)
response2 = model.invoke("Qual é a capital do Brasil?")
print(response2.content)  # Mesma resposta, sem chamada de API
```

### Cache SQLite

Para cache persistente entre sessões:

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

# Cache persiste no disco
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

### Cache Redis

Para ambientes de produção com cache distribuído:

```python
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache
import redis

# Conectar ao Redis
redis_client = redis.Redis(host="localhost", port=6379)
set_llm_cache(RedisCache(redis_client))
```

## Como as Chaves de Cache Funcionam

As chaves de cache são geradas com base em:

1. **O prompt/mensagens**: Texto exato da entrada
2. **Parâmetros do modelo**: temperature, max_tokens, top_p, etc.
3. **Nome do modelo**: sabia-3.1, sabiazinho-3.1, etc.

Isso significa:

```python
model1 = ChatMaritaca(temperature=0.5)
model2 = ChatMaritaca(temperature=0.9)

# Estes terão chaves de cache DIFERENTES (temperature diferente)
model1.invoke("Olá")  # Cache miss
model2.invoke("Olá")  # Cache miss (chave diferente)

# Este será um cache HIT (mesmo modelo, mesma entrada)
model1.invoke("Olá")  # Usa resposta em cache
```

## Desabilitando Cache

Para desabilitar o cache:

```python
from langchain_core.globals import set_llm_cache

# Desabilitar cache global
set_llm_cache(None)
```

## Boas Práticas

1. **Use cache em desenvolvimento**: Acelera iterações e reduz custos
2. **Considere invalidação de cache**: Respostas em cache não refletem atualizações do modelo
3. **Use cache persistente em produção**: SQLite ou Redis para durabilidade
4. **Monitore taxa de cache hit**: Acompanhe a eficácia da sua estratégia de cache

## Exemplo: Rastreamento de Custos com Cache

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

set_llm_cache(InMemoryCache())
model = ChatMaritaca()

chamadas_api = 0

def invoke_rastreado(prompt: str) -> str:
    global chamadas_api
    response = model.invoke(prompt)
    if response.response_metadata.get("from_cache"):
        print(f"Cache HIT: {prompt[:30]}...")
    else:
        chamadas_api += 1
        print(f"Cache MISS: {prompt[:30]}... (Chamada API #{chamadas_api})")
    return response.content

# Primeira chamada - API
invoke_rastreado("O que é Python?")

# Segunda chamada - Cache
invoke_rastreado("O que é Python?")

print(f"Total de chamadas API: {chamadas_api}")  # 1
```

## Limitações

- **Streaming**: Cache funciona apenas com chamadas sem streaming
- **Tool Calling**: Respostas de tool calls são cacheadas com base na conversa completa
- **Memória**: Cache em memória é perdido quando o processo termina

## Veja Também

- [Documentação de Cache do LangChain](https://python.langchain.com/docs/how_to/llm_caching/)
- [Uso Básico](basic-usage.md)
- [Configuração](../getting-started/configuration.md)
