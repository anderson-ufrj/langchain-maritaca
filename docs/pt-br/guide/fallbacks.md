# Cadeias Inteligentes de Fallback

`ChatMaritaca` oferece um helper que envolve `Runnable.with_fallbacks()` do LangChain com uma ordem de modelos curada e um filtro de erros transientes.

## Uso básico

```python
from langchain_maritaca import ChatMaritaca

modelo = ChatMaritaca(model="sabia-3.1", temperature=0.3).with_smart_fallbacks()
resposta = modelo.invoke("Explique o artigo 5 da Constituição brasileira.")
```

A chamada tenta `sabia-3.1` primeiro e só recorre a `sabiazinho-4` e depois `sabiazinho-3.1` em falhas transientes (`429`, `502`, `503`, `504` ou timeout). Erros 4xx de autenticação/validação propagam imediatamente para que você identifique bugs de configuração.

## Construindo do zero

```python
from langchain_maritaca import ChatMaritaca

cadeia = ChatMaritaca.with_smart_fallbacks_from_primary(
    primary="sabia-3.1",
    api_key="...",
    temperature=0.3,
)
```

Todos os argumentos nomeados são passados para o `ChatMaritaca` primário e herdados pelos fallbacks.

## Customizando a cadeia

```python
modelo = ChatMaritaca(model="sabia-3.1").with_smart_fallbacks(
    fallbacks=["sabiazinho-3.1"],
)
```

Também é possível sobrescrever o filtro de exceções se precisar de comportamento de retry mais amplo:

```python
import httpx

modelo = ChatMaritaca(model="sabia-3.1").with_smart_fallbacks(
    exceptions_to_handle=(httpx.TimeoutException, httpx.HTTPStatusError),
)
```

## Cadeia padrão

| Primário | Ordem de fallback |
|---|---|
| `sabia-3.1` | `sabiazinho-4` → `sabiazinho-3.1` |
| `sabiazinho-4` | `sabia-3.1` → `sabiazinho-3.1` |
| `sabiazinho-3.1` | `sabia-3.1` → `sabiazinho-4` |

Modelos primários desconhecidos (ex.: variantes futuras do Sabiá ainda não conhecidas pelo pacote) exigem uma lista `fallbacks=[...]` explícita, caso contrário o helper levanta `ValueError` na construção.

## Quando não usar

- Se você só precisa de retries no mesmo modelo em caso de rate limit, prefira `retry_if_rate_limited` — fallback trocaria o modelo sem necessidade.
- Em cenários que exigem pinagem estrita de modelo (reprodutibilidade de avaliação, por exemplo), não use fallbacks para manter o comportamento determinístico.
