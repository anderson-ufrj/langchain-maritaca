# langchain-maritaca

[![PyPI version](https://img.shields.io/pypi/v/langchain-maritaca.svg)](https://pypi.org/project/langchain-maritaca/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-maritaca.svg)](https://pypi.org/project/langchain-maritaca/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-maritaca.svg)](https://pypi.org/project/langchain-maritaca/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/anderson-ufrj/langchain-maritaca/actions/workflows/ci.yml/badge.svg)](https://github.com/anderson-ufrj/langchain-maritaca/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anderson-ufrj/langchain-maritaca/graph/badge.svg)](https://codecov.io/gh/anderson-ufrj/langchain-maritaca)

[🇺🇸 Read in English](README.md)

Pacote de integração conectando [Maritaca AI](https://www.maritaca.ai/) e [LangChain](https://langchain.com/) para modelos de linguagem otimizados para Português Brasileiro.

**Autor:** Anderson Henrique da Silva
**Localização:** Minas Gerais, Brasil
**GitHub:** [anderson-ufrj](https://github.com/anderson-ufrj)

## Visão Geral

A Maritaca AI oferece modelos de linguagem de última geração para Português Brasileiro, incluindo a família de modelos Sabiá. Esta integração permite usar os modelos da Maritaca de forma transparente dentro do ecossistema LangChain.

### Modelos Disponíveis

| Modelo | Contexto | Input (R$/1M) | Output (R$/1M) | Vision |
|--------|----------|---------------|----------------|--------|
| `sabia-4` | 128k | R$5,00 | R$20,00 | Sim |
| `sabiazinho-4` | 128k | R$1,00 | R$4,00 | Sim |

> **Nota:** Todos os modelos suportam entradas multimodais (imagens).

## Instalação

```bash
pip install langchain-maritaca
```

## Configuração

Defina sua chave de API da Maritaca como variável de ambiente:

```bash
export MARITACA_API_KEY="sua-chave-api"
```

Ou passe diretamente para o modelo:

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(api_key="sua-chave-api")
```

## Uso

### Uso Básico

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(
    model="sabia-4",
    temperature=0.7,
)

messages = [
    ("system", "Você é um assistente prestativo especializado em cultura brasileira."),
    ("human", "Quais são as principais festas populares do Brasil?"),
]

response = model.invoke(messages)
print(response.content)
```

### Streaming

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(model="sabia-4", streaming=True)

for chunk in model.stream("Conte uma história sobre o folclore brasileiro"):
    print(chunk.content, end="", flush=True)
```

### Uso Assíncrono

```python
import asyncio
from langchain_maritaca import ChatMaritaca

async def main():
    model = ChatMaritaca(model="sabia-4")
    response = await model.ainvoke("Qual é a receita de pão de queijo?")
    print(response.content)

asyncio.run(main())
```

### Com LangChain Expression Language (LCEL)

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.prompts import ChatPromptTemplate

model = ChatMaritaca(model="sabia-4")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {topic}."),
    ("human", "{question}"),
])

chain = prompt | model

response = chain.invoke({
    "topic": "história do Brasil",
    "question": "Quem foi Tiradentes?"
})
print(response.content)
```

### Com Tool Calling (Chamada de Funções)

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Obtém o clima atual para uma cidade."""
    return f"O clima em {city} está ensolarado, 25°C"

model = ChatMaritaca(model="sabia-4")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("Como está o tempo em São Paulo?")
print(response)
```

### Vision / Multimodal (Imagens)

Todos os modelos da Maritaca suportam entrada de imagens. Você pode enviar imagens via URL ou base64:

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca(model="sabiazinho-4")

# Com URL da imagem
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "O que você vê nesta imagem?"},
        {"type": "image", "url": "https://example.com/imagem.jpg"}
    ])
])
print(response.content)

# Com imagem codificada em base64
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Descreva esta imagem em detalhes"},
        {"type": "image", "base64": "iVBORw0KGgo...", "mime_type": "image/png"}
    ])
])
```

Também compatível com o formato `image_url` da OpenAI:

```python
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "O que há nesta imagem?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/foto.jpg"}}
    ])
])
```

### Com Saída Estruturada

```python
from langchain_maritaca import ChatMaritaca
from pydantic import BaseModel, Field

class Pessoa(BaseModel):
    """Informações sobre uma pessoa."""
    nome: str = Field(description="Nome da pessoa")
    idade: int = Field(description="Idade da pessoa")

model = ChatMaritaca(model="sabia-4")
structured_model = model.with_structured_output(Pessoa)

result = structured_model.invoke("João tem 25 anos e mora em São Paulo")
print(result)  # Pessoa(nome="João", idade=25)
```

### Com Embeddings para RAG

```python
from langchain_maritaca import ChatMaritaca, DeepInfraEmbeddings

# Embeddings para recuperação de documentos
embeddings = DeepInfraEmbeddings()
vectors = embeddings.embed_documents([
    "O Brasil foi descoberto em 1500",
    "A capital do Brasil é Brasília"
])

# Chat para geração de respostas
chat = ChatMaritaca(model="sabia-4")
```

### Com Cache

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_maritaca import ChatMaritaca

# Habilitar cache globalmente
set_llm_cache(InMemoryCache())

model = ChatMaritaca(model="sabia-4")

# Primeira chamada - acessa a API
response1 = model.invoke("Qual é a capital do Brasil?")

# Segunda chamada - usa cache (instantâneo, sem custo de API!)
response2 = model.invoke("Qual é a capital do Brasil?")
```

Para match fuzzy (paráfrases acertam a mesma entrada de cache), veja o [guia de Cache Semântico](docs/pt-br/guide/semantic-cache.md).

### Com Fallbacks Inteligentes

Falhas transientes (`429`, `502`, `503`, `504`, timeouts) são roteadas para uma cadeia curada de fallbacks Sabiá. Erros não-transientes (`401`, `403`, `422`) propagam sem alteração.

```python
from langchain_maritaca import ChatMaritaca

modelo = ChatMaritaca(model="sabia-4", temperature=0.3).with_smart_fallbacks()
resposta = modelo.invoke("Explique o artigo 5 da Constituição brasileira.")
```

Veja o [guia de Fallbacks Inteligentes](docs/pt-br/guide/fallbacks.md) para a cadeia padrão completa e customização.

### Com Callbacks para Observabilidade

```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback, LatencyTrackingCallback

# Criar callbacks para monitoramento
cost_cb = CostTrackingCallback()
latency_cb = LatencyTrackingCallback()

model = ChatMaritaca(callbacks=[cost_cb, latency_cb])

# Fazer algumas chamadas
model.invoke("Olá!")
model.invoke("Como você está?")

# Verificar métricas
print(f"Custo total: ${cost_cb.total_cost:.6f}")
print(f"Tokens totais: {cost_cb.total_tokens}")
print(f"Latência média: {latency_cb.average_latency:.2f}s")
print(f"P95: {latency_cb.p95_latency:.2f}s")
```

### Contagem de Tokens e Estimativa de Custos

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca(model="sabia-4")

# Contar tokens no texto
tokens = model.get_num_tokens("Olá, como você está?")
print(f"Tokens: {tokens}")

# Estimar custo antes de fazer uma requisição
messages = [HumanMessage(content="Me conte sobre o Brasil")]
estimate = model.estimate_cost(messages, max_output_tokens=1000)
print(f"Custo estimado: ${estimate['total_cost']:.6f}")
```

> **Dica**: Instale com `pip install langchain-maritaca[tokenizer]` para contagem precisa de tokens usando tiktoken.

## Por que Maritaca AI?

Os modelos da Maritaca AI são especificamente treinados para Português Brasileiro, oferecendo:

- **Compreensão Nativa do Português**: Melhor entendimento de expressões idiomáticas, gírias e contexto cultural brasileiro
- **Treinamento com Dados Locais**: Treinado em fontes diversas de dados em Português Brasileiro
- **Custo-Benefício**: Preços competitivos para tarefas em português
- **Baixa Latência**: Servidores localizados no Brasil para respostas mais rápidas

## Usado em Produção

**[Cidadão.AI](https://cidadao-ai-frontend.vercel.app/pt)** - Plataforma brasileira de transparência governamental alimentada por agentes de IA, processando mais de 331K requisições/mês.

- Frontend: [github.com/anderson-ufrj/cidadao.ai-frontend](https://github.com/anderson-ufrj/cidadao.ai-frontend)
- Backend: [github.com/anderson-ufrj/cidadao.ai-backend](https://github.com/anderson-ufrj/cidadao.ai-backend)

> *Usando este pacote em produção? [Abra uma issue](https://github.com/anderson-ufrj/langchain-maritaca/issues) para ser destacado!*

## Referência da API

### ChatMaritaca

Classe principal para interagir com os modelos da Maritaca AI.

**Parâmetros:**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model` | str | `"sabia-4"` | Nome do modelo a usar |
| `temperature` | float | `0.7` | Temperatura de amostragem (0.0-2.0) |
| `max_tokens` | int | None | Máximo de tokens a gerar |
| `top_p` | float | `0.9` | Parâmetro top-p de amostragem |
| `api_key` | str | None | Chave de API da Maritaca (ou use var de ambiente) |
| `base_url` | str | `"https://chat.maritaca.ai/api"` | URL base da API |
| `timeout` | float | `60.0` | Timeout da requisição em segundos |
| `max_retries` | int | `2` | Máximo de tentativas de retry |
| `retry_if_rate_limited` | bool | `True` | Auto-retry em rate limit (HTTP 429) |
| `retry_delay` | float | `1.0` | Delay inicial entre retries (segundos) |
| `retry_max_delay` | float | `60.0` | Delay máximo entre retries (segundos) |
| `retry_multiplier` | float | `2.0` | Multiplicador para backoff exponencial |
| `streaming` | bool | `False` | Habilitar respostas em streaming |

### DeepInfraEmbeddings

Classe para gerar embeddings usando DeepInfra (recomendado pela Maritaca AI).

**Parâmetros:**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model` | str | `"intfloat/multilingual-e5-large"` | Modelo de embeddings |
| `api_key` | str | None | Chave de API DeepInfra (ou use var de ambiente) |
| `batch_size` | int | `32` | Tamanho do lote para processamento |

## Desenvolvimento

### Configuração

```bash
# Clone o repositório
git clone https://github.com/anderson-ufrj/langchain-maritaca.git
cd langchain-maritaca

# Instale as dependências
pip install -e ".[dev]"

# Execute os testes
pytest

# Execute o linting
ruff check .
ruff format .

# Execute a verificação de tipos
mypy langchain_maritaca
```

### Executando Testes

```bash
# Apenas testes unitários
pytest tests/unit_tests/

# Testes de integração (requer MARITACA_API_KEY)
pytest tests/integration_tests/

# Com cobertura
pytest --cov=langchain_maritaca --cov-report=html
```

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para enviar um Pull Request.

1. Faça um fork do repositório
2. Crie sua branch de feature (`git checkout -b feature/feature-incrivel`)
3. Commit suas alterações (`git commit -m 'feat: adiciona feature incrível'`)
4. Push para a branch (`git push origin feature/feature-incrivel`)
5. Abra um Pull Request

## Changelog

Veja [CHANGELOG.md](CHANGELOG.md) para a lista de alterações.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Projetos Relacionados

- [LangChain](https://github.com/langchain-ai/langchain) - Construindo aplicações com LLMs através de composabilidade
- [Maritaca AI](https://www.maritaca.ai/) - Modelos de linguagem para Português Brasileiro
