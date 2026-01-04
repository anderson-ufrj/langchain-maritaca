# Processamento em Lote

Este guia aborda como processar múltiplas requisições de forma eficiente usando os recursos de otimização de lote do ChatMaritaca.

## Visão Geral

Quando você precisa processar muitas requisições, o processamento em lote pode melhorar significativamente a taxa de transferência executando múltiplas requisições simultaneamente, respeitando os limites de taxa da API.

## Processamento Assíncrono em Lote

Use `abatch_with_concurrency()` para processamento paralelo eficiente:

```python
import asyncio
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca()

# Múltiplas perguntas para processar
inputs = [
    [HumanMessage(content="Qual é a capital do Brasil?")],
    [HumanMessage(content="Qual é a capital da Argentina?")],
    [HumanMessage(content="Qual é a capital do Chile?")],
    [HumanMessage(content="Qual é a capital do Peru?")],
    [HumanMessage(content="Qual é a capital da Colômbia?")],
]

# Processar com máximo de 3 requisições simultâneas
resultados = asyncio.run(
    model.abatch_with_concurrency(inputs, max_concurrency=3)
)

for resultado in resultados:
    print(resultado.generations[0].message.content)
```

### Controle de Concorrência

O parâmetro `max_concurrency` controla quantas requisições rodam simultaneamente:

```python
# Conservador: 2 requisições simultâneas
resultados = await model.abatch_with_concurrency(inputs, max_concurrency=2)

# Agressivo: 10 requisições simultâneas (pode atingir limites de taxa)
resultados = await model.abatch_with_concurrency(inputs, max_concurrency=10)
```

Valores recomendados:
- **2-3**: Seguro para a maioria dos casos
- **5**: Bom equilíbrio entre velocidade e confiabilidade
- **10+**: Apenas se você tiver limites de taxa altos

### Tratamento de Erros

Use `return_exceptions=True` para tratar erros graciosamente:

```python
resultados = await model.abatch_with_concurrency(
    inputs,
    max_concurrency=5,
    return_exceptions=True,  # Não falha em erros individuais
)

for i, resultado in enumerate(resultados):
    if isinstance(resultado, Exception):
        print(f"Requisição {i} falhou: {resultado}")
    else:
        print(f"Requisição {i}: {resultado.generations[0].message.content}")
```

## Lote com Acompanhamento de Progresso

Use `batch_with_progress()` para processamento síncrono com atualizações de progresso:

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage

model = ChatMaritaca()

def no_progresso(concluido, total, resultado):
    print(f"Progresso: {concluido}/{total} ({100*concluido/total:.0f}%)")

inputs = [
    [HumanMessage(content=f"Pergunta {i}")] for i in range(20)
]

resultados = model.batch_with_progress(
    inputs,
    max_concurrency=3,
    callback=no_progresso,
)
```

### Callback de Progresso

O callback recebe:
- `completed`: Número de requisições concluídas
- `total`: Número total de requisições
- `result`: O ChatResult da requisição recém-concluída

```python
def progresso_detalhado(concluido, total, resultado):
    conteudo = resultado.generations[0].message.content[:50]
    print(f"[{concluido}/{total}] Recebido: {conteudo}...")
```

## Estimativa de Custo

Estime custos do lote antes da execução com `abatch_estimate_cost()`:

```python
inputs = [
    [HumanMessage(content="Pergunta curta")],
    [HumanMessage(content="Outra pergunta")],
    [HumanMessage(content="Terceira pergunta")],
]

estimativa = await model.abatch_estimate_cost(
    inputs,
    max_output_tokens=500,  # Saída esperada por requisição
)

print(f"Total de requisições: {estimativa['total_requests']}")
print(f"Tokens de entrada estimados: {estimativa['total_input_tokens']}")
print(f"Tokens de saída estimados: {estimativa['total_output_tokens']}")
print(f"Custo estimado: ${estimativa['total_cost']:.6f}")
```

### Planejamento de Custo

```python
# Verificar custo antes de processar lote grande
if estimativa['total_cost'] > 1.00:  # Limite de orçamento
    print("Aviso: Custo do lote excede $1.00")
    continuar = input("Continuar? (s/n): ")
    if continuar.lower() != 's':
        exit()

resultados = await model.abatch_with_concurrency(inputs)
```

## Boas Práticas

### 1. Escolher Concorrência Apropriada

```python
# Para APIs com limite de taxa
resultados = await model.abatch_with_concurrency(inputs, max_concurrency=2)

# Para APIs de alta capacidade
resultados = await model.abatch_with_concurrency(inputs, max_concurrency=10)
```

### 2. Tratar Falhas Graciosamente

```python
async def lote_seguro(model, inputs):
    resultados = await model.abatch_with_concurrency(
        inputs,
        return_exceptions=True,
    )

    sucesso = []
    falhas = []

    for i, resultado in enumerate(resultados):
        if isinstance(resultado, Exception):
            falhas.append((i, inputs[i], resultado))
        else:
            sucesso.append(resultado)

    # Retentar requisições que falharam
    if falhas:
        retry_inputs = [item[1] for item in falhas]
        retry_resultados = await model.abatch_with_concurrency(
            retry_inputs,
            max_concurrency=1,  # Mais lento, mais confiável
        )
        sucesso.extend(retry_resultados)

    return sucesso
```

### 3. Combinar com Estimativa de Custo

```python
async def lote_com_orcamento(model, inputs, custo_max=1.0):
    estimativa = await model.abatch_estimate_cost(inputs)

    if estimativa['total_cost'] > custo_max:
        raise ValueError(f"Custo do lote ${estimativa['total_cost']:.4f} excede orçamento ${custo_max}")

    return await model.abatch_with_concurrency(inputs)
```

## Referência da API

### `abatch_with_concurrency(inputs, max_concurrency=5, stop=None, return_exceptions=False)`

Processa múltiplas listas de mensagens simultaneamente.

**Parâmetros:**
- `inputs`: Lista de listas de mensagens para processar
- `max_concurrency`: Máximo de requisições simultâneas (padrão: 5)
- `stop`: Sequências de parada opcionais
- `return_exceptions`: Se True, retorna exceções em vez de lançá-las

**Retorna:** Lista de objetos ChatResult

### `batch_with_progress(inputs, max_concurrency=5, stop=None, callback=None)`

Processa múltiplas listas de mensagens com callback de progresso.

**Parâmetros:**
- `inputs`: Lista de listas de mensagens para processar
- `max_concurrency`: Máximo de requisições simultâneas
- `stop`: Sequências de parada opcionais
- `callback`: Função de callback de progresso

**Retorna:** Lista de objetos ChatResult

### `abatch_estimate_cost(inputs, max_output_tokens=1000)`

Estima custo do lote antes da execução.

**Parâmetros:**
- `inputs`: Lista de listas de mensagens para estimar
- `max_output_tokens`: Tokens de saída esperados por requisição

**Retorna:** Dicionário com estimativas de custo
