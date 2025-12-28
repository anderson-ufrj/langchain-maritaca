# Contagem de Tokens e Estimativa de Custos

langchain-maritaca fornece métodos integrados para contagem de tokens e estimativa de custos antes de fazer chamadas à API. Isso ajuda a gerenciar custos e garantir que suas requisições caibam nos limites de contexto do modelo.

## Métodos de Contagem de Tokens

### get_num_tokens

Conte o número de tokens em uma string de texto:

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca()

# Contar tokens no texto
tokens = model.get_num_tokens("Olá, como você está?")
print(f"Tokens: {tokens}")
```

### get_num_tokens_from_messages

Conte tokens em uma lista de mensagens, incluindo overhead de formatação:

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatMaritaca()

messages = [
    SystemMessage(content="Você é um assistente prestativo."),
    HumanMessage(content="Qual é a capital do Brasil?"),
    AIMessage(content="A capital do Brasil é Brasília."),
    HumanMessage(content="Me conte mais sobre ela."),
]

tokens = model.get_num_tokens_from_messages(messages)
print(f"Total de tokens: {tokens}")
```

## Estimativa de Custos

### estimate_cost

Estime o custo de uma requisição antes de fazê-la:

```python
from langchain_core.messages import HumanMessage

model = ChatMaritaca(model="sabia-3.1")

messages = [
    HumanMessage(content="Me conte uma longa história sobre o Brasil")
]

# Estimar custo com tokens de saída esperados
estimate = model.estimate_cost(messages, max_output_tokens=2000)

print(f"Tokens de entrada: {estimate['input_tokens']}")
print(f"Tokens de saída: {estimate['output_tokens']}")
print(f"Custo de entrada: ${estimate['input_cost']:.6f}")
print(f"Custo de saída: ${estimate['output_cost']:.6f}")
print(f"Custo total: ${estimate['total_cost']:.6f}")
```

### Comparar Custos de Modelos

Compare custos entre diferentes modelos:

```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Me conte sobre a história do Brasil")]

# Comparar sabia vs sabiazinho
sabia = ChatMaritaca(model="sabia-3.1")
sabiazinho = ChatMaritaca(model="sabiazinho-3.1")

sabia_cost = sabia.estimate_cost(messages, max_output_tokens=1000)
sabiazinho_cost = sabiazinho.estimate_cost(messages, max_output_tokens=1000)

print(f"Custo sabia-3.1: ${sabia_cost['total_cost']:.6f}")
print(f"Custo sabiazinho-3.1: ${sabiazinho_cost['total_cost']:.6f}")
print(f"Economia com sabiazinho: {(1 - sabiazinho_cost['total_cost']/sabia_cost['total_cost'])*100:.1f}%")
```

## Referência de Preços

Preços estimados atuais para modelos Maritaca AI:

| Modelo | Entrada (por 1M tokens) | Saída (por 1M tokens) |
|--------|-------------------------|----------------------|
| sabia-3.1 | $0.50 | $1.50 |
| sabiazinho-3.1 | $0.10 | $0.30 |

> **Nota**: Os preços são estimativas e podem mudar. Consulte [Maritaca AI](https://www.maritaca.ai/) para preços atualizados.

## Opções de Tokenizador

Por padrão, a contagem de tokens usa tiktoken se disponível, com fallback baseado em caracteres:

### Com tiktoken (Recomendado)

Instale com o extra tokenizer para contagem precisa:

```bash
pip install langchain-maritaca[tokenizer]
```

Isso usa a codificação `cl100k_base` da OpenAI, que fornece boa precisão para texto em português.

### Sem tiktoken (Fallback)

Se tiktoken não estiver instalado, uma estimativa baseada em caracteres é usada (~4 caracteres por token). Isso é menos preciso mas não requer dependências adicionais.

## Casos de Uso

### Monitoramento de Orçamento

Rastreie custos em múltiplas chamadas:

```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback

# Use callback para custos reais após as chamadas
cost_cb = CostTrackingCallback()
model = ChatMaritaca(callbacks=[cost_cb])

# Ou estime antes das chamadas
messages = [HumanMessage(content="Olá")]
estimate = model.estimate_cost(messages)
print(f"Estimado: ${estimate['total_cost']:.6f}")

model.invoke(messages)
print(f"Real: ${cost_cb.total_cost:.6f}")
```

### Gerenciamento de Janela de Contexto

Garanta que as mensagens caibam nos limites de contexto:

```python
MAX_CONTEXT = 8192  # Exemplo de janela de contexto

model = ChatMaritaca()
messages = [...]  # Seu histórico de conversa

tokens = model.get_num_tokens_from_messages(messages)

if tokens > MAX_CONTEXT - 1000:  # Deixe espaço para resposta
    print("Aviso: Aproximando do limite de contexto!")
    # Truncar ou resumir mensagens antigas
```

### Roteamento Consciente de Custos

Roteie para diferentes modelos baseado em custo:

```python
def get_model_for_task(messages, complexity="simple"):
    """Escolha modelo baseado na complexidade e custo."""
    sabia = ChatMaritaca(model="sabia-3.1")
    sabiazinho = ChatMaritaca(model="sabiazinho-3.1")

    if complexity == "complex":
        return sabia

    # Para tarefas simples, use o modelo mais barato
    estimate = sabiazinho.estimate_cost(messages)
    if estimate["total_cost"] < 0.001:  # Menos de $0.001
        return sabiazinho

    return sabia
```

## Boas Práticas

1. **Estime antes de operações longas**: Para processamento em lote, estime o custo total antes de começar
2. **Use sabiazinho para tarefas simples**: É 5x mais barato para entrada e saída
3. **Monitore real vs estimado**: Use `CostTrackingCallback` para verificar estimativas
4. **Instale tiktoken**: Para contagem mais precisa, especialmente com texto em português
5. **Cache consultas repetidas**: Use cache do LangChain para evitar chamadas redundantes
