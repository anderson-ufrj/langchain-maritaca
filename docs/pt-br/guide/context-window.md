# Gerenciamento de Janela de Contexto

Este guia aborda como gerenciar os limites da janela de contexto ao trabalhar com o ChatMaritaca.

## Visão Geral

Modelos de linguagem têm um número máximo de tokens que podem processar em uma única requisição (a "janela de contexto"). Quando sua conversa excede esse limite, a API retorna um erro. O ChatMaritaca fornece ferramentas para ajudar a gerenciar essa limitação.

## Limites de Contexto por Modelo

Cada modelo Maritaca tem tamanhos diferentes de janela de contexto:

| Modelo | Limite de Contexto | Melhor Para |
|--------|-------------------|-------------|
| sabia-3.1 | 32.768 tokens | Conversas longas, análise de documentos |
| sabiazinho-3.1 | 8.192 tokens | Perguntas rápidas, tarefas simples |

## Verificando Uso do Contexto

Use `check_context_usage()` para monitorar quanto da sua janela de contexto está sendo utilizada:

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatMaritaca()

messages = [
    SystemMessage(content="Você é um assistente prestativo."),
    HumanMessage(content="Me conte sobre a história do Brasil..."),
]

usage = model.check_context_usage(messages)
print(f"Tokens usados: {usage['tokens']}")
print(f"Limite de contexto: {usage['limit']}")
print(f"Uso: {usage['usage_percent']:.1%}")
```

### Avisos Automáticos

Por padrão, o ChatMaritaca emite avisos quando:
- O uso do contexto excede 90% do limite (configurável via `context_warning_threshold`)
- O limite de contexto é excedido

```python
# Configurar limite de aviso
model = ChatMaritaca(context_warning_threshold=0.8)  # Avisar em 80%
```

## Truncando Mensagens

Quando sua conversa excede o limite de contexto, use `truncate_messages()` para ajustar:

```python
model = ChatMaritaca()

# Conversa longa que excede o limite
conversa_longa = [
    SystemMessage(content="Você é um assistente prestativo."),
    HumanMessage(content="Primeira pergunta"),
    AIMessage(content="Primeira resposta"),
    HumanMessage(content="Segunda pergunta"),
    AIMessage(content="Segunda resposta"),
    # ... muitas mais mensagens ...
    HumanMessage(content="Pergunta mais recente"),
]

# Truncar para caber no contexto
truncada = model.truncate_messages(conversa_longa)

# Agora é seguro usar
resposta = model.invoke(truncada)
```

### Opções de Truncamento

```python
truncada = model.truncate_messages(
    messages,
    max_tokens=4096,       # Limite personalizado de tokens
    preserve_system=True,  # Manter mensagens do sistema (padrão: True)
    preserve_recent=2,     # Manter últimas N mensagens não-sistema (padrão: 1)
)
```

O algoritmo de truncamento:
1. Sempre preserva mensagens do sistema (se `preserve_system=True`)
2. Sempre preserva as mensagens mais recentes (especificado por `preserve_recent`)
3. Remove mensagens antigas começando pela mais antiga

## Limites Personalizados de Contexto

Você pode definir um limite de contexto personalizado menor que o máximo do modelo:

```python
# Limitar a 4096 tokens mesmo que o modelo suporte mais
model = ChatMaritaca(max_context_tokens=4096)
```

Isso é útil quando:
- Você quer deixar espaço para a saída do modelo
- Está construindo um chatbot com limites de memória consistentes
- Quer controlar custos limitando o tamanho da entrada

## Boas Práticas

### 1. Monitorar Antes de Conversas Longas

```python
def chat_com_monitoramento(model, messages, entrada_usuario):
    messages.append(HumanMessage(content=entrada_usuario))

    # Verificar antes de enviar
    usage = model.check_context_usage(messages, warn=False)
    if usage['usage_percent'] > 0.8:
        messages = model.truncate_messages(messages, preserve_recent=4)

    resposta = model.invoke(messages)
    messages.append(resposta)
    return resposta
```

### 2. Usar Modelos Apropriados

Para conversas longas, prefira `sabia-3.1` com seu contexto de 32K. Para interações rápidas, `sabiazinho-3.1` é mais econômico.

### 3. Resumir Contexto Antigo

Em vez de truncar, considere resumir mensagens antigas:

```python
def resumir_e_continuar(model, messages):
    if len(messages) > 10:
        # Manter mensagem do sistema
        sistema = messages[0] if isinstance(messages[0], SystemMessage) else None

        # Resumir mensagens antigas
        msgs_antigas = messages[1:-2]  # Tudo exceto sistema e últimas 2
        prompt_resumo = "Resuma esta conversa brevemente: " + str(msgs_antigas)
        resumo = model.invoke([HumanMessage(content=prompt_resumo)])

        # Construir novo contexto
        novas_msgs = []
        if sistema:
            novas_msgs.append(sistema)
        novas_msgs.append(SystemMessage(content=f"Contexto anterior: {resumo.content}"))
        novas_msgs.extend(messages[-2:])  # Manter últimas 2 mensagens

        return novas_msgs
    return messages
```

## Referência da API

### `get_context_limit()`

Retorna o limite da janela de contexto para o modelo atual.

```python
limite = model.get_context_limit()
```

### `check_context_usage(messages, warn=True)`

Retorna informações de uso do contexto e opcionalmente emite avisos.

**Retorna:**
- `tokens`: Contagem estimada de tokens
- `limit`: Limite da janela de contexto
- `usage_percent`: Porcentagem do contexto usado
- `exceeds_limit`: Se o contexto foi excedido
- `exceeds_threshold`: Se o limite de aviso foi excedido

### `truncate_messages(messages, max_tokens=None, preserve_system=True, preserve_recent=1)`

Trunca mensagens para caber na janela de contexto.

**Parâmetros:**
- `messages`: Lista de mensagens para truncar
- `max_tokens`: Máximo de tokens (padrão: limite do modelo)
- `preserve_system`: Se deve manter mensagens do sistema
- `preserve_recent`: Número de mensagens recentes para sempre manter

**Retorna:** Lista truncada de mensagens
