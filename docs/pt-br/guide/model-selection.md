# Seleção de Modelo

Este guia ajuda você a escolher o melhor modelo Maritaca para seu caso de uso.

## Modelos Disponíveis

O ChatMaritaca fornece utilitários para descobrir e selecionar o modelo ideal:

```python
from langchain_maritaca import ChatMaritaca

# Listar todos os modelos disponíveis com especificações
modelos = ChatMaritaca.list_available_models()

for nome, spec in modelos.items():
    print(f"{nome}:")
    print(f"  Contexto: {spec['context_limit']:,} tokens")
    print(f"  Custo entrada: ${spec['input_cost_per_1m']}/1M tokens")
    print(f"  Custo saída: ${spec['output_cost_per_1m']}/1M tokens")
    print(f"  Velocidade: {spec['speed']}")
    print(f"  Complexidade: {spec['complexity']}")
```

## Comparação de Modelos

| Modelo | Contexto | Velocidade | Complexidade | Custo (Entrada/Saída por 1M) |
|--------|----------|------------|--------------|------------------------------|
| sabia-3.1 | 32.768 | Média | Alta | $0,50 / $1,50 |
| sabiazinho-3.1 | 8.192 | Rápida | Média | $0,10 / $0,30 |

### Quando Usar sabia-3.1

- Tarefas de raciocínio complexo
- Análise de documentos longos
- Resolução de problemas em múltiplas etapas
- Quando qualidade de saída é crítica
- Conversas longas que precisam de contexto de 32K

### Quando Usar sabiazinho-3.1

- Perguntas e respostas simples
- Quando respostas rápidas são necessárias
- Aplicações sensíveis a custo
- Tarefas simples de alto volume
- Conversas com menos de 8K tokens

## Recomendação Automática de Modelo

Use `recommend_model()` para obter sugestões inteligentes:

```python
# Tarefa simples, otimizar para custo
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    priority="cost",
)
print(f"Recomendado: {rec['model']}")
print(f"Razão: {rec['reason']}")

# Usar o modelo recomendado
model = ChatMaritaca(model=rec['model'])
```

### Complexidade da Tarefa

- **simple**: Perguntas rápidas, texto simples, resumos
- **medium**: Conversas padrão, raciocínio moderado
- **complex**: Raciocínio complexo, análise, geração de texto longo

### Prioridade

- **cost**: Minimizar custos da API
- **speed**: Maximizar velocidade de resposta
- **quality**: Maximizar qualidade da saída

### Consideração do Tamanho da Entrada

```python
# Para entradas longas, recomendar modelos com contexto suficiente
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    input_length=10000,  # 10K tokens necessários
)

# Vai recomendar sabia-3.1 devido aos requisitos de contexto
print(rec['model'])  # 'sabia-3.1'
```

## Exemplos Práticos

### Otimização de Custo

```python
# Para um chatbot lidando com consultas simples
rec = ChatMaritaca.recommend_model(
    task_complexity="simple",
    priority="cost",
)

model = ChatMaritaca(model=rec['model'])  # sabiazinho-3.1
```

### Otimização de Qualidade

```python
# Para análise de documentos que requer compreensão profunda
rec = ChatMaritaca.recommend_model(
    task_complexity="complex",
    priority="quality",
)

model = ChatMaritaca(model=rec['model'])  # sabia-3.1
```

### Otimização de Velocidade

```python
# Para aplicações em tempo real que precisam de respostas rápidas
rec = ChatMaritaca.recommend_model(
    task_complexity="medium",
    priority="speed",
)

model = ChatMaritaca(model=rec['model'])  # sabiazinho-3.1
```

### Seleção Dinâmica

```python
def selecionar_modelo_para_tarefa(descricao_tarefa, texto_entrada):
    # Estimar tamanho da entrada
    tamanho_entrada = len(texto_entrada) // 4  # Estimativa aproximada de tokens

    # Determinar complexidade
    if any(palavra in descricao_tarefa.lower() for palavra in ['analisar', 'complexo', 'detalhado']):
        complexidade = 'complex'
    elif any(palavra in descricao_tarefa.lower() for palavra in ['simples', 'rápido', 'breve']):
        complexidade = 'simple'
    else:
        complexidade = 'medium'

    rec = ChatMaritaca.recommend_model(
        task_complexity=complexidade,
        input_length=tamanho_entrada,
        priority='quality',
    )

    return ChatMaritaca(model=rec['model'])
```

## Alternativas

A recomendação inclui alternativas quando disponíveis:

```python
rec = ChatMaritaca.recommend_model(
    task_complexity="medium",
    priority="quality",
)

print(f"Recomendação principal: {rec['model']}")
print("Alternativas:")
for alt in rec['alternatives']:
    print(f"  - {alt['model']}: {alt['specs']['description']}")
```

## Referência da API

### `list_available_models()`

Retorna dicionário de todos os modelos disponíveis com suas especificações.

**Retorna:** Dict mapeando nomes de modelos para especificações incluindo:
- `context_limit`: Janela máxima de contexto
- `input_cost_per_1m`: Custo por 1M de tokens de entrada
- `output_cost_per_1m`: Custo por 1M de tokens de saída
- `complexity`: Nível de capacidade do modelo
- `speed`: Velocidade de resposta
- `capabilities`: Lista de recursos suportados
- `description`: Descrição legível

### `recommend_model(task_complexity, input_length, priority)`

Obtém recomendação inteligente de modelo.

**Parâmetros:**
- `task_complexity`: 'simple', 'medium' ou 'complex'
- `input_length`: Contagem estimada de tokens de entrada (opcional)
- `priority`: 'cost', 'speed' ou 'quality'

**Retorna:** Dicionário com:
- `model`: Nome do modelo recomendado
- `reason`: Explicação para a recomendação
- `specs`: Especificações do modelo
- `alternatives`: Lista de outras opções viáveis
