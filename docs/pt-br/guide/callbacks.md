# Callbacks para Observabilidade

langchain-maritaca fornece handlers de callback para monitoramento e observabilidade das suas interações com LLM. Esses callbacks ajudam a rastrear custos, medir latência e monitorar performance de streaming.

## Callbacks Disponíveis

| Callback | Propósito |
|----------|-----------|
| `CostTrackingCallback` | Rastrear uso de tokens e estimar custos da API |
| `LatencyTrackingCallback` | Medir tempos de resposta com estatísticas de percentil |
| `TokenStreamingCallback` | Monitorar taxas de tokens em streaming |
| `CombinedCallback` | Rastreamento combinado de custo e latência |

## Rastreamento de Custos

Rastreie o uso de tokens e estime custos baseados nos preços da Maritaca AI:

```python
from langchain_maritaca import ChatMaritaca, CostTrackingCallback

# Criar callback
cost_callback = CostTrackingCallback()

# Usar com o modelo
model = ChatMaritaca(callbacks=[cost_callback])

# Fazer algumas chamadas
model.invoke("Olá, como você está?")
model.invoke("Me conte sobre o Brasil")

# Verificar custos
print(f"Custo total: ${cost_callback.total_cost:.6f}")
print(f"Tokens totais: {cost_callback.total_tokens}")
print(f"Tokens de entrada: {cost_callback.total_input_tokens}")
print(f"Tokens de saída: {cost_callback.total_output_tokens}")
print(f"Número de chamadas: {cost_callback.call_count}")

# Obter resumo detalhado
summary = cost_callback.get_summary()
print(f"Custo médio por chamada: ${summary['average_cost_per_call']:.6f}")

# Resetar para nova sessão de rastreamento
cost_callback.reset()
```

### Informações de Preços

O callback usa preços estimados baseados nos preços públicos da Maritaca AI:

| Modelo | Entrada (por 1M tokens) | Saída (por 1M tokens) |
|--------|-------------------------|----------------------|
| sabia-3.1 | $0.50 | $1.50 |
| sabiazinho-3.1 | $0.10 | $0.30 |

> **Nota**: Os preços são estimativas e podem mudar. Consulte [Maritaca AI](https://www.maritaca.ai/) para preços atualizados.

## Rastreamento de Latência

Monitore tempos de resposta da API com análise estatística:

```python
from langchain_maritaca import ChatMaritaca, LatencyTrackingCallback

# Criar callback
latency_callback = LatencyTrackingCallback()

# Usar com o modelo
model = ChatMaritaca(callbacks=[latency_callback])

# Fazer múltiplas chamadas
for _ in range(10):
    model.invoke("Resposta rápida por favor")

# Verificar estatísticas de latência
print(f"Latência média: {latency_callback.average_latency:.2f}s")
print(f"Latência mínima: {latency_callback.min_latency:.2f}s")
print(f"Latência máxima: {latency_callback.max_latency:.2f}s")
print(f"P50 (mediana): {latency_callback.p50_latency:.2f}s")
print(f"P95: {latency_callback.p95_latency:.2f}s")
print(f"P99: {latency_callback.p99_latency:.2f}s")

# Obter todas as latências
latencies = latency_callback.latencies
print(f"Todas as latências: {latencies}")

# Obter resumo
summary = latency_callback.get_summary()
```

### Tratamento de Erros

O callback de latência também rastreia latência para chamadas com falha:

```python
try:
    model.invoke("Isso pode falhar")
except Exception:
    pass

# Latência é rastreada mesmo em caso de erro
print(f"Chamadas rastreadas: {latency_callback.call_count}")
```

## Streaming de Tokens

Monitore taxas de tokens em streaming:

```python
from langchain_maritaca import ChatMaritaca, TokenStreamingCallback

# Criar callback
streaming_callback = TokenStreamingCallback()

# Usar com modelo em streaming
model = ChatMaritaca(streaming=True, callbacks=[streaming_callback])

# Fazer streaming de uma resposta
for chunk in model.stream("Me conte uma história"):
    print(chunk.content, end="", flush=True)

print()  # Nova linha

# Verificar estatísticas de streaming
print(f"Tokens transmitidos: {streaming_callback.token_count}")
print(f"Tokens por segundo: {streaming_callback.tokens_per_second:.1f}")
print(f"Tempo decorrido: {streaming_callback.elapsed_time:.2f}s")

# Obter todos os tokens
tokens = streaming_callback.tokens
```

## Rastreamento Combinado

Use `CombinedCallback` para rastrear custo e latência:

```python
from langchain_maritaca import ChatMaritaca, CombinedCallback

# Criar callback combinado
callback = CombinedCallback()

# Usar com o modelo
model = ChatMaritaca(callbacks=[callback])

# Fazer chamadas
model.invoke("Olá!")
model.invoke("Como você está?")

# Acessar rastreamento de custos
print(f"Custo total: ${callback.cost.total_cost:.6f}")
print(f"Tokens totais: {callback.cost.total_tokens}")

# Acessar rastreamento de latência
print(f"Latência média: {callback.latency.average_latency:.2f}s")
print(f"P95: {callback.latency.p95_latency:.2f}s")

# Obter resumo combinado
summary = callback.get_summary()
print(f"Resumo de custos: {summary['cost']}")
print(f"Resumo de latência: {summary['latency']}")

# Resetar ambos
callback.reset()
```

## Usando Múltiplos Callbacks

Você pode usar múltiplos callbacks simultaneamente:

```python
from langchain_maritaca import (
    ChatMaritaca,
    CostTrackingCallback,
    LatencyTrackingCallback,
    TokenStreamingCallback,
)

# Criar múltiplos callbacks
cost_cb = CostTrackingCallback()
latency_cb = LatencyTrackingCallback()
streaming_cb = TokenStreamingCallback()

# Usar todos os callbacks
model = ChatMaritaca(
    streaming=True,
    callbacks=[cost_cb, latency_cb, streaming_cb]
)

# Fazer streaming de uma resposta
for chunk in model.stream("Me conte sobre Python"):
    print(chunk.content, end="", flush=True)

print()

# Verificar todas as métricas
print(f"Custo: ${cost_cb.total_cost:.6f}")
print(f"Latência: {latency_cb.average_latency:.2f}s")
print(f"Taxa de streaming: {streaming_cb.tokens_per_second:.1f} tokens/s")
```

## Integração com Ferramentas de Monitoramento

### Exemplo com Prometheus

```python
from prometheus_client import Counter, Histogram, start_http_server
from langchain_maritaca import ChatMaritaca, CombinedCallback

# Criar métricas do Prometheus
llm_cost_total = Counter('llm_cost_usd_total', 'Custo total do LLM em USD')
llm_latency = Histogram('llm_latency_seconds', 'Latência das chamadas LLM')
llm_tokens_total = Counter('llm_tokens_total', 'Total de tokens usados', ['type'])

class PrometheusCallback(CombinedCallback):
    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)

        # Atualizar métricas do Prometheus
        llm_cost_total.inc(self.cost._costs_by_call[-1]['cost'])
        llm_latency.observe(self.latency.latencies[-1])

        last_call = self.cost._costs_by_call[-1]
        llm_tokens_total.labels(type='input').inc(last_call['input_tokens'])
        llm_tokens_total.labels(type='output').inc(last_call['output_tokens'])

# Iniciar servidor Prometheus
start_http_server(8000)

# Usar callback customizado
callback = PrometheusCallback()
model = ChatMaritaca(callbacks=[callback])
```

### Exemplo com Logging

```python
import logging
from langchain_maritaca import ChatMaritaca, CombinedCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingCallback(CombinedCallback):
    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)

        last_cost = self.cost._costs_by_call[-1]
        last_latency = self.latency.latencies[-1]

        logger.info(
            f"Chamada LLM concluída: "
            f"modelo={last_cost['model']}, "
            f"tokens={last_cost['total_tokens']}, "
            f"custo=${last_cost['cost']:.6f}, "
            f"latência={last_latency:.2f}s"
        )

callback = LoggingCallback()
model = ChatMaritaca(callbacks=[callback])
```

## Boas Práticas

1. **Resete os callbacks** entre sessões de rastreamento independentes
2. **Use CombinedCallback** quando precisar de métricas de custo e latência
3. **Crie callbacks customizados** estendendo as classes base para integração com sua stack de monitoramento
4. **Monitore em produção** para rastrear custos e detectar regressões de performance
5. **Configure alertas** para picos incomuns de custo ou aumento de latência
