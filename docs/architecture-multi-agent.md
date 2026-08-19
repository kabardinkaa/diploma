# Архитектура multi-agent эксперимента

Схема получена из скомпилированного LangGraph через
`app.get_graph().draw_mermaid()`. Supervisor маршрутизирует запрос к researcher
с RAG-инструментом, затем к writer без инструментов. Post-model guard сохраняет
решение supervisor, если тот вызвал `transfer_to_writer`, и детерминированно
направляет успешный результат researcher в writer только при некорректном stop.
После writer узел `supervisor_final` завершает граф без дополнительного
LLM-синтеза.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	supervisor(supervisor)
	researcher(researcher)
	writer(writer)
	supervisor_final(supervisor_final)
	__end__([<p>__end__</p>]):::last
	__start__ --> supervisor;
	researcher --> supervisor;
	supervisor -.-> __end__;
	supervisor -.-> researcher;
	supervisor -.-> writer;
	writer --> supervisor_final;
	supervisor_final --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
