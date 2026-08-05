import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_graph import custom_graph, prebuilt_graph


DOCS_DIR = Path("docs")


def _write_graph(name: str, graph: object) -> None:
    mermaid = graph.get_graph().draw_mermaid().strip() + "\n"
    (DOCS_DIR / f"agent-graph-{name}.mmd").write_text(mermaid, encoding="utf-8")
    (DOCS_DIR / f"agent-graph-{name}.md").write_text(
        f"# Agent Graph: {name.title()}\n\n```mermaid\n{mermaid}```\n",
        encoding="utf-8",
    )


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    _write_graph("custom", custom_graph)
    _write_graph("prebuilt", prebuilt_graph)


if __name__ == "__main__":
    main()
