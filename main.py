import os
import requests
import nbformat
from nbconvert import WebPDFExporter
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
import time

# ─────────────────────────────────────────────
load_dotenv()
console = Console()

MODEL_NAME  = os.getenv("model", "minimax-m2.5:cloud")
TEMPERATURE = float(os.getenv("temperature", "0.7"))


# ── Shared helper ─────────────────────────────────────────────────────────────
def unique_path(folder: Path, filename: str) -> Path:
    """
    Return folder/filename.
    If that already exists, return folder/stem-1.ext, folder/stem-2.ext … etc.
    Never overwrites an existing file.
    """
    stem      = Path(filename).stem
    suffix    = Path(filename).suffix        # includes the dot, e.g. ".ipynb"
    candidate = folder / filename
    counter   = 1
    while candidate.exists():
        candidate = folder / f"{stem}-{counter}{suffix}"
        counter  += 1
    return candidate


# ─────────────────────────────────────────────
def print_banner():
    banner = Text()
    banner.append("  ╔══════════════════════════════════════╗\n", style="bold cyan")
    banner.append("  ║   ", style="bold cyan")
    banner.append("📓  ipynb  →  PDF  Converter", style="bold white")
    banner.append("   ║\n", style="bold cyan")
    banner.append("  ║   ", style="bold cyan")
    banner.append(f"  Model : {MODEL_NAME:<24}", style="dim white")
    banner.append("║\n", style="bold cyan")
    banner.append("  ║   ", style="bold cyan")
    banner.append(f"  Temp  : {TEMPERATURE:<24}", style="dim white")
    banner.append("║\n", style="bold cyan")
    banner.append("  ╚══════════════════════════════════════╝", style="bold cyan")
    console.print(banner)
    console.print()


# ─────────────────────────────────────────────
def build_llm_with_spinner():
    with Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[bold cyan]Loading model[/bold cyan] [dim]{task.description}[/dim]"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"{MODEL_NAME} …", total=None)
        llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
        time.sleep(0.4)
    console.print(f"  [bold green]✔[/bold green]  Model [cyan]{MODEL_NAME}[/cyan] ready\n")
    return llm


# ─────────────────────────────────────────────
@tool
def ipynb_to_pdf(source: str) -> str:
    """
    Converts a Jupyter Notebook (.ipynb) from a local file or URL into a PDF.
    Saves:
    - downloaded notebooks in /notebooks  (never overwrites — appends -1, -2 …)
    - converted PDFs       in /pdf        (never overwrites — appends -1, -2 …)
    """
    try:
        os.makedirs("notebooks", exist_ok=True)
        os.makedirs("pdf",       exist_ok=True)

        with Progress(
            SpinnerColumn(spinner_name="aesthetic", style="bold magenta"),
            TextColumn("[bold white]{task.description}"),
            BarColumn(bar_width=30, style="magenta", complete_style="bold green"),
            TextColumn("[dim]{task.percentage:>3.0f}%[/dim]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            task = progress.add_task("Starting …", total=100)

            # ── Step 1 : dirs ──────────────────────────────
            progress.update(task, description="📁  Preparing directories …", advance=10)
            time.sleep(0.3)

            # ── Step 2 : fetch / read ──────────────────────
            progress.update(task, description="📥  Fetching notebook …", advance=15)

            if source.startswith("http://") or source.startswith("https://"):
                response = requests.get(source, timeout=30)
                if response.status_code != 200:
                    return f"Error: HTTP {response.status_code} – could not fetch URL."

                raw_name   = Path(source).name
                file_name  = raw_name if raw_name.endswith(".ipynb") else "downloaded_notebook.ipynb"
                save_path  = unique_path(Path("notebooks"), file_name)  # ← dedup
                save_path.write_text(response.text, encoding="utf-8")
                nb_content = nbformat.reads(response.text, as_version=4)

            else:
                src = Path(source)
                if not src.exists():
                    return f"Error: File '{source}' not found."

                file_name  = src.name
                save_path  = unique_path(Path("notebooks"), file_name)  # ← dedup
                save_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                nb_content = nbformat.read(src.open("r", encoding="utf-8"), as_version=4)

            progress.advance(task, 15)

            # ── Step 3 : parse ────────────────────────────
            progress.update(task, description="🔍  Parsing notebook …", advance=10)
            time.sleep(0.4)

            # ── Step 4 : convert ──────────────────────────
            progress.update(task, description="🔄  Converting to PDF …", advance=10)
            pdf_exporter = WebPDFExporter()
            pdf_data, _  = pdf_exporter.from_notebook_node(nb_content)
            progress.advance(task, 30)

            # ── Step 5 : save ─────────────────────────────
            progress.update(task, description="💾  Saving PDF …", advance=5)

            # Derive PDF name from the saved notebook path (already deduplicated),
            # so notebook and PDF always share the same numeric suffix.
            # e.g.  notebooks/my_notebook-2.ipynb  →  pdf/my_notebook-2.pdf
            pdf_name = save_path.stem + ".pdf"
            pdf_path = unique_path(Path("pdf"), pdf_name)   # extra safety dedup
            pdf_path.write_bytes(pdf_data)

            progress.update(task, description="[bold green]✔  Done!", advance=5)
            time.sleep(0.3)

        # ── Summary table ──────────────────────────────────
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="bold white")
        table.add_row("📥 Notebook saved", str(save_path))
        table.add_row("📄 PDF saved",      str(pdf_path))
        table.add_row("📦 PDF size",       f"{pdf_path.stat().st_size / 1024:.1f} KB")

        console.print()
        console.print(Panel(table, title="[bold green]Conversion Successful[/bold green]",
                            border_style="green", padding=(1, 2)))

        return f"✅ PDF saved to: {pdf_path}"

    except Exception as e:
        console.print_exception()
        return f"❌ Conversion failed: {e}"


# ─────────────────────────────────────────────
def build_agent(llm):
    return create_agent(
        llm,
        tools=[ipynb_to_pdf],
        system_prompt=(
            "You are a helpful AI agent specialising in converting Jupyter Notebooks "
            "to PDF. Use the ipynb_to_pdf tool whenever the user provides a file path "
            "or URL. Always confirm success or surface errors clearly."
        ),
    )


# ─────────────────────────────────────────────
def run_interactive_loop(agent):
    console.print(Rule("[bold cyan]Session started[/bold cyan]", style="cyan"))
    console.print("  Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to stop.\n")

    while True:
        raw = Prompt.ask("\n  [bold cyan]📂  Enter .ipynb path or URL[/bold cyan]").strip()

        if raw.lower() in ("exit", "quit", "q"):
            console.print("\n  [bold yellow]👋  Goodbye![/bold yellow]\n")
            break

        if not raw:
            console.print("  [dim]No input provided — try again.[/dim]")
            continue

        console.print()
        console.print(Rule("[dim]Agent thinking …[/dim]", style="dim magenta"))

        with Progress(
            SpinnerColumn(spinner_name="point", style="bold magenta"),
            TextColumn("[bold magenta]{task.description}"),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task("Invoking agent …", total=None)
            response = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": f"Convert this ipynb file to PDF: {raw}"
                }]
            })

        final = response["messages"][-1].content
        console.print()
        console.print(Panel(
            Text(final, style="white"),
            title="[bold cyan]Agent Response[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))

        console.print()
        again = Prompt.ask(
            "  [dim]Convert another file?[/dim]",
            choices=["y", "n"],
            default="y",
            show_choices=True,
        )
        if again.lower() == "n":
            console.print("\n  [bold yellow]👋  Goodbye![/bold yellow]\n")
            break


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    llm   = build_llm_with_spinner()
    agent = build_agent(llm)
    run_interactive_loop(agent)