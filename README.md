# Jupyter Notebook to PDF Converter

An AI-powered agentic tool that converts Jupyter Notebooks (.ipynb) to PDF format using an interactive agent interface.

## Features

- Convert local `.ipynb` files or fetch from URLs
- AI-powered conversion using Ollama and LangChain
- Interactive CLI with rich formatting
- Automatic file deduplication (never overwrites existing files)
- Downloads stored in `/notebooks` folder
- PDFs saved in `/pdf` folder

## Installation

```bash
# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install Playwright (required for PDF export)
playwright install chromium
```

## Configuration

Create a `.env` file with the following variables:

```env
model=minimax-m2.5:cloud
temperature=0.7
```

- `model`: Ollama model to use (default: minimax-m2.5:cloud)
- `temperature`: LLM temperature setting (default: 0.7)

Make sure Ollama is running:

```bash
ollama serve
```

## Usage

Run the converter:

```bash
# Using uv
uv run python main.py

# Or after activating the environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
python main.py
```

Enter a local file path or URL when prompted. Type `exit` or `quit` to stop.

## Project Structure

```
.
├── main.py           # Main application entry point
├── notebooks/        # Downloaded notebooks storage
├── pdf/              # Converted PDF output
├── pyproject.toml    # Project dependencies
└── .env              # Environment configuration
```

## Dependencies

- langchain & langchain-ollama
- nbconvert (with webpdf support)
- nbformat
- playwright
- rich (for CLI formatting)
- requests
- python-dotenv
