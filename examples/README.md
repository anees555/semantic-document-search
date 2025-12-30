# Examples

Demo scripts and usage examples for the Semantic Document Search project.

## Scripts

### `test_pdf_chunking.py`
- Tests PDF processing with Grobid
- Demonstrates chunk creation and analysis
- Shows configuration options

### `grobid_example.py`
- Complete Grobid demonstration
- Processes documents with structure awareness
- Displays detailed statistics

### `setup_grobid.py`
- Helper script for Grobid server setup
- Docker integration
- Server status checking

## Usage

Make sure you're in the project root directory and run:

```bash
# Test PDF chunking
python examples/test_pdf_chunking.py

# Run full Grobid demo
python examples/grobid_example.py

# Setup Grobid server
python examples/setup_grobid.py
```

## Requirements

- Grobid server running on port 8070
- PDF files in the `documents/` directory (optional)
- Virtual environment activated

## Quick Start

1. Start Grobid server:
   ```bash
   docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0
   ```

2. Add PDF files to `documents/` folder

3. Run any example script to see the processing in action