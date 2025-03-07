# ClaudeCodingCompanion.py

A Python tool that scans directory structures, analyzes contained files, and generates a consolidated text file capturing the current state of a project. The output can be shared with AI assistants for better analysis and understanding of your codebase.

## Requirements

- Python 3.6 or higher
- Tkinter (included in standard Python installations)

## Installation

No installation required. The script is a self-contained Python file that can be run directly.

## Usage

### Command Line Interface (CLI)

#### Basic Usage

```bash
python3 claudecodingcompanion.py [directory_path] [output_file]
```

#### Parameters

- `directory_path`: The directory you want to scan
- `output_file`: Where to save the scan results (optional)

#### Examples

Scan a directory and let the tool auto-generate an output filename:

```bash
python3 claudecodingcompanion.py /path/to/your/project
```

Scan a directory and specify an output file:

```bash
python3 claudecodingcompanion.py /path/to/your/project output_file.txt
```

Enable verbose logging:

```bash
python3 claudecodingcompanion.py /path/to/your/project --verbose
```

Use a custom configuration file:

```bash
python3 claudecodingcompanion.py /path/to/your/project --config my_config.json
```

### Graphical User Interface (GUI)

#### Launching the GUI

There are two ways to launch the GUI:

Without any parameters:

```bash
python3 claudecodingcompanion.py
```

With the `--gui` flag:

```bash
python3 claudecodingcompanion.py --gui
```

## Configuration

The tool uses a configuration file stored at `~/.file_scanner_config.json`. Key settings include:

- File extensions to include/exclude
- Directories to exclude
- Maximum file size limit
- Output format options
- Performance settings like parallel workers

The configuration file is automatically created with default values on first run.

## Output Format

The tool generates a text file with XML-formatted content, including:

- Project metadata (file count, directory structure)
- Table of contents
- File relationships
- Full content of each file

## Troubleshooting

- **GUI Not Showing Buttons**: Try resizing the window
- **Scan Fails or Hangs**: Try reducing the "Number of Parallel Workers" in Settings
- **Output File Too Large**: Exclude more directories or file types
- **File Encoding Issues**: Some files might fail to read due to encoding issues
