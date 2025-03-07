#!/usr/bin/env python3
"""
  ____ _                 _         ____          _ _             
 / ___| | __ _ _   _  __| | ___   / ___|___   __| (_)_ __   __ _ 
| |   | |/ _` | | | |/ _` |/ _ \ | |   / _ \ / _` | | '_ \ / _` |
| |___| | (_| | |_| | (_| |  __/ | |__| (_) | (_| | | | | | (_| |
 \____|_|\__,_|\__,_|\__,_|\___|  \____\___/ \__,_|_|_| |_|\__, |
 / ___|___  _ __ ___  _ __   __ _ _ __ (_) ___  _ __       |___/ 
| |   / _ \| '_ ` _ \| '_ \ / _` | '_ \| |/ _ \| '_ \            
| |__| (_) | | | | | | |_) | (_| | | | | | (_) | | | |           
 \____\___/|_| |_| |_| .__/ \__,_|_| |_|_|\___/|_| |_|           
                     |_|                                         

This script scans a directory structure, analyzes all contained files,
and generates a consolidated text file containing the content of all
relevant files. This helps in capturing the current state of a project
for sharing with AI assistants.

Usage:
    python claudecodingcompanion.py [directory_path] [output_file]
    python claudecodingcompanion.py --gui

    If no arguments are provided, a GUI will be launched.

Author: Mark Ulett
V 0.1.0
"""

# Standard library imports
import os
import sys
import json
import time
import argparse
import logging
import re
import shutil
import subprocess
import webbrowser
import concurrent.futures  
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union


# Third-party imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mimetypes
import hashlib

class ProjectMetadata:
    """Class to store and manage project metadata."""
    
    def __init__(self, root_directory):
        self.root_directory = os.path.abspath(root_directory)
        self.directories = {}  # Directory structure info
        self.files = {}  # File metadata keyed by relative path
        self.file_relationships = []  # Store relationships between files
        self.file_count = 0
        self.directory_count = 0
        self.file_types = defaultdict(int)  # Count of each file type
        self.scan_date = datetime.now()
        
    def to_dict(self):
        """Convert metadata to a dictionary for serialization."""
        return {
            "root_directory": self.root_directory,
            "directories": self.directories,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "file_types": dict(self.file_types),
            "scan_date": self.scan_date.isoformat()
        }
        
    def to_xml(self):
        """Generate XML representation of project metadata."""
        lines = []
        lines.append("<project>")
        lines.append(f"<name>{os.path.basename(self.root_directory)}</name>")
        lines.append(f"<scan_date>{self.scan_date.isoformat()}</scan_date>")
        lines.append(f"<file_count>{self.file_count}</file_count>")
        lines.append(f"<directory_count>{self.directory_count}</directory_count>")
        
        # Add file type summary
        lines.append("<file_types>")
        for file_type, count in self.file_types.items():
            lines.append(f"<type name=\"{file_type}\" count=\"{count}\"/>")
        lines.append("</file_types>")
        
        # Add directory structure
        lines.append("<directory_structure>")
        self._append_directory_structure(lines, "", self.directories.get("", {}))
        lines.append("</directory_structure>")
        
        lines.append("</project>")
        return "\n".join(lines)
    
    def _append_directory_structure(self, lines, path, directory_info, indent="  "):
        """Helper method to recursively append directory structure as XML."""
        dir_name = os.path.basename(path) or self.root_directory
        lines.append(f"{indent}<directory name=\"{dir_name}\" path=\"{path}\">")
        
        # Add files in this directory
        if "files" in directory_info:
            for file_info in directory_info["files"]:
                file_path = os.path.join(path, file_info["name"])
                lines.append(f"{indent}  <file path=\"{file_path}\" type=\"{file_info['type']}\" size=\"{file_info['size']}\"/>")
        
        # Add subdirectories recursively
        if "subdirs" in directory_info:
            for subdir, subdir_info in directory_info["subdirs"].items():
                subdir_path = os.path.join(path, subdir) if path else subdir
                self._append_directory_structure(lines, subdir_path, subdir_info, indent + "  ")
        
        lines.append(f"{indent}</directory>")
def get_file_type(file_path):
    """Determine file type from extension and content analysis."""
    # First try by extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Map common extensions to more descriptive types
    type_mapping = {
        ".py": "python",
        ".js": "javascript",
        ".html": "html",
        ".css": "css",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "header",
        ".cs": "csharp",
        ".php": "php",
        ".go": "go",
        ".rs": "rust",
        ".ts": "typescript",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "shell",
        ".bat": "batch",
        ".ps1": "powershell",
        ".md": "markdown",
        ".txt": "text",
        ".rst": "restructuredtext",
        ".adoc": "asciidoc",
        ".csv": "csv",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".xml": "xml",
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".gif": "image",
        ".svg": "vector_image"
    }
    
    if ext in type_mapping:
        return type_mapping[ext]
    
    # If extension doesn't give us enough info, check mime type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        major_type, minor_type = mime_type.split('/', 1)
        
        if major_type == 'text':
            return 'text'
        elif major_type == 'image':
            return 'image'
        elif major_type == 'audio':
            return 'audio'
        elif major_type == 'video':
            return 'video'
        elif major_type == 'application':
            if minor_type.startswith('x-'):
                minor_type = minor_type[2:]
            return minor_type
    
    # If we can't determine the type, check if it's binary
    if is_binary_file(file_path):
        return "binary"
    
    # Default to unknown
    return "unknown"

def extract_file_metadata(file_path):
    """Extract metadata from a file based on its type."""
    try:
        file_type = get_file_type(file_path)
        metadata = {
            "type": file_type,
            "size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            "hash": calculate_file_hash(file_path)
        }
        
        # Add type-specific metadata
        if file_type == "python":
            try:
                metadata.update(extract_python_metadata(file_path))
            except Exception as e:
                logger.warning(f"Python metadata extraction error for {file_path}: {e}")
        
        elif file_type == "csv":
            try:
                metadata.update(extract_csv_metadata(file_path))
            except Exception as e:
                logger.warning(f"CSV metadata extraction error for {file_path}: {e}")
        
        elif file_type in ["json", "yaml", "xml"]:
            try:
                metadata.update(extract_config_metadata(file_path, file_type))
            except Exception as e:
                logger.warning(f"Config metadata extraction error for {file_path}: {e}")
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata for {file_path}: {e}")
        return {"type": "unknown", "error": str(e)}

def calculate_file_hash(file_path, block_size=65536):
    """Calculate a hash for the file to detect changes."""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Error calculating hash for {file_path}: {e}")
        return "hash_error"

def extract_python_metadata(file_path):
    """Extract metadata specific to Python files."""
    metadata = {
        "imports": [],
        "classes": [],
        "functions": [],
        "docstring": None
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract docstring (simplistic approach)
        import_pattern = r'^import\s+(\w+)|^from\s+(\w+)'
        metadata["imports"] = list(set(re.findall(import_pattern, content, re.MULTILINE)))
        
        # Find classes
        class_pattern = r'^class\s+(\w+)'
        metadata["classes"] = re.findall(class_pattern, content, re.MULTILINE)
        
        # Find functions
        func_pattern = r'^def\s+(\w+)'
        metadata["functions"] = re.findall(func_pattern, content, re.MULTILINE)
        
        # Get docstring (first triple-quoted string)
        docstring_pattern = r'"""(.*?)"""'
        docstring_match = re.search(docstring_pattern, content, re.DOTALL)
        if docstring_match:
            metadata["docstring"] = docstring_match.group(1).strip()
            
    except Exception as e:
        logger.warning(f"Error extracting Python metadata from {file_path}: {e}")
    
    return metadata

def extract_csv_metadata(file_path):
    """Extract metadata specific to CSV files."""
    metadata = {
        "columns": [],
        "row_count": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        if not lines:
            return metadata
        
        # Get column names from header row
        if ',' in lines[0]:
            metadata["columns"] = [col.strip() for col in lines[0].split(',')]
        
        # Count rows (excluding header)
        metadata["row_count"] = len([line for line in lines[1:] if line.strip()])
    except Exception as e:
        logger.warning(f"Error extracting CSV metadata from {file_path}: {e}")
    
    return metadata

def analyze_project_structure(directory_path, config):
    """
    Generate a hierarchical representation of the project.
    
    Args:
        directory_path: Path to the directory to analyze
        config: Configuration dictionary with exclusion patterns
        
    Returns:
        ProjectMetadata object containing the project structure
    """
    metadata = ProjectMetadata(directory_path)
    
    # Use a queue to avoid recursion depth issues with large projects
    directories_to_process = [("", directory_path)]
    
    while directories_to_process:
        rel_path, abs_path = directories_to_process.pop(0)
        
        try:
            # Get directory contents
            items = os.listdir(abs_path)
            files = []
            subdirs = {}
            
            for item in items:
                item_path = os.path.join(abs_path, item)
                item_rel_path = os.path.join(rel_path, item) if rel_path else item
                
                if os.path.isdir(item_path):
                    # Check if this directory should be excluded
                    if item in config["exclude_directories"]:
                        logger.debug(f"Skipping excluded directory: {item}")
                        continue
                    
                    # Add to processing queue
                    directories_to_process.append((item_rel_path, item_path))
                    subdirs[item] = {"files": [], "subdirs": {}}
                    metadata.directory_count += 1
                    
                elif os.path.isfile(item_path):
                    # Check if this file should be processed
                    if not should_process_file(item_path, config):
                        continue
                    
                    # Extract file metadata
                    file_type = get_file_type(item_path)
                    metadata.file_types[file_type] += 1
                    
                    file_info = {
                        "name": item,
                        "type": file_type,
                        "size": os.path.getsize(item_path),
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                    }
                    
                    files.append(file_info)
                    
                    # Store detailed metadata
                    try:
                        file_metadata = extract_file_metadata(item_path)
                        metadata.files[item_rel_path] = file_metadata
                    except Exception as e:
                        logger.warning(f"Error extracting metadata for {item_path}: {e}")
                    
                    metadata.file_count += 1
            
            # Store directory info
            if rel_path not in metadata.directories:
                metadata.directories[rel_path] = {}
            
            metadata.directories[rel_path]["files"] = files
            metadata.directories[rel_path]["subdirs"] = subdirs
            
        except Exception as e:
            logger.error(f"Error processing directory {abs_path}: {e}")
    
    # Extract file relationships (this would be a complex analysis in practice)
    # For now, we'll just identify imports between Python files
    identify_python_relationships(metadata)
    
    return metadata

def identify_python_relationships(metadata):
    """
    Identify relationships between Python files based on imports.
    
    Args:
        metadata: ProjectMetadata object to update with relationships
    """
    # Map from module name to file path
    module_map = {}
    
    # First pass: build a map of module names to file paths
    for file_path, file_meta in metadata.files.items():
        if file_meta.get("type") == "python":
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module_map[module_name] = file_path
    
    # Second pass: identify imports
    for file_path, file_meta in metadata.files.items():
        if file_meta.get("type") == "python":
            for imported in file_meta.get("imports", []):
                # Handle both "import x" and "from x import y" patterns
                if isinstance(imported, tuple):
                    imported = next((x for x in imported if x), "")
                
                if imported and imported in module_map:
                    target_path = module_map[imported]
                    if target_path != file_path:  # Don't record self-imports
                        metadata.file_relationships.append({
                            "type": "imports",
                            "source": file_path,
                            "target": target_path
                        })

def extract_config_metadata(file_path, file_type):
    """Extract metadata for configuration files."""
    metadata = {"keys": []}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_type == "json":
            # For JSON, extract top-level keys
            import json
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    metadata["keys"] = list(data.keys())
            except:
                pass
        
        # Add similar extraction for YAML, XML, etc. if needed
    except Exception as e:
        logger.warning(f"Error extracting config metadata from {file_path}: {e}")
    
    return metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("file_scanner")

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "include_extensions": [
        # Code files
        ".py", ".js", ".html", ".css", ".java", ".c", ".cpp", ".h", ".cs", ".php", 
        ".go", ".rs", ".ts", ".rb", ".swift", ".kt", ".scala", ".sh", ".bat", ".ps1",
        # Documentation files
        ".md", ".txt", ".rst", ".adoc", ".csv",
        # Config files
        ".json", ".yaml", ".yml", ".toml", ".ini", ".xml"
    ],
    "exclude_extensions": [
        # Binary files
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg", ".webp",
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
        ".zip", ".tar", ".gz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".dylib",
        ".pyc", ".pyo", ".pyd",
        # Other exclusions
        ".DS_Store"
    ],
    "exclude_directories": [
        ".git", "node_modules", "venv", "env", ".env", ".venv",
        "__pycache__", ".pytest_cache", ".vscode", ".idea"
    ],
    "max_file_size_kb": 1024,  # Skip files larger than this size
    "max_output_size_mb": 50,  # Warning threshold for output file
    "output_format": {
        "file_header": "<document index=\"{index}\">\n<source>{path}</source>\n<document_content>{content}</document_content>\n</document>",
        "directory_header": "# Directory: {path}\n",
        "separator": "\n\n"
    },
    "max_workers": 4,  # Number of parallel workers for file processing
    "retry_attempts": 3  # Number of retry attempts for file operations
}

def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return Path.home() / ".file_scanner_config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                merged_config = DEFAULT_CONFIG.copy()
                
                # Deep merge for nested dictionaries
                for key, value in config.items():
                    if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                
                return merged_config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return DEFAULT_CONFIG
    
    # Create default config if it doesn't exist
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    config_path = get_config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        return False

def is_binary_file(file_path: str) -> bool:
    """
    Determine if a file is binary by checking its content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file appears to be binary, False otherwise
    """
    try:
        # Check first 8KB of the file for null bytes
        chunk_size = 8192
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size)
            # Files with null bytes are likely binary
            if b'\x00' in chunk:
                return True
            
            # Check if the chunk contains mostly non-text characters
            text_characters = bytearray(
                {7, 8, 9, 10, 12, 13, 27} | 
                set(range(0x20, 0x7F)) | 
                set(range(0x80, 0x100))
            )
            # If >30% of chars are non-text, probably binary
            non_text = sum(x not in text_characters for x in chunk)
            return non_text > 0.3 * len(chunk)
    except Exception:
        # If we can't read the file, assume it's not binary
        return False
    
    return False

def should_process_file(file_path: str, config: Dict[str, Any]) -> bool:
    """
    Determine if a file should be processed based on configuration.
    
    Args:
        file_path: Path to the file
        config: Configuration dictionary
        
    Returns:
        True if the file should be processed, False otherwise
    """
    # Sanitize and normalize the path
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Skip files with excluded extensions
    if ext in config["exclude_extensions"]:
        return False
    
    # Check if file is in an excluded directory
    path_parts = Path(file_path).parts
    for excluded_dir in config["exclude_directories"]:
        if excluded_dir in path_parts:
            return False
    
    # Check file size
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > config["max_file_size_kb"]:
            logger.debug(f"Skipping large file: {file_path} ({file_size_kb:.2f} KB)")
            return False
    except Exception as e:
        logger.warning(f"Error checking file size for {file_path}: {e}")
        return False
    
    # For files without specific extensions we want, check if they're binary
    if ext not in config["include_extensions"]:
        if is_binary_file(file_path):
            return False
    
    # Include files with specific extensions
    return ext in config["include_extensions"]

def read_file_content(file_path: str, retry_attempts: int = 3) -> str:
    """
    Read content from a file with error handling for different encodings.
    
    Args:
        file_path: Path to the file
        retry_attempts: Number of retry attempts
        
    Returns:
        File content as a string or error message
    """
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
    attempt = 0
    
    while attempt < retry_attempts:
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                break
        
        # Wait before retrying
        attempt += 1
        if attempt < retry_attempts:
            time.sleep(0.5)
    
    return f"[Error: Unable to read file content]"

def process_file(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Process a single file.
    
    Args:
        args: Tuple containing (file_path, root_directory, config)
        
    Returns:
        Dictionary with file path and content
    """
    file_path, root_directory, config = args
    
    try:
        relative_path = os.path.relpath(file_path, root_directory)
        content = read_file_content(file_path, config["retry_attempts"])
        
        return {
            "path": relative_path,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {
            "path": os.path.basename(file_path),
            "content": f"[Error processing file: {str(e)}]"
        }

def scan_directory(
    directory_path: str, 
    config: Dict[str, Any], 
    callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, str]]:
    """
    Scan a directory recursively and extract content from files.
    
    Args:
        directory_path: Path to the directory to scan
        config: Configuration dictionary
        callback: Optional callback function for progress updates
        
    Returns:
        A list of dictionaries with file paths and their content
    """
    # Normalize directory path
    directory_path = os.path.normpath(os.path.abspath(directory_path))
    
    # Collect files to process
    files_to_process = []
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in config["exclude_directories"]]
        
        for file in files:
            file_path = os.path.join(root, file)
            if should_process_file(file_path, config):
                files_to_process.append((file_path, directory_path, config))
    
    total_files = len(files_to_process)
    logger.info(f"Found {total_files} files to process")
    
    if callback:
        callback(0, total_files)
    
    results = []
    processed_count = 0
    
    # Use parallel processing for file reading
    max_workers = min(config.get("max_workers", 2), 16)  # Limit to reasonable number
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file, args): args[0] 
                for args in files_to_process
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    # Add debug logging to catch None values
                    if result is None:
                        logger.warning(f"Got None result for file: {file_path}")
                        # Create a placeholder instead of None
                        result = {
                            "path": os.path.relpath(file_path, directory_path),
                            "content": "[Error: Unexpected None result]"
                        }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Add the failed file with an error message
                    results.append({
                        "path": os.path.relpath(file_path, directory_path),
                        "content": f"[Error: {str(e)}]"
                    })
                
                processed_count += 1
                if callback:
                    callback(processed_count, total_files)
    except Exception as e:
        logger.error(f"Error in thread pool execution: {e}")
        # Continue with any files processed so far

    valid_results = []
    for item in results:
        if item is not None:
            valid_results.append(item)
        else:
            logger.warning("Found None item in results before sorting")
    
    # Sort valid results by path
    valid_results.sort(key=lambda x: x["path"])
    return valid_results

def generate_output(scan_results: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """
    Generate formatted output from scan results.
    
    Args:
        scan_results: List of dictionaries with file paths and content
        config: Configuration dictionary
        
    Returns:
        Formatted string with all file content
    """
    output_parts = []
    
    for i, result in enumerate(scan_results, 1):
        file_header = config["output_format"]["file_header"].format(
            index=i,
            path=result["path"],
            content=result["content"]
        )
        output_parts.append(file_header)
    
    return config["output_format"]["separator"].join(output_parts)

def validate_xml(xml_string):
    """Basic validation to ensure XML is well-formed."""
    try:
        import xml.etree.ElementTree as ET
        ET.fromstring(f"<root>{xml_string}</root>")
        return True
    except Exception as e:
        logger.error(f"XML validation error: {e}")
        return False

def write_output_file(content: str, output_path: str) -> bool:
    """
    Write content to output file.
    
    Args:
        content: Content to write
        output_path: Path to the output file
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            return False
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return False

def open_file(file_path: str) -> bool:
    """
    Open a file using the system's default application.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use webbrowser module which handles paths with spaces safely
        file_url = f"file://{os.path.abspath(file_path)}"
        webbrowser.open(file_url)
        return True
    except Exception as e:
        logger.error(f"Error opening file: {e}")
        
        # Fallback to platform-specific methods
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', file_path], check=True)
            elif sys.platform == 'win32':  # Windows
                os.startfile(os.path.normpath(file_path))
            else:  # Linux or other Unix
                subprocess.run(['xdg-open', file_path], check=True)
            return True
        except Exception as e2:
            logger.error(f"Error in fallback file opening: {e2}")
            return False

def run_scan(
    directory_path, 
    output_path, 
    config=None,
    progress_callback=None
) -> Optional[Dict[str, Any]]:
    """
    Run the full scanning process with enhanced metadata analysis.
    """
    if config is None:
        config = load_config()
    
    start_time = time.time()
    logger.info(f"Starting scan of {directory_path}")
    
    # Define total steps
    total_steps = 3  # Metadata, content scan, output generation
    current_step = 0
    
    def update_progress(step_progress, step_total):
        if progress_callback:
            # Calculate overall progress
            total_progress = ((current_step * 100) + (step_progress / step_total * 100)) / total_steps
            progress_callback(int(total_progress), 100)
    
    # Step 1: Analyze project structure
    logger.info("Analyzing project structure...")
    current_step = 0
    update_progress(0, 1)
    project_metadata = analyze_project_structure(directory_path, config)
    current_step = 1
    update_progress(1, 1)
    
    # Step 2: Scan directory content
    logger.info(f"Scanning {project_metadata.file_count} files for content...")
    
    # Custom progress tracker for content scanning
    def content_progress(current, total):
        update_progress(current, total)
    
    scan_results = scan_directory(directory_path, config, content_progress)
    logger.info(f"Completed scanning {len(scan_results)} files")
    current_step = 2
    update_progress(0, 1)
    
    # Step 3: Generate enhanced output
    logger.info("Generating enhanced output file")
    output_content = generate_enhanced_output(project_metadata, scan_results, config)
    update_progress(1, 1)
        
    # Check output size
    output_size_mb = len(output_content) / (1024 * 1024)
    if output_size_mb > config["max_output_size_mb"]:
        logger.warning(f"Warning: Output file is large ({output_size_mb:.2f} MB)")
    
    # Write output file
    logger.info(f"Writing output to {output_path}")
    success = write_output_file(output_content, output_path)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if success:
        summary = {
            "directory": directory_path,
            "output_file": output_path,
            "files_processed": len(scan_results),
            "output_size_mb": output_size_mb,
            "duration_seconds": duration,
            "project_info": {
                "file_count": project_metadata.file_count,
                "directory_count": project_metadata.directory_count,
                "file_types": dict(project_metadata.file_types)
            }
        }
        logger.info(f"Scan completed in {duration:.2f} seconds")
        return summary
    else:
        logger.error("Failed to write output file")
        return None
    
def generate_enhanced_output(project_metadata, scan_results, config):
    """
    Generate enhanced XML output with metadata and directory structure.
    
    Args:
        project_metadata: ProjectMetadata object with project structure
        scan_results: List of dictionaries with file content
        config: Configuration dictionary
        
    Returns:
        Formatted string with all file content and metadata
    """
    output_parts = []
    
    # 1. Add the project summary
    output_parts.append(project_metadata.to_xml())
    
    # 2. Create a file path lookup for the scan results
    file_path_map = {result["path"]: i for i, result in enumerate(scan_results)}
    
    # 3. Add a table of contents
    output_parts.append(generate_table_of_contents(project_metadata, file_path_map))
    
    # 4. Add file relationship information
    output_parts.append(generate_relationship_section(project_metadata, file_path_map))
    
    # 5. Add the content for each scanned file
    for i, result in enumerate(scan_results, 1):
        file_path = result["path"]
        content = result["content"]
        
        # Get any metadata we have for this file
        file_metadata = project_metadata.files.get(file_path, {})
        
        # Generate the enhanced document entry
        doc_entry = config["output_format"]["file_header"]
        
        # Insert file metadata if available
        if file_metadata:
            # Create metadata XML
            metadata_xml = "<metadata>\n"
            for key, value in file_metadata.items():
                if key == "type":
                    doc_entry = doc_entry.replace('<document index=', f'<document type="{value}" index=')
                elif isinstance(value, (list, dict)):
                    # Format complex metadata as nested XML
                    metadata_xml += f"  <{key}>\n"
                    if isinstance(value, list):
                        for item in value:
                            metadata_xml += f"    <item>{item}</item>\n"
                    else:  # dict
                        for k, v in value.items():
                            metadata_xml += f"    <{k}>{v}</{k}>\n"
                    metadata_xml += f"  </{key}>\n"
                else:
                    # Simple key-value pair
                    metadata_xml += f"  <{key}>{value}</{key}>\n"
            metadata_xml += "</metadata>\n"
            
            # Insert metadata before the content
            doc_entry = doc_entry.replace('<document_content>', metadata_xml + '<document_content>')
        
        # Format the document entry
        doc_entry = doc_entry.format(
            index=i,
            path=file_path,
            content=content
        )
        
        output_parts.append(doc_entry)
    
    # Join all parts with the configured separator
    return config["output_format"]["separator"].join(output_parts)

def generate_table_of_contents(project_metadata, file_path_map):
    """
    Generate a table of contents XML section.
    
    Args:
        project_metadata: ProjectMetadata object with project info
        file_path_map: Dictionary mapping file paths to their index in scan results
        
    Returns:
        XML string with table of contents
    """
    lines = []
    lines.append("<table_of_contents>")
    
    # Group files by type for the TOC
    type_sections = defaultdict(list)
    
    for file_path, metadata in project_metadata.files.items():
        if file_path in file_path_map:  # Only include files that were scanned
            file_type = metadata.get("type", "unknown")
            index = file_path_map[file_path]
            type_sections[file_type].append((file_path, index))
    
    # Add each section to the TOC
    for file_type, files in sorted(type_sections.items()):
        # Make section name more readable
        section_name = file_type.replace("_", " ").title() + " Files"
        
        lines.append(f"  <section name=\"{section_name}\">")
        
        # Sort files by path
        for file_path, index in sorted(files, key=lambda x: x[0]):
            lines.append(f"    <file index=\"{index + 1}\" path=\"{file_path}\"/>")
        
        lines.append("  </section>")
    
    lines.append("</table_of_contents>")
    return "\n".join(lines)

def generate_relationship_section(project_metadata, file_path_map):
    """
    Generate XML section for file relationships.
    
    Args:
        project_metadata: ProjectMetadata object with relationship info
        file_path_map: Dictionary mapping file paths to their index in scan results
        
    Returns:
        XML string with relationship information
    """
    lines = []
    lines.append("<file_relationships>")
    
    for rel in project_metadata.file_relationships:
        rel_type = rel.get("type", "unknown")
        source_path = rel.get("source", "")
        target_path = rel.get("target", "")
        
        # Only include relationships where both files are in the scanned results
        if source_path in file_path_map and target_path in file_path_map:
            source_index = file_path_map[source_path] + 1  # 1-based index
            target_index = file_path_map[target_path] + 1  # 1-based index
            
            lines.append(f"  <relationship type=\"{rel_type}\">")
            lines.append(f"    <source file_index=\"{source_index}\" path=\"{source_path}\"/>")
            lines.append(f"    <target file_index=\"{target_index}\" path=\"{target_path}\"/>")
            lines.append("  </relationship>")
    
    lines.append("</file_relationships>")
    return "\n".join(lines)

class FileScanner(tk.Tk):
    """GUI application for the file scanner."""
    
    def __init__(self):
        super().__init__()
        self.title("File Scanner")
        self.geometry("650x550")
        self.minsize(650, 650)
        self.config = load_config()
        
        self.create_widgets()
        self.center_window()
    
    def center_window(self):
        """Center the window on the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create the GUI widgets."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the main frame
        main_frame = ttk.Frame(self.notebook, padding=10)
        settings_frame = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(main_frame, text="Scan")
        self.notebook.add(settings_frame, text="Settings")
        
        # Main tab
        self.create_main_tab(main_frame)
        
        # Settings tab
        self.create_settings_tab(settings_frame)
    
    def create_main_tab(self, parent):
        """Create the main scanning tab."""
        # Directory selection
        ttk.Label(parent, text="Directory to scan:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.directory_var = tk.StringVar()
        directory_entry = ttk.Entry(parent, textvariable=self.directory_var, width=50)
        directory_entry.grid(column=0, row=1, sticky=(tk.W, tk.E), pady=5)
        
        browse_button = ttk.Button(parent, text="Browse...", command=self.browse_directory, state=tk.NORMAL)
        browse_button.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Output file selection - Reordered to keep related elements together
        ttk.Label(parent, text="Output file:").grid(column=0, row=2, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        self.output_var.set(os.path.join(os.getcwd(), "scan_output.txt"))
        output_entry = ttk.Entry(parent, textvariable=self.output_var, width=50)
        output_entry.grid(column=0, row=3, sticky=(tk.W, tk.E), pady=5)
        
        output_button = ttk.Button(parent, text="Browse...", command=self.browse_output)
        output_button.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)
        
        # Progress information
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(column=0, row=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to scan")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(column=0, row=1, sticky=tk.W, pady=5)
        
        # Detail frame for logged output
        detail_frame = ttk.LabelFrame(parent, text="Details", padding=10)
        detail_frame.grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.detail_text = tk.Text(detail_frame, height=10, width=60, wrap=tk.WORD)
        self.detail_text.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
        scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.detail_text.configure(yscrollcommand=scrollbar.set)
        
        # Scan and Open buttons
       # Replace the button_frame section with this code
        button_frame = ttk.Frame(parent)
        button_frame.grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        button_frame.grid_propagate(False)  # Prevent the frame from shrinking
        button_frame.config(height=50)      # Force minimum height
        reset_button = ttk.Button(detail_frame, text="Clear Log", command=self.reset_details)
        reset_button.grid(column=0, row=1, sticky=tk.E, pady=5)

        # Center the buttons using pack with side and expand
        self.scan_button = ttk.Button(button_frame, text="Start Scan", command=self.start_scan)
        self.scan_button.pack(side=tk.LEFT, padx=5, expand=True)

        self.open_button = ttk.Button(button_frame, text="Open Output", command=self.open_output, state=tk.DISABLED)
        self.open_button.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Configure grid
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(6, weight=0)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)

        parent.rowconfigure(6, minsize=50)  # Ensure button row has minimum height
        # At the end of create_main_tab
        parent.grid_rowconfigure(6, minsize=60)  # Force minimum height for button row

    def reset_details(self):
        """Clear the details text area."""
        self.detail_text.delete(1.0, tk.END)
        self.log_message("Log cleared.") 
   
    def create_settings_tab(self, parent):
        """Create the settings tab."""
        # File extensions to include
        ttk.Label(parent, text="File Extensions to Include:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.include_ext_text = tk.Text(parent, height=5, width=50, wrap=tk.WORD)
        self.include_ext_text.grid(column=0, row=1, sticky=(tk.W, tk.E), pady=5)
        self.include_ext_text.insert(tk.END, ', '.join(self.config["include_extensions"]))
        
        # File extensions to exclude
        ttk.Label(parent, text="File Extensions to Exclude:").grid(column=0, row=2, sticky=tk.W, pady=5)
        self.exclude_ext_text = tk.Text(parent, height=5, width=50, wrap=tk.WORD)
        self.exclude_ext_text.grid(column=0, row=3, sticky=(tk.W, tk.E), pady=5)
        self.exclude_ext_text.insert(tk.END, ', '.join(self.config["exclude_extensions"]))
        
        # Directories to exclude
        ttk.Label(parent, text="Directories to Exclude:").grid(column=0, row=4, sticky=tk.W, pady=5)
        self.exclude_dir_text = tk.Text(parent, height=5, width=50, wrap=tk.WORD)
        self.exclude_dir_text.grid(column=0, row=5, sticky=(tk.W, tk.E), pady=5)
        self.exclude_dir_text.insert(tk.END, ', '.join(self.config["exclude_directories"]))
        
        # File size limit
        ttk.Label(parent, text="Maximum File Size (KB):").grid(column=0, row=6, sticky=tk.W, pady=5)
        self.max_file_size_var = tk.StringVar(value=str(self.config["max_file_size_kb"]))
        max_file_size_entry = ttk.Entry(parent, textvariable=self.max_file_size_var, width=10)
        max_file_size_entry.grid(column=0, row=7, sticky=tk.W, pady=5)
        
        # Parallel workers
        ttk.Label(parent, text="Number of Parallel Workers:").grid(column=0, row=8, sticky=tk.W, pady=5)
        self.max_workers_var = tk.StringVar(value=str(self.config.get("max_workers", 4)))
        max_workers_entry = ttk.Entry(parent, textvariable=self.max_workers_var, width=10)
        max_workers_entry.grid(column=0, row=9, sticky=tk.W, pady=5)
        
        # Save settings button
        save_button = ttk.Button(parent, text="Save Settings", command=self.save_settings)
        save_button.grid(column=0, row=10, pady=10)
        
        # Configure grid
        parent.columnconfigure(0, weight=1)
    
    def browse_directory(self):
        """Browse for directory to scan."""
        directory = filedialog.askdirectory()
        if directory:
            self.directory_var.set(directory)
            
            # Auto-generate output file name based on directory name
            dir_name = os.path.basename(os.path.abspath(directory))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(os.getcwd(), f"{dir_name}_scan_{timestamp}.txt")
            
            self.output_var.set(output_path)
    
    def browse_output(self):
        """Browse for output file location."""
        output_file = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if output_file:
            self.output_var.set(output_file)
    
    def update_progress(self, current, total):
        """Update progress bar and status."""
        if total > 0:
            progress_percent = (current / total) * 100
            self.progress_var.set(progress_percent)
            self.status_var.set(f"Processing file {current} of {total}")
            self.update_idletasks()
    
    def log_message(self, message):
        """Add a message to the detail text area."""
        self.detail_text.insert(tk.END, f"{message}\n")
        self.detail_text.see(tk.END)
        self.update_idletasks()
    
    def start_scan(self):
        """Start the scanning process."""
        directory = self.directory_var.get()
        output_file = self.output_var.get()
        
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid directory")
            return
        
        if not output_file:
            messagebox.showerror("Error", "Please specify an output file")
            return
        
        # Disable UI during scan
        self.scan_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Starting scan...")
        self.open_button.configure(state=tk.DISABLED)
        self.detail_text.delete(1.0, tk.END)
        self.update_idletasks()
        
        # Custom logging handler to show logs in the UI
        class TextHandler(logging.Handler):
            def __init__(self, log_callback):
                super().__init__()
                self.log_callback = log_callback
            
            def emit(self, record):
                msg = self.format(record)
                self.log_callback(msg)
        
        # Add handler to logger
        handler = TextHandler(self.log_message)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)
        
        try:
            # Run scan in a separate thread
            def run_scan_thread():
                summary = run_scan(directory, output_file, self.config, self.update_progress)
                self.after(0, lambda: self.scan_completed(summary, handler))
            
            import threading
            scan_thread = threading.Thread(target=run_scan_thread)
            scan_thread.daemon = True
            scan_thread.start()
            
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
            # Re-enable UI
            self.scan_button.configure(state=tk.NORMAL)
            self.status_var.set("Ready to scan")
            
            # Remove custom handler
            logger.removeHandler(handler)
    
    def scan_completed(self, summary, handler=None):
        """Handle scan completion."""
        # Re-enable UI
        self.scan_button.configure(state=tk.NORMAL)
        self.status_var.set("Scan completed")
        
        # Remove custom handler if provided
        if handler:
            logger.removeHandler(handler)
        
        if summary:
            message = (
                f"Scan completed successfully!\n\n"
                f"Files processed: {summary['files_processed']}\n"
                f"Output size: {summary['output_size_mb']:.2f} MB\n"
                f"Duration: {summary['duration_seconds']:.2f} seconds"
            )
            self.log_message("\n" + message)
            messagebox.showinfo("Scan Complete", message)
            
            # Enable open button if the output file exists
            if os.path.exists(summary['output_file']):
                self.open_button.configure(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Failed to complete the scan")
    
    def open_output(self):
        """Open the output file."""
        output_file = self.output_var.get()
        if os.path.exists(output_file):
            if not open_file(output_file):
                messagebox.showerror("Error", "Failed to open the output file")
        else:
            messagebox.showerror("Error", "Output file does not exist")
    
    def save_settings(self):
        """Save settings to configuration file."""
        try:
            # Parse text inputs
            include_extensions = [
                ext.strip() for ext in self.include_ext_text.get(1.0, tk.END).replace(',', ' ').split()
                if ext.strip()
            ]
            
            exclude_extensions = [
                ext.strip() for ext in self.exclude_ext_text.get(1.0, tk.END).replace(',', ' ').split()
                if ext.strip()
            ]
            
            exclude_directories = [
                dir.strip() for dir in self.exclude_dir_text.get(1.0, tk.END).replace(',', ' ').split()
                if dir.strip()
            ]
            
            max_file_size = int(self.max_file_size_var.get())
            max_workers = int(self.max_workers_var.get())
            
            # Update config
            self.config["include_extensions"] = include_extensions
            self.config["exclude_extensions"] = exclude_extensions
            self.config["exclude_directories"] = exclude_directories
            self.config["max_file_size_kb"] = max_file_size
            self.config["max_workers"] = max_workers
            
            # Save config
            if save_config(self.config):
                messagebox.showinfo("Settings", "Settings saved successfully")
            else:
                messagebox.showerror("Error", "Failed to save settings")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number value: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving settings: {str(e)}")

def create_gui():
    """Create and run the GUI application."""
    app = FileScanner()
    app.mainloop()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="File Scanner - Code and Documentation Capture Tool")
    parser.add_argument("directory", nargs="?", help="Directory to scan")
    parser.add_argument("output", nargs="?", help="Output file path")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load custom config if specified
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading custom configuration: {e}")
            return
    else:
        config = load_config()
    
    # Launch GUI if requested or no arguments
    if args.gui or (not args.directory and not args.output):
        create_gui()
        return
    
    # Command line mode
    if not args.directory:
        logger.error("Error: Please specify a directory to scan")
        return
    
    if not args.output:
        # Default output file name based on directory name and timestamp
        dir_name = os.path.basename(os.path.abspath(args.directory))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{dir_name}_scan_{timestamp}.txt"
        logger.info(f"Output file not specified. Using: {args.output}")
    
    # Run scan
    logger.info(f"Scanning directory: {args.directory}")
    logger.info(f"Output file: {args.output}")
    
    def print_progress(current, total):
        percent = (current / total) * 100 if total > 0 else 0
        print(f"Progress: {current}/{total} files ({percent:.1f}%)", end="\r")
    
    summary = run_scan(args.directory, args.output, config, print_progress)
    
    if summary:
        print("\nScan completed successfully!")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Output size: {summary['output_size_mb']:.2f} MB")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Output file: {summary['output_file']}")
    else:
        print("\nFailed to complete the scan")

if __name__ == "__main__":
    main()
