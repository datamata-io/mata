#!/usr/bin/env python3
r"""
Clean personally identifiable information (PII) from Jupyter notebooks.

Removes or sanitizes:
1. Development directory paths (d:\Documents\OneDrive\Code\mtp\datamata_io\mata\)
2. User temporary directory paths (C:\Users\<username>\AppData\Local\Temp\)
3. Full environment paths in warning messages
4. Absolute paths in metadata
"""

import json
import re
import sys
from pathlib import Path


def sanitize_output_text(text):
    """Sanitize text content in notebook outputs."""
    if not isinstance(text, str):
        return text
    
    # Replace user temp directory paths
    # C:\Users\biman\AppData\Local\Temp\tmp... -> <USER_TEMP_DIR>/...
    text = re.sub(
        r'C:\\Users\\[^\\]+\\AppData\\Local\\Temp\\',
        '<USER_TEMP_DIR>/',
        text,
        flags=re.IGNORECASE
    )
    
    # Replace development directory paths in warnings
    # d:\Documents\OneDrive\Code\mtp\datamata_io\mata\ -> <REPO>/
    text = re.sub(
        r'd:\\Documents\\OneDrive\\Code\\mtp\\datamata_io\\mata\\',
        '<REPO>/',
        text,
        flags=re.IGNORECASE
    )
    
    # Replace development directory paths (double backslash version)
    text = re.sub(
        r'd:\\\\Documents\\\\OneDrive\\\\Code\\\\mtp\\\\datamata_io\\\\mata\\\\',
        '<REPO>/',
        text,
        flags=re.IGNORECASE
    )
    
    # Replace full paths that look like relative image paths at the end of long absolute paths
    # But keep relative paths like ../../examples/images/...
    text = re.sub(
        r'd:\\\\Documents\\\\OneDrive\\\\Code\\\\mtp\\\\datamata_io\\\\mata\\\\examples\\\\notebooks\\\\\.\.\\\\\.\.\\\\examples\\\\',
        '<REPO>/examples/',
        text,
        flags=re.IGNORECASE
    )
    
    return text


def should_remove_stderr(text):
    """Check if a stderr output should be removed."""
    if not isinstance(text, str):
        return False
    
    # Remove TqdmWarning about jupyter/ipywidgets
    if 'TqdmWarning' in text and 'IProgress not found' in text:
        return True
    
    return False


def clean_notebook(notebook_path):
    """Clean a single notebook file."""
    print(f"Processing: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    # Process all cells
    for cell in notebook.get('cells', []):
        # Process outputs
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stderr':
                # Check if stderr should be removed (TqdmWarning)
                stderr_text = ''.join(output.get('text', []))
                if should_remove_stderr(stderr_text):
                    # Mark for removal (we'll handle this after iteration)
                    output['_remove'] = True
                    changes_made += 1
                    print(f"  - Removing TqdmWarning stderr output")
            
            # Sanitize text content in all outputs
            if 'text' in output:
                for i, line in enumerate(output['text']):
                    sanitized = sanitize_output_text(line)
                    if sanitized != line:
                        output['text'][i] = sanitized
                        changes_made += 1
                        print(f"  - Sanitized output text")
            
            # Sanitize stdout
            if output.get('output_type') == 'stdout':
                stdout_text = ''.join(output.get('text', []))
                sanitized = sanitize_output_text(stdout_text)
                if sanitized != stdout_text:
                    output['text'] = sanitized.splitlines(keepends=True)
                    changes_made += 1
                    print(f"  - Sanitized stdout")
            
            # Sanitize metadata or other fields that might contain paths
            if 'metadata' in output:
                for key, value in output['metadata'].items():
                    if isinstance(value, str):
                        sanitized = sanitize_output_text(value)
                        if sanitized != value:
                            output['metadata'][key] = sanitized
                            changes_made += 1
        
        # Remove marked outputs
        cell['outputs'] = [
            output for output in cell.get('outputs', [])
            if not output.get('_remove')
        ]
    
    # Write back if changes were made
    if changes_made > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[OK] Cleaned {changes_made} occurrences in {notebook_path}\n")
        return True
    else:
        print(f"[OK] No changes needed\n")
        return False


def main():
    """Main entry point."""
    # Get the current script directory
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    notebooks_dir = script_dir.parent / 'examples' / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"Error: Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    notebooks = list(notebooks_dir.glob('*.ipynb'))
    if not notebooks:
        print(f"Error: No notebooks found in {notebooks_dir}")
        sys.exit(1)
    
    print(f"Found {len(notebooks)} notebooks to clean\n")
    
    cleaned_count = 0
    for notebook_path in sorted(notebooks):
        if clean_notebook(notebook_path):
            cleaned_count += 1
    
    print(f"\n[OK] Cleaned {cleaned_count} notebook(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
