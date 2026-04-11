"""Async API for programmatic scientific document generation."""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Union, Literal
from datetime import datetime
from dotenv import load_dotenv

from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
from claude_agent_sdk.types import HookMatcher, StopHookInput, HookContext

from .core import (
    get_api_key,
    load_system_instructions,
    ensure_output_folder,
    get_data_files,
    process_data_files,
    create_data_context_message,
    setup_claude_skills,
)
from .models import ProgressUpdate, TextUpdate, PaperResult, PaperMetadata, PaperFiles, TokenUsage
from .utils import (
    scan_paper_directory,
    count_citations_in_bib,
    extract_citation_style,
    count_words_in_tex,
    extract_title_from_tex,
)

# Model mapping for effort levels
EFFORT_LEVEL_MODELS = {
    "low": "claude-haiku-4-5",
    "medium": "claude-sonnet-4-6",
    "high": "claude-opus-4-6",
}


def create_completion_check_stop_hook(auto_continue: bool = True):
    """
    Create a stop hook that optionally forces continuation.
    """
    async def completion_check_stop_hook(
        hook_input: StopHookInput,
        matcher: str | None,
        context: HookContext,
    ) -> dict:
        if auto_continue:
            return {"continue_": True}
        return {"continue_": False}
    
    return completion_check_stop_hook


async def generate_paper(
    query: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    effort_level: Literal["low", "medium", "high"] = "low",
    data_files: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    track_token_usage: bool = True,
    auto_continue: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate a scientific document asynchronously with progress updates.
    """
    start_time = time.time()
    
    if model is None:
        model = EFFORT_LEVEL_MODELS[effort_level]
    
    if cwd:
        work_dir = Path(cwd).resolve()
    else:
        work_dir = Path.cwd().resolve()
    
    # Load .env from working directory
    env_file = work_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    
    # Get API key
    try:
        api_key_value = get_api_key(api_key)
    except ValueError as e:
        yield _create_error_result(str(e))
        return
    
    # Get package directory for copying skills to working directory
    package_dir = Path(__file__).parent.absolute()
    
    # Set up Claude skills in the working directory (includes WRITER.md)
    setup_claude_skills(package_dir, work_dir)
    
    # Ensure output folder exists
    output_folder = ensure_output_folder(work_dir, output_dir)
    
    yield ProgressUpdate(
        message="Initializing document generation",
        stage="initialization",
    ).to_dict()
    
    # Load system instructions
    system_instructions = load_system_instructions(work_dir)
    
    # ── ONLY CHANGE FROM ORIGINAL ──────────────────────────────────────────
    # Removed the "NEVER write to /tmp/" restriction so Celery workers can
    # use OUTPUT_BASE_DIR freely regardless of where it lives on disk.
    system_instructions += "\n\n" + f"""
CRITICAL - OUTPUT DIRECTORY (MUST FOLLOW EXACTLY):
- Your ONLY working directory is: {work_dir}
- The writing_outputs folder is: {work_dir}/writing_outputs/
- ALL files MUST be created inside: {work_dir}/writing_outputs/<timestamp>_<description>/
- DO NOT create files anywhere else — not in the current directory, not in ~/,
  not in any other path. ONLY inside {work_dir}/writing_outputs/
- If you are unsure of the path, run: echo $PWD to confirm you are in {work_dir}
- The first bash command you run MUST be:
  mkdir -p "{work_dir}/writing_outputs"

IMPORTANT - CONVERSATION CONTINUITY:
- This is a NEW paper request - create a new paper directory
- Create a unique timestamped directory in the writing_outputs folder
- Do NOT assume there's an existing paper unless explicitly told in the prompt context
"""
    # ── END OF CHANGE ──────────────────────────────────────────────────────
    
    data_context = ""
    temp_paper_path = None
    
    if data_files:
        data_file_paths = get_data_files(work_dir, data_files)
        if data_file_paths:
            yield ProgressUpdate(
                message=f"Found {len(data_file_paths)} data file(s) to process",
                stage="initialization",
            ).to_dict()
    
    env_auto_continue = os.environ.get("SCIENTIFIC_WRITER_AUTO_CONTINUE", "").lower()
    if env_auto_continue in ("false", "0", "no"):
        auto_continue = False
    
    options = ClaudeAgentOptions(
        system_prompt=system_instructions,
        model=model,
        allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "research-lookup"],
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        cwd=str(work_dir),
        max_turns=500,
        hooks={
            "Stop": [
                HookMatcher(
                    matcher=None,
                    hooks=[create_completion_check_stop_hook(auto_continue=auto_continue)],
                )
            ]
        },
    )
    
    current_stage = "initialization"
    output_directory = None
    last_message = ""
    tool_call_count = 0
    files_written = []
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0

    def _normalize_usage_value(usage, key):
        if isinstance(usage, dict):
            value = usage.get(key, 0)
        else:
            value = getattr(usage, key, 0)
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    
    yield ProgressUpdate(
        message="Starting document generation",
        stage="initialization",
        details={"query_length": len(query)},
    ).to_dict()
    
    try:
        accumulated_text = ""
        async for message in claude_query(prompt=query, options=options):
            if track_token_usage and hasattr(message, "usage") and message.usage:
                usage = message.usage
                total_input_tokens += _normalize_usage_value(usage, "input_tokens")
                total_output_tokens += _normalize_usage_value(usage, "output_tokens")
                total_cache_creation_tokens += _normalize_usage_value(usage, "cache_creation_input_tokens")
                total_cache_read_tokens += _normalize_usage_value(usage, "cache_read_input_tokens")
            
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        text = block.text
                        accumulated_text += text
                        
                        yield TextUpdate(content=text).to_dict()
                        
                        stage, msg = _analyze_progress(accumulated_text, current_stage)
                        
                        if stage != current_stage and msg and msg != last_message:
                            current_stage = stage
                            last_message = msg
                            yield ProgressUpdate(
                                message=msg,
                                stage=stage,
                            ).to_dict()
                    
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_call_count += 1
                        tool_name = getattr(block, "name", "unknown")
                        tool_input = getattr(block, "input", {})
                        
                        if tool_name.lower() == "write":
                            file_path = tool_input.get("file_path", tool_input.get("path", ""))
                            if file_path:
                                files_written.append(file_path)
                        elif tool_name.lower() == "bash":
                            # Also track any mkdir writing_outputs commands
                            cmd = tool_input.get("command", "")
                            if "writing_outputs" in cmd and "mkdir" in cmd:
                                files_written.append(cmd)
                        
                        tool_progress = _analyze_tool_use(tool_name, tool_input, current_stage)
                        
                        if tool_progress:
                            stage, msg = tool_progress
                            if msg != last_message:
                                current_stage = stage
                                last_message = msg
                                yield ProgressUpdate(
                                    message=msg,
                                    stage=stage,
                                    details={
                                        "tool": tool_name,
                                        "tool_calls": tool_call_count,
                                        "files_created": len(files_written),
                                    },
                                ).to_dict()
        
        yield ProgressUpdate(
            message="Scanning output directory",
            stage="complete",
        ).to_dict()
        
        output_directory = _find_most_recent_output(output_folder, start_time)

        # Fallback: AI ignored cwd and wrote somewhere else — find it from files_written
        if not output_directory and files_written:
            output_directory = _find_output_from_written_files(files_written, start_time)

        # Last resort: scan cwd itself for a writing_outputs folder
        if not output_directory:
            for candidate in [Path.cwd(), Path.home()]:
                fallback = candidate / "writing_outputs"
                if fallback.exists():
                    found = _find_most_recent_output(fallback, start_time)
                    if found:
                        output_directory = found
                        break

        if not output_directory:
            error_result = _create_error_result("Output directory not found after generation")
            if track_token_usage:
                error_result['token_usage'] = TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                ).to_dict()
            yield error_result
            return
        
        if data_files:
            data_file_paths = get_data_files(work_dir, data_files)
            if data_file_paths:
                processed_info = process_data_files(
                    work_dir, 
                    data_file_paths, 
                    str(output_directory),
                    delete_originals=False
                )
                if processed_info:
                    manuscript_count = len(processed_info.get('manuscript_files', []))
                    message = f"Processed {len(processed_info['all_files'])} file(s)"
                    if manuscript_count > 0:
                        message += f" ({manuscript_count} manuscript(s) copied to drafts/)"
                    yield ProgressUpdate(
                        message=message,
                        stage="complete",
                    ).to_dict()
        
        file_info = scan_paper_directory(output_directory)
        result = _build_paper_result(output_directory, file_info)
        
        if track_token_usage:
            result.token_usage = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )
        
        yield ProgressUpdate(
            message="Document generation complete",
            stage="complete",
        ).to_dict()
        
        yield result.to_dict()
        
    except Exception as e:
        error_result = _create_error_result(f"Error during document generation: {str(e)}")
        if track_token_usage:
            error_result['token_usage'] = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            ).to_dict()
        yield error_result


def _analyze_progress(text: str, current_stage: str) -> tuple:
    text_lower = text.lower()
    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    
    if current_idx < stage_order.index("compilation"):
        if "pdflatex" in text_lower or "latexmk" in text_lower or "compiling" in text_lower:
            return "compilation", "Compiling document"
    
    if current_idx < stage_order.index("complete"):
        if "successfully compiled" in text_lower or "pdf generated" in text_lower:
            return "complete", "Finalizing output"
    
    return current_stage, None


def _detect_document_type(file_path: str) -> str:
    path_lower = file_path.lower()
    if "slide" in path_lower or "presentation" in path_lower or "beamer" in path_lower:
        return "slides"
    elif "poster" in path_lower:
        return "poster"
    elif "report" in path_lower:
        return "report"
    elif "grant" in path_lower or "proposal" in path_lower:
        return "grant"
    return "document"


def _get_section_from_filename(filename: str) -> str:
    name_lower = filename.lower().replace('.tex', '').replace('.md', '')
    section_mappings = {
        'abstract': 'abstract',
        'intro': 'introduction',
        'introduction': 'introduction',
        'method': 'methods',
        'methods': 'methods',
        'methodology': 'methodology',
        'result': 'results',
        'results': 'results',
        'discussion': 'discussion',
        'conclusion': 'conclusion',
        'conclusions': 'conclusions',
        'background': 'background',
        'related': 'related work',
        'experiment': 'experiments',
        'experiments': 'experiments',
        'evaluation': 'evaluation',
        'appendix': 'appendix',
        'supplement': 'supplementary material',
    }
    for key, section in section_mappings.items():
        if key in name_lower:
            return section
    return None


def _analyze_tool_use(tool_name: str, tool_input: Dict[str, Any], current_stage: str) -> tuple:
    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    command = tool_input.get("command", "")
    filename = Path(file_path).name if file_path else ""
    doc_type = _detect_document_type(file_path)
    
    if tool_name.lower() == "read":
        if ".bib" in file_path:
            return ("writing", f"Reading bibliography: {filename}")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Reading {section} section")
            return ("writing", f"Reading {filename}")
        elif ".pdf" in file_path:
            return ("research", f"Analyzing PDF: {filename}")
        elif ".csv" in file_path:
            return ("research", f"Loading data from {filename}")
        elif ".json" in file_path:
            return ("research", f"Reading configuration: {filename}")
        elif ".md" in file_path:
            return ("planning", f"Reading {filename}")
        elif file_path:
            return (current_stage, f"Reading {filename}")
        return None
    
    elif tool_name.lower() == "write":
        if ".bib" in file_path:
            return ("writing", f"Creating bibliography with references")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Writing {section} section")
            elif "main" in filename.lower():
                return ("writing", f"Creating main {doc_type} structure")
            elif current_idx < stage_order.index("writing"):
                return ("writing", f"Writing {doc_type}: {filename}")
            else:
                return ("compilation", f"Updating {filename}")
        elif ".md" in file_path:
            if "progress" in filename.lower():
                return ("writing", "Updating progress log")
            elif "readme" in filename.lower():
                return ("complete", "Creating documentation")
            return ("writing", f"Writing {filename}")
        elif ".sty" in file_path:
            return ("writing", f"Creating style file: {filename}")
        elif ".cls" in file_path:
            return ("writing", f"Creating document class: {filename}")
        elif file_path:
            return (current_stage, f"Creating {filename}")
        return None
    
    elif tool_name.lower() == "edit":
        if ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Refining {section} section")
            return ("writing", f"Editing {filename}")
        elif ".bib" in file_path:
            return ("writing", "Updating bibliography")
        elif file_path:
            return (current_stage, f"Editing {filename}")
        return None
    
    elif tool_name.lower() == "bash":
        if "pdflatex" in command:
            if "-output-directory" in command:
                return ("compilation", "Compiling PDF with output directory")
            return ("compilation", "Compiling LaTeX to PDF")
        elif "latexmk" in command:
            return ("compilation", "Running full LaTeX compilation pipeline")
        elif "bibtex" in command:
            return ("compilation", "Processing bibliography citations")
        elif "makeindex" in command:
            return ("compilation", "Building document index")
        elif "mkdir" in command:
            if "writing_outputs" in command or "output" in command.lower():
                return ("initialization", "Creating output directory")
            elif "figures" in command.lower():
                return ("initialization", "Setting up figures directory")
            elif "drafts" in command.lower():
                return ("initialization", "Setting up drafts directory")
            return ("initialization", "Creating directory structure")
        elif "cp " in command:
            if ".pdf" in command:
                return ("complete", "Copying final PDF to output")
            elif ".tex" in command:
                return ("complete", "Archiving LaTeX source")
            return ("complete", "Organizing files")
        elif "mv " in command:
            return ("complete", "Moving files to final location")
        elif "ls " in command or "cat " in command:
            return None
        elif command:
            cmd_preview = command.split()[0] if command.split() else command[:30]
            return (current_stage, f"Running {cmd_preview}")
        return None
    
    elif "research" in tool_name.lower() or "lookup" in tool_name.lower():
        query_text = tool_input.get("query", "")
        if query_text:
            truncated = query_text[:50] + "..." if len(query_text) > 50 else query_text
            return ("research", f"Searching: {truncated}")
        return ("research", "Searching literature databases")
    
    elif "search" in tool_name.lower() or "web" in tool_name.lower():
        query_text = tool_input.get("query", tool_input.get("search_term", ""))
        if query_text:
            truncated = query_text[:40] + "..." if len(query_text) > 40 else query_text
            return ("research", f"Web search: {truncated}")
        return ("research", "Searching online resources")
    
    return None



def _find_output_from_written_files(files_written: list, start_time: float) -> Optional[Path]:
    """
    Fallback: when the AI ignores cwd and writes somewhere else, derive the
    actual output directory from the file paths it actually wrote to.
    Looks for the nearest ancestor named like a timestamped paper folder
    (e.g. 20260411_143000_deep_learning_abstract) inside any writing_outputs dir.
    """
    for file_path in files_written:
        try:
            p = Path(str(file_path)).resolve()
            # Walk up the tree looking for a writing_outputs parent
            for parent in p.parents:
                if parent.name == "writing_outputs" and parent.exists():
                    # Return the most recent child dir
                    result = _find_most_recent_output(parent, start_time)
                    if result:
                        return result
        except Exception:
            continue
    return None


def _find_most_recent_output(output_folder: Path, start_time: float) -> Optional[Path]:
    try:
        output_dirs = [d for d in output_folder.iterdir() if d.is_dir()]
        if not output_dirs:
            return None
        
        recent_dirs = [
            d for d in output_dirs 
            if d.stat().st_mtime >= start_time - 5
        ]
        
        if not recent_dirs:
            recent_dirs = output_dirs
        
        most_recent = max(recent_dirs, key=lambda d: d.stat().st_mtime)
        return most_recent
    except Exception:
        return None


def _build_paper_result(paper_dir: Path, file_info: Dict[str, Any]) -> PaperResult:
    tex_file = file_info['tex_final'] or (file_info['tex_drafts'][0] if file_info['tex_drafts'] else None)
    
    title = extract_title_from_tex(tex_file)
    word_count = count_words_in_tex(tex_file)
    
    topic = ""
    parts = paper_dir.name.split('_', 2)
    if len(parts) >= 3:
        topic = parts[2].replace('_', ' ')
    
    metadata = PaperMetadata(
        title=title,
        created_at=datetime.fromtimestamp(paper_dir.stat().st_ctime).isoformat() + "Z",
        topic=topic,
        word_count=word_count,
    )
    
    files = PaperFiles(
        pdf_final=file_info['pdf_final'],
        tex_final=file_info['tex_final'],
        pdf_drafts=file_info['pdf_drafts'],
        tex_drafts=file_info['tex_drafts'],
        bibliography=file_info['bibliography'],
        figures=file_info['figures'],
        data=file_info['data'],
        progress_log=file_info['progress_log'],
        summary=file_info['summary'],
    )
    
    citation_count = count_citations_in_bib(file_info['bibliography'])
    citation_style = extract_citation_style(file_info['bibliography'])
    
    citations = {
        'count': citation_count,
        'style': citation_style,
        'file': file_info['bibliography'],
    }
    
    status = "success"
    compilation_success = file_info['pdf_final'] is not None

    if not compilation_success:
        # Any output file (tex, md, bib, figures, data) counts as partial not failed
        has_any_output = any([
            file_info.get('tex_final'),
            file_info.get('tex_drafts'),
            file_info.get('bibliography'),
            file_info.get('figures'),
            file_info.get('data'),
            file_info.get('progress_log'),
            file_info.get('summary'),
        ])
        status = "partial" if has_any_output else "failed"
    
    result = PaperResult(
        status=status,
        paper_directory=str(paper_dir),
        paper_name=paper_dir.name,
        metadata=metadata,
        files=files,
        citations=citations,
        figures_count=len(file_info['figures']),
        compilation_success=compilation_success,
        errors=[],
    )
    
    return result


def _create_error_result(error_message: str) -> Dict[str, Any]:
    result = PaperResult(
        status="failed",
        paper_directory="",
        paper_name="",
        errors=[error_message],
    )
    return result.to_dict()
