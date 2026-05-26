
# import re

# # What we remove:
# #   - Unicode control characters (except newlines/tabs)          → corrupt tokens in embedder
# #   - Runs of 3+ blank lines                                     → wastes chunk space
# #   - Lines that are only whitespace                             → no semantic content
# #   - Null bytes                                                  → breaks parsers
# #   - Common boilerplate patterns (page numbers, header repeats) → dilutes retrieval
# #
# # What we PRESERVE:
# #   - Headings and section labels         → critical for retrieval scoping
# #   - Bullet / numbered list markers      → structure signals
# #   - Domain terms, acronyms              → high-value retrieval tokens
# #   - Tables (as text)                    → structured facts
# #   - Line breaks between paragraphs      → chunk boundary signals
# #   - Single dashes and hyphens           → compound terms


# _CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# _FENCED_CODE = re.compile(r'```.*?```', re.DOTALL)
# _TILDE_CODE = re.compile(r'~~~.*?~~~', re.DOTALL)
# _INLINE_CODE = re.compile(r'`([^`\n]+)`')
# _HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
# _MULTI_BLANK   = re.compile(r'\n{3,}')
# _PAGE_NUM      = re.compile(r'^\s*[-–—]?\s*\d+\s*[-–—]?\s*$', re.MULTILINE)
# _TABLE_DIVIDER = re.compile(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$', re.MULTILINE)
# _REPEATED_SEP  = re.compile(r'^\s*([=\-_*#~])(?:\s*\1){4,}\s*$', re.MULTILINE)
# _WHITESPACE    = re.compile(r'[ \t]{2,}')   # collapse horizontal whitespace only
# _BROKEN_LINE   = re.compile(r'(?<![.!?:]\s)\n(?=[a-z])')  # repair mid-sentence line breaks


# def _looks_like_ascii_art(line: str) -> bool:
#     """Return True when a line is mostly diagram noise instead of prose."""
#     stripped = line.strip()
#     if len(stripped) < 6:
#         return False

#     alnum = sum(ch.isalnum() for ch in stripped)
#     noise = sum(ch in "|/\\+-=_*#<>:^~[](){}" for ch in stripped)

#     if alnum == 0 and noise >= 4:
#         return True
#     if noise >= 8 and alnum <= 2:
#         return True
#     if stripped.count("|") >= 2 and alnum <= 3:
#         return True
#     if set(stripped) <= set("-=*_#~|+ "):
#         return True
#     return False


# def _flatten_tables(text: str) -> str:
#     """Convert markdown-style tables into plain text rows and drop divider rows."""
#     lines = []
#     for raw_line in text.splitlines():
#         line = raw_line.rstrip()
#         if not line.strip():
#             lines.append("")
#             continue
#         if _TABLE_DIVIDER.match(line):
#             continue
#         if "|" in line:
#             cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
#             cells = [cell for cell in cells if cell]
#             if cells:
#                 line = "  ".join(cells)
#         if _looks_like_ascii_art(line):
#             continue
#         lines.append(line)
#     return "\n".join(lines)


# def clean(text: str) -> str:
#     if not text:
#         return ""

#     # 1. Remove comments and code blocks that usually hurt retrieval quality
#     text = _HTML_COMMENT.sub('', text)
#     text = _FENCED_CODE.sub('', text)
#     text = _TILDE_CODE.sub('', text)

#     # 2. Strip null bytes and control chars that corrupt tokenizers
#     text = _CONTROL_CHARS.sub('', text)

#     # 3. Preserve inline code content but drop the markers
#     text = _INLINE_CODE.sub(r'\1', text)

#     # 4. Remove lone page-number lines produced by PDF extractors
#     text = _PAGE_NUM.sub('', text)

#     # 5. Repair broken lines: join lines that continue a sentence
#     #    "The report found that\nrevenues increased" → single line
#     #    Preserves paragraph breaks (blank line before capital)
#     text = _BROKEN_LINE.sub(' ', text)

#     # 6. Flatten tables and remove diagram-like separators
#     text = _flatten_tables(text)

#     # 7. Collapse runs of spaces/tabs to single space (not newlines — they're structural)
#     text = _WHITESPACE.sub(' ', text)

#     # 8. Collapse excessive blank lines to at most two (paragraph separator)
#     text = _MULTI_BLANK.sub('\n\n', text)

#     # 9. Strip leading/trailing whitespace per line
#     lines = [line.rstrip() for line in text.splitlines()]
#     text = "\n".join(lines)

#     return text.strip()

import re
import unicodedata

# Preserve structure, remove only noise.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCED_CODE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_TILDE_CODE = re.compile(r"~~~([^\n`]*)\n(.*?)~~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_BLANK = re.compile(r"\n{3,}")
_PAGE_NUM = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE)
_TABLE_DIVIDER = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$", re.MULTILINE)
_REPEATED_SEP = re.compile(r"^\s*([=\-_*#~])(?:\s*\1){4,}\s*$", re.MULTILINE)
_WHITESPACE = re.compile(r"[ \t]{2,}")
_BROKEN_LINE = re.compile(r"(?<![.!?:]\s)\n(?=[a-z])")


def _code_block_repl(tag: str):
    def repl(match: re.Match) -> str:
        lang = (match.group(1) or "").strip()
        body = (match.group(2) or "").strip("\n")
        header = f"{tag} {lang}".strip()
        return f"\n{header}\n{body}\nEND_{tag}\n"
    return repl


def _looks_like_ascii_art(line: str) -> bool:
    """Return True when a line is mostly diagram noise instead of prose."""
    stripped = line.strip()
    if len(stripped) < 6:
        return False

    alnum = sum(ch.isalnum() for ch in stripped)
    noise = sum(ch in "|/\\+-=_*#<>:^~[](){}" for ch in stripped)

    if alnum == 0 and noise >= 4:
        return True
    if noise >= 8 and alnum <= 2:
        return True
    if stripped.count("|") >= 2 and alnum <= 3:
        return True
    if set(stripped) <= set("-=*_#~|+ "):
        return True
    return False


def _flatten_tables(text: str) -> str:
    """Convert markdown-style tables into plain text rows and drop divider rows."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            lines.append("")
            continue

        if _TABLE_DIVIDER.match(line):
            continue

        if "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if cells:
                line = " | ".join(cells)

        if _looks_like_ascii_art(line):
            continue

        if _REPEATED_SEP.match(line):
            continue

        lines.append(line)

    return "\n".join(lines)


def clean(text: str) -> str:
    """
    Retrieval-safe cleaning:
    - removes real noise
    - preserves headings, lists, tables, and code content
    - keeps paragraph breaks for chunking
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # 1) Remove comments but preserve code content.
    text = _HTML_COMMENT.sub("", text)
    text = _FENCED_CODE.sub(_code_block_repl("CODE_BLOCK"), text)
    text = _TILDE_CODE.sub(_code_block_repl("TILDE_BLOCK"), text)

    # 2) Remove corrupting characters.
    text = _CONTROL_CHARS.sub("", text)

    # 3) Preserve inline code content, drop backticks.
    text = _INLINE_CODE.sub(r"\1", text)

    # 4) Remove isolated page numbers.
    text = _PAGE_NUM.sub("", text)

    # 5) Repair common PDF line wrapping.
    text = _BROKEN_LINE.sub(" ", text)

    # 6) Keep table semantics but remove divider rows and ASCII art.
    text = _flatten_tables(text)

    # 7) Compact horizontal whitespace only.
    text = _WHITESPACE.sub(" ", text)

    # 8) Keep paragraph boundaries, but limit blank lines.
    text = _MULTI_BLANK.sub("\n\n", text)

    # 9) Strip trailing spaces per line.
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()