import re
from typing import Optional


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    source_title: str = "",
    page_number: Optional[int] = None,
) -> list[dict]:
    sections = _split_by_headers(text)
    chunks = []
    current_chunk = ""
    current_section = ""
    chunk_index = 0

    for section in sections:
        section_header = section.get("header", "")
        section_text = section.get("text", "")

        if section_header:
            current_section = section_header

        paragraphs = section_text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            tokens_est = len(para) // 4

            if len(current_chunk) // 4 + tokens_est > chunk_size:
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "section": current_section,
                        "page": page_number,
                        "position": chunk_index,
                        "source_title": source_title,
                        "char_count": len(current_chunk.strip()),
                        "token_count": len(current_chunk.strip()) // 4,
                    })
                    chunk_index += 1

                overlap_text = current_chunk[-chunk_overlap * 4:] if chunk_overlap > 0 else ""
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "section": current_section,
            "page": page_number,
            "position": chunk_index,
            "source_title": source_title,
            "char_count": len(current_chunk.strip()),
            "token_count": len(current_chunk.strip()) // 4,
        })

    return chunks


def _split_by_headers(text: str) -> list[dict]:
    pattern = r'^(#{1,3}\s+.+)$'
    lines = text.split('\n')
    sections = []
    current_header = ""
    current_text = []

    for line in lines:
        if re.match(pattern, line):
            if current_text:
                sections.append({"header": current_header, "text": "\n".join(current_text)})
            current_header = line.strip("# ").strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections.append({"header": current_header, "text": "\n".join(current_text)})

    return sections
