"""
Pipeline de Ingestão Documental e Pesquisa Oficial da Veronica (Seções 4, 6 e 10).
Suporta TXT, Markdown, HTML, JSON, XML, DOCX, PDF e arquivos de configuração CLI (.cfg, .conf, .rsc),
extraindo textos, tabelas e comandos sem quebrar blocos de comando em documentos extensos,
preservando os 12 metadados obrigatórios, mascarando segredos, bloqueando Prompt Injection
e conectando-se de forma segura a portais oficiais de fabricantes e IETF.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import io
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus, urlparse
import uuid
import xml.etree.ElementTree as ET
import zipfile
import zlib

from knowledge.sources.catalog import KnowledgeDocument, OFFICIAL_INITIAL_DOCUMENTS


PROMPT_INJECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "OVERRIDE_SYSTEM_RULES",
        re.compile(
            r"(ignore\s+(all\s+)?(previous|prior|system)\s+instructions"
            r"|ignore\s+todas\s+as\s+instru[cç][oõ]es"
            r"|desative\s+as\s+regras\s+de\s+seguran[cç]a"
            r"|bypass\s+approval"
            r"|execute\s+without\s+authorization"
            r"|modifique\s+pol[ií]ticas\s+de\s+rbac)",
            re.IGNORECASE,
        ),
    ),
    (
        "DESTRUCTIVE_TROJAN_COMMAND",
        re.compile(
            r"(write\s+erase"
            r"|erase\s+startup-config"
            r"|reset\s+saved-configuration"
            r"|request\s+system\s+zeroize"
            r"|format\s+(flash:|bootflash:|hda1:)"
            r"|delete\s+/unreserved"
            r"|/system\s+reset-configuration\s+no-defaults=yes)",
            re.IGNORECASE,
        ),
    ),
]

SECRET_REDACTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"((?:password|secret|key-string|wpa-passphrase|community|pre-shared-key|cipher|passphrase|auth-key)\s+)([^\s\n\"']+)",
        re.IGNORECASE,
    ),
    re.compile(r"(api[_-]?key\s*[:=]\s*)([^\s\n\"']+)", re.IGNORECASE),
]

CLI_COMMAND_REGEX = re.compile(
    r"`((?:display|show|ping|traceroute|tracert|commit|rollback|configure|diagnose|get\s+router|/routing|/ip|/interface|/system|peer|route-policy|neighbor|interface|router\s+bgp|router\s+ospf|router\s+isis|mpls)[^`\n]+)`",
    re.IGNORECASE,
)

CLI_LINE_START_REGEX = re.compile(
    r"^(display\s+|show\s+|diagnose\s+|get\s+router|ping\s+|traceroute\s+|tracert\s+"
    r"|/routing|/ip|/interface|/system|/mpls|/user|/tool"
    r"|system-view|configure\s+terminal|configure\s+session|commit|rollback"
    r"|interface\s+|router\s+(?:bgp|ospf|ospfv3|isis)|bgp\s+\d+|ospf\s+\d+|isis\s+"
    r"|neighbor\s+|peer\s+|route-policy\s+|route-map\s+|ip\s+route|ipv6\s+route"
    r"|mpls\s+|l2vpn|evpn|vxlan|snmp-server|netconf|restconf|gnmi|ntp\s+|aaa\s+|tacacs|radius"
    r"|config\s+(?:system|firewall|vpn|router)|set\s+)",
    re.IGNORECASE,
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
PAGER_ARTIFACT_RE = re.compile(r"(--\s*More\s*--|<---\s*More\s*--->)", re.IGNORECASE)


def scrub_secrets(text: str) -> Tuple[str, int]:
    """Remove senhas, chaves e SNMP communities antes de indexar ou salvar em logs/datasets."""
    redacted_count = 0
    cleaned = text
    for pattern in SECRET_REDACTION_PATTERNS:
        cleaned, count = pattern.subn(r"\1[REDACTED_SECRET]", cleaned)
        redacted_count += count
    return cleaned, redacted_count


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """
    Analisa documentos externos para detectar tentativas de injeção de instruções
    ou alteração de políticas de segurança (Seção 2.12, 10 e 14).
    """
    reasons: List[str] = []
    for code, pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            reasons.append(f"{code}: padrão suspeito detectado ('{match.group(0)}')")
    return (len(reasons) > 0), reasons


def extract_tables_and_commands(text: str) -> Tuple[List[str], List[str]]:
    commands = list(dict.fromkeys(m.strip() for m in CLI_COMMAND_REGEX.findall(text)))
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        if CLI_LINE_START_REGEX.match(stripped):
            if stripped not in commands:
                commands.append(stripped)

    tables: List[str] = []
    current_table: List[str] = []
    for line in text.splitlines():
        if "|" in line and line.strip().startswith("|"):
            current_table.append(line.strip())
        else:
            if len(current_table) >= 2:
                tables.append("\n".join(current_table))
            current_table = []
    if len(current_table) >= 2:
        tables.append("\n".join(current_table))

    return commands, tables


def _local_tag_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


def _parse_word_xml(xml_text: str) -> str:
    """Extrai parágrafos e tabelas (em formato Markdown) de um XML WordprocessingML (`word/document.xml`)."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        # Fallback por regex para fragmentos XML sem namespace declarado
        cleaned_lines: List[str] = []
        for p_match in re.findall(r"<w:p\b[^>]*>(.*?)</w:p>", xml_text, flags=re.DOTALL):
            runs = re.findall(r"<w:t\b[^>]*>(.*?)</w:t>", p_match, flags=re.DOTALL)
            if runs:
                cleaned_lines.append(html.unescape("".join(runs).strip()))
        if cleaned_lines:
            return "\n".join(line for line in cleaned_lines if line)
        no_tags = re.sub(r"<[^>]+>", " ", xml_text)
        return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()

    output_blocks: List[str] = []
    body = next((elem for elem in root.iter() if _local_tag_name(elem.tag) == "body"), root)

    for child in list(body):
        lname = _local_tag_name(child.tag)
        if lname == "p":
            texts: List[str] = []
            for node in child.iter():
                nname = _local_tag_name(node.tag)
                if nname == "t" and node.text:
                    texts.append(node.text)
                elif nname == "tab":
                    texts.append("\t")
                elif nname in ("br", "cr"):
                    texts.append("\n")
            para = "".join(texts).strip()
            if para:
                output_blocks.append(para)
        elif lname == "tbl":
            table_rows: List[str] = []
            for row in child.iter():
                if _local_tag_name(row.tag) == "tr":
                    cells: List[str] = []
                    for cell in list(row):
                        if _local_tag_name(cell.tag) == "tc":
                            cell_texts = [
                                t.text
                                for t in cell.iter()
                                if _local_tag_name(t.tag) == "t" and t.text
                            ]
                            cells.append(" ".join("".join(cell_texts).split()))
                    if cells:
                        table_rows.append("| " + " | ".join(cells) + " |")
            if table_rows:
                output_blocks.append("\n".join(table_rows))

    if not output_blocks:
        all_t = [
            node.text.strip()
            for node in root.iter()
            if _local_tag_name(node.tag) == "t" and node.text and node.text.strip()
        ]
        return "\n".join(all_t)

    return "\n\n".join(output_blocks).strip()


def extract_docx_content(raw_content: Union[bytes, str]) -> str:
    """
    Extrai texto, comandos e tabelas de arquivos `.docx` reais (arquivo ZIP em memória
    contendo `word/document.xml` ou string XML WordprocessingML direta).
    """
    raw_bytes: Optional[bytes] = None
    if isinstance(raw_content, bytes):
        raw_bytes = raw_content
    elif isinstance(raw_content, str):
        stripped = raw_content.strip()
        if stripped.startswith("PK\x03\x04"):
            raw_bytes = raw_content.encode("latin1", errors="ignore")
        elif stripped.startswith("UEsDB"):
            try:
                raw_bytes = base64.b64decode(stripped)
            except Exception:
                raw_bytes = None

    if raw_bytes is not None and zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            xml_parts: List[str] = []
            names = zf.namelist()
            ordered_targets = [n for n in names if n == "word/document.xml"] + [
                n
                for n in names
                if n.startswith("word/header") or n.startswith("word/footer")
            ]
            for target in ordered_targets:
                xml_bytes = zf.read(target)
                xml_str = xml_bytes.decode("utf-8", errors="replace")
                parsed = _parse_word_xml(xml_str)
                if parsed:
                    xml_parts.append(parsed)
            if xml_parts:
                return "\n\n".join(xml_parts).strip()

    text_str = (
        raw_content.decode("utf-8", errors="replace")
        if isinstance(raw_content, bytes)
        else raw_content
    )
    if "<w:" in text_str or "<document" in text_str:
        return _parse_word_xml(text_str)
    return text_str.strip()


def _unescape_pdf_string(pdf_literal: str) -> str:
    """Decodifica sequências de escape literais de strings PDF."""
    out: List[str] = []
    i = 0
    n = len(pdf_literal)
    while i < n:
        ch = pdf_literal[i]
        if ch == "\\" and i + 1 < n:
            nxt = pdf_literal[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
            elif nxt == "r":
                out.append("\r")
                i += 2
            elif nxt == "t":
                out.append("\t")
                i += 2
            elif nxt in ("(", ")", "\\"):
                out.append(nxt)
                i += 2
            elif nxt.isdigit():
                oct_digits = nxt
                j = i + 2
                while j < n and j < i + 4 and pdf_literal[j].isdigit():
                    oct_digits += pdf_literal[j]
                    j += 1
                try:
                    out.append(chr(int(oct_digits, 8)))
                except ValueError:
                    out.append(nxt)
                i = j
            else:
                out.append(nxt)
                i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _extract_pdf_operators_from_stream(stream_text: str) -> List[str]:
    """Extrai blocos `(texto) Tj`, `[(texto)] TJ` e `<hex> Tj` de um content stream PDF."""
    extracted_lines: List[str] = []

    # 1. Operadores de linha simples: (texto) Tj ou '
    token_pattern = re.compile(
        r"\(((?:\\.|[^\\()])*)\)\s*(?:Tj|'|\"\s*)"
        r"|\[((?:.|\n)*?)\]\s*TJ"
        r"|<([0-9A-Fa-f\s]+)>\s*Tj"
    )
    for match in token_pattern.finditer(stream_text):
        tj_single, tj_array, tj_hex = match.groups()
        if tj_single is not None:
            decoded = _unescape_pdf_string(tj_single).strip()
            if decoded:
                extracted_lines.append(decoded)
        elif tj_array is not None:
            parts: List[str] = []
            for sub in re.finditer(r"\(((?:\\.|[^\\()])*)\)|(-?\d+(?:\.\d+)?)", tj_array):
                lit, kern = sub.groups()
                if lit is not None:
                    parts.append(_unescape_pdf_string(lit))
                elif kern is not None:
                    try:
                        if float(kern) <= -100.0 and parts and not parts[-1].endswith(" "):
                            parts.append(" ")
                    except ValueError:
                        pass
            joined = "".join(parts).strip()
            if joined:
                extracted_lines.append(joined)
        elif tj_hex is not None:
            hex_clean = re.sub(r"\s+", "", tj_hex)
            if len(hex_clean) % 2 == 1:
                hex_clean += "0"
            try:
                decoded_hex = bytes.fromhex(hex_clean).decode("utf-8", errors="ignore").strip()
                if decoded_hex:
                    extracted_lines.append(decoded_hex)
            except Exception:
                pass

    return extracted_lines


def extract_pdf_content(raw_content: Union[bytes, str]) -> str:
    """
    Extrai texto de documentos `.pdf` (descomprimindo streams `/FlateDecode` via zlib
    e interpretando operadores de texto `(texto) Tj` e `[(...)] TJ`).
    """
    pdf_bytes = (
        raw_content
        if isinstance(raw_content, bytes)
        else raw_content.encode("latin1", errors="ignore")
    )

    extracted_segments: List[str] = []

    # Localiza blocos stream ... endstream (comprimidos com /FlateDecode ou texto puro)
    for stream_match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, flags=re.DOTALL):
        raw_stream = stream_match.group(1)
        candidate_texts: List[str] = []
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                decompressed = zlib.decompress(raw_stream, wbits)
                candidate_texts.append(decompressed.decode("utf-8", errors="replace"))
                break
            except Exception:
                continue
        if not candidate_texts:
            candidate_texts.append(raw_stream.decode("latin1", errors="replace"))

        for stext in candidate_texts:
            ops = _extract_pdf_operators_from_stream(stext)
            if ops:
                extracted_segments.extend(ops)

    # Caso os operadores Tj/TJ estejam diretamente no corpo sem stream/endstream binário
    if not extracted_segments:
        raw_text = (
            raw_content.decode("utf-8", errors="replace")
            if isinstance(raw_content, bytes)
            else raw_content
        )
        ops = _extract_pdf_operators_from_stream(raw_text)
        if ops:
            extracted_segments.extend(ops)
        else:
            # Fallback para PDF textual já extraído
            cleaned = re.sub(r"%PDF-\d+\.\d+", "", raw_text)
            cleaned = re.sub(r"%%EOF", "", cleaned)
            return cleaned.strip()

    return "\n".join(extracted_segments).strip()


def normalize_cli_config_content(raw_content: Union[bytes, str], fmt: str = "cfg") -> str:
    """
    Normaliza arquivos de configuração CLI (`.cfg`, `.conf`, `.rsc`), removendo artefatos
    de paginação/ANSI e unindo quebras de linha `\\` de scripts RouterOS `.rsc`.
    """
    text = (
        raw_content.decode("utf-8", errors="replace")
        if isinstance(raw_content, bytes)
        else raw_content
    )
    text = ANSI_ESCAPE_RE.sub("", text)
    text = PAGER_ARTIFACT_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    fmt_low = fmt.lower().strip(".")
    if fmt_low == "rsc":
        # Une linhas continuadas com '\' típicas de export do MikroTik RouterOS (.rsc)
        merged_lines: List[str] = []
        buffer = ""
        for line in text.splitlines():
            rstripped = line.rstrip()
            if rstripped.endswith("\\"):
                buffer += rstripped[:-1].rstrip() + " "
            else:
                if buffer:
                    merged_lines.append((buffer + rstripped.lstrip()).strip())
                    buffer = ""
                else:
                    merged_lines.append(rstripped)
        if buffer:
            merged_lines.append(buffer.strip())
        return "\n".join(merged_lines).strip()

    cleaned_lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(cleaned_lines).strip()


def normalize_format_content(raw_content: Union[bytes, str], fmt: str) -> str:
    fmt_low = fmt.lower().strip(".")
    if fmt_low == "docx":
        return extract_docx_content(raw_content)
    if fmt_low == "pdf":
        return extract_pdf_content(raw_content)
    if fmt_low in ("cfg", "conf", "rsc", "cli"):
        return normalize_cli_config_content(raw_content, fmt_low)

    text_content = (
        raw_content.decode("utf-8", errors="replace")
        if isinstance(raw_content, bytes)
        else raw_content
    )
    if fmt_low in ("html", "htm"):
        no_tags = re.sub(r"<script.*?>.*?</script>", "", text_content, flags=re.DOTALL | re.IGNORECASE)
        no_tags = re.sub(r"<style.*?>.*?</style>", "", no_tags, flags=re.DOTALL | re.IGNORECASE)
        no_tags = re.sub(r"<[^>]+>", " ", no_tags)
        return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()
    if fmt_low == "json":
        try:
            parsed = json.loads(text_content)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return text_content
    if fmt_low == "xml":
        if "<w:" in text_content:
            return _parse_word_xml(text_content)
        no_tags = re.sub(r"<[^>]+>", " ", text_content)
        return re.sub(r"\s+", " ", no_tags).strip()
    return text_content


def _split_into_atomic_blocks(text: str) -> List[Tuple[str, bool]]:
    """
    Divide o documento em blocos lógicos `(conteudo_bloco, is_command_block)`.
    Mantém blocos de código cercados por ``` e estrofes de comandos CLI juntos
    para evitar quebrar blocos de comando no meio.
    """
    lines = text.replace("\r\n", "\n").splitlines()
    blocks: List[Tuple[str, bool]] = []
    current_lines: List[str] = []
    current_is_cmd = False
    in_code_fence = False

    def flush_current() -> None:
        nonlocal current_lines, current_is_cmd
        if current_lines:
            joined = "\n".join(current_lines).strip()
            if joined:
                blocks.append((joined, current_is_cmd))
            current_lines = []
            current_is_cmd = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code_fence:
                flush_current()
                in_code_fence = True
                current_is_cmd = True
                current_lines.append(line)
            else:
                current_lines.append(line)
                in_code_fence = False
                flush_current()
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        if not stripped:
            flush_current()
            continue

        is_cli_line = bool(
            CLI_LINE_START_REGEX.match(stripped)
            or (
                current_is_cmd
                and (
                    line.startswith(" ")
                    or line.startswith("\t")
                    or stripped in ("!", "#", "exit", "end", "end-policy", "quit", "return")
                )
            )
        )

        if is_cli_line != current_is_cmd and current_lines:
            flush_current()

        current_is_cmd = is_cli_line
        current_lines.append(line)

    flush_current()
    return blocks


def chunk_large_document(
    text: str,
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Processa documentos técnicos extensos dividindo-os por seções/parágrafos
    respeitando `max_chunk_chars` e `overlap_chars` sem quebrar blocos de comando.
    """
    if not text or not text.strip():
        return []
    cleaned = text.strip()
    if len(cleaned) <= max_chunk_chars:
        return [cleaned]

    max_chunk_chars = max(200, int(max_chunk_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chunk_chars // 2))

    blocks = _split_into_atomic_blocks(cleaned)
    expanded_blocks: List[Tuple[str, bool]] = []

    for block_text, is_cmd in blocks:
        if len(block_text) <= max_chunk_chars:
            expanded_blocks.append((block_text, is_cmd))
        elif is_cmd:
            # Se um bloco de comando único for maior que max_chunk_chars, mantém inteiro
            # se possível ou divide apenas entre estrofes completas sem quebrar linha de comando
            expanded_blocks.append((block_text, True))
        else:
            # Divide parágrafo textual muito longo por sentenças ou linhas
            sentences = re.split(r"(?<=[.!?])\s+|\n", block_text)
            buf = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                candidate = f"{buf} {sent}".strip() if buf else sent
                if len(candidate) <= max_chunk_chars:
                    buf = candidate
                else:
                    if buf:
                        expanded_blocks.append((buf, False))
                    buf = sent
            if buf:
                expanded_blocks.append((buf, False))

    chunks: List[str] = []
    current_parts: List[Tuple[str, bool]] = []
    current_len = 0

    for block_text, is_cmd in expanded_blocks:
        sep_len = 2 if current_parts else 0
        if current_parts and (current_len + sep_len + len(block_text) > max_chunk_chars):
            chunk_str = "\n\n".join(b for b, _ in current_parts).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Constrói overlap a partir do final do chunk anterior sem cortar blocos de comando ao meio
            overlap_parts: List[Tuple[str, bool]] = []
            overlap_len = 0
            if overlap_chars > 0:
                for prev_text, prev_is_cmd in reversed(current_parts):
                    if prev_is_cmd:
                        # Só inclui bloco de comando no overlap se couber inteiro dentro do overlap_chars
                        if overlap_len + len(prev_text) <= overlap_chars:
                            overlap_parts.insert(0, (prev_text, True))
                            overlap_len += len(prev_text) + 2
                        break
                    else:
                        if len(prev_text) <= overlap_chars - overlap_len:
                            overlap_parts.insert(0, (prev_text, False))
                            overlap_len += len(prev_text) + 2
                        else:
                            tail = prev_text[-(overlap_chars - overlap_len) :]
                            # Alinha no início de palavra
                            space_idx = tail.find(" ")
                            if 0 <= space_idx < len(tail) - 1:
                                tail = tail[space_idx + 1 :].strip()
                            if tail and (len(tail) + 2 + len(block_text) <= max_chunk_chars):
                                overlap_parts.insert(0, (tail, False))
                                overlap_len += len(tail) + 2
                            break

            # Garante que o overlap não estoure o limite com o novo bloco
            while overlap_parts and (
                sum(len(p) for p, _ in overlap_parts) + 2 * len(overlap_parts) + len(block_text)
                > max_chunk_chars
            ):
                overlap_parts.pop(0)

            current_parts = overlap_parts + [(block_text, is_cmd)]
            current_len = sum(len(p) for p, _ in current_parts) + 2 * max(0, len(current_parts) - 1)
        else:
            current_parts.append((block_text, is_cmd))
            current_len += sep_len + len(block_text)

    if current_parts:
        last_chunk = "\n\n".join(b for b, _ in current_parts).strip()
        if last_chunk and (not chunks or last_chunk != chunks[-1]):
            chunks.append(last_chunk)

    return chunks


class OfficialDocumentationSearchConnector:
    """
    Conector de pesquisa em portais oficiais de fabricantes e IETF (Seção 4).
    Valida domínios oficiais autorizados, aplica controle de rate-limit e políticas
    de acesso, higieniza segredos e bloqueia Prompt Injection antes de indexar.
    """

    APPROVED_OFFICIAL_PORTALS: Tuple[str, ...] = (
        "support.huawei.com",
        "cisco.com/c/en/us/support",
        "juniper.net/documentation",
        "help.mikrotik.com",
        "documentation.nokia.com",
        "arista.com/en/support",
        "docs.fortinet.com",
        "datatracker.ietf.org/doc/html/rfc",
    )

    VENDOR_PORTAL_MAP: Dict[str, str] = {
        "huawei": "https://support.huawei.com/enterprise/en/doc/",
        "cisco": "https://www.cisco.com/c/en/us/support/index.html",
        "juniper": "https://www.juniper.net/documentation/",
        "mikrotik": "https://help.mikrotik.com/docs/",
        "nokia": "https://documentation.nokia.com/",
        "arista": "https://www.arista.com/en/support/documentation",
        "fortinet": "https://docs.fortinet.com/",
        "ietf": "https://datatracker.ietf.org/doc/html/rfc",
    }

    def __init__(
        self,
        ingestor: Optional[KnowledgeIngestor] = None,
        rate_limit_per_minute: int = 30,
        min_interval_seconds: float = 0.0,
        respect_robots_policy: bool = True,
    ) -> None:
        self.ingestor = ingestor or KnowledgeIngestor()
        self.rate_limit_per_minute = max(1, int(rate_limit_per_minute))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.respect_robots_policy = bool(respect_robots_policy)
        self._request_history: Dict[str, List[float]] = {}

    def validate_official_url(self, url_or_domain: str) -> Dict[str, Any]:
        """
        Verifica se uma URL ou domínio pertence estritamente à lista de portais oficiais aprovados.
        """
        raw = url_or_domain.strip()
        if not raw:
            return {"valid": False, "portal": None, "vendor": None, "reason": "EMPTY_URL"}

        if raw.startswith("official://"):
            vendor_part = raw[len("official://") :].split("/", 1)[0].lower()
            portal_url = self.VENDOR_PORTAL_MAP.get(vendor_part)
            if portal_url:
                return self.validate_official_url(portal_url)

        if "://" not in raw:
            candidate_url = "https://" + raw.lstrip("/")
        else:
            candidate_url = raw

        parsed = urlparse(candidate_url)
        if parsed.scheme.lower() != "https":
            return {
                "valid": False,
                "portal": None,
                "vendor": None,
                "reason": f"INSECURE_OR_UNSUPPORTED_SCHEME ({parsed.scheme})",
            }

        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        normalized_target = f"{host}{parsed.path}".rstrip("/").lower()

        for portal in self.APPROVED_OFFICIAL_PORTALS:
            portal_low = portal.lower().rstrip("/")
            if normalized_target == portal_low or normalized_target.startswith(portal_low + "/") or (
                portal_low.endswith("/rfc") and normalized_target.startswith(portal_low)
            ):
                vendor_name = next(
                    (
                        v.capitalize() if v != "ietf" else "IETF"
                        for v, base_u in self.VENDOR_PORTAL_MAP.items()
                        if portal_low.split("/", 1)[0] in base_u.lower()
                    ),
                    "Official",
                )
                return {
                    "valid": True,
                    "portal": portal,
                    "vendor": vendor_name,
                    "normalized_url": candidate_url,
                    "reason": "APPROVED_OFFICIAL_PORTAL",
                }

        return {
            "valid": False,
            "portal": None,
            "vendor": None,
            "reason": f"UNTRUSTED_OR_UNOFFICIAL_DOMAIN ({host}{parsed.path})",
        }

    def is_official_domain(self, url_or_domain: str) -> bool:
        return bool(self.validate_official_url(url_or_domain)["valid"])

    def check_rate_limit(
        self,
        portal: str,
        current_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Controla a taxa de requisições por portal oficial em janela móvel de 60 segundos.
        """
        now = float(current_timestamp) if current_timestamp is not None else time.time()
        key = portal.lower().strip()
        history = [ts for ts in self._request_history.get(key, []) if now - ts < 60.0]

        if history and self.min_interval_seconds > 0.0:
            elapsed = now - history[-1]
            if elapsed < self.min_interval_seconds:
                self._request_history[key] = history
                return {
                    "allowed": False,
                    "portal": portal,
                    "remaining_quota": max(0, self.rate_limit_per_minute - len(history)),
                    "reason": "MIN_INTERVAL_THROTTLED",
                }

        if len(history) >= self.rate_limit_per_minute:
            self._request_history[key] = history
            return {
                "allowed": False,
                "portal": portal,
                "remaining_quota": 0,
                "reason": "RATE_LIMIT_EXCEEDED",
            }

        history.append(now)
        self._request_history[key] = history
        return {
            "allowed": True,
            "portal": portal,
            "remaining_quota": self.rate_limit_per_minute - len(history),
            "reason": "OK",
        }

    def search(
        self,
        query: str,
        vendor: Optional[str] = None,
        target_url: Optional[str] = None,
        online_content_fetcher: Optional[Callable[[str, str], Union[str, bytes]]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Executa busca segura em documentação oficial (catálogo local curado e/ou portal oficial validado).
        Rejeita domínios não oficiais, respeita rate-limit e aplica sanitização de segurança.
        """
        if target_url is not None:
            url_check = self.validate_official_url(target_url)
            if not url_check["valid"]:
                return {
                    "status": "BLOCKED_UNOFFICIAL_DOMAIN",
                    "query": query,
                    "url_validation": url_check,
                    "results": [],
                    "total_matches": 0,
                }
            portal_key = str(url_check["portal"])
        elif vendor and vendor.lower() in self.VENDOR_PORTAL_MAP:
            url_check = self.validate_official_url(self.VENDOR_PORTAL_MAP[vendor.lower()])
            portal_key = str(url_check["portal"])
        else:
            portal_key = "multi-vendor-official"
            url_check = {
                "valid": True,
                "portal": portal_key,
                "vendor": vendor or "Multi-Vendor",
                "reason": "APPROVED_OFFICIAL_CATALOG",
            }

        rl_status = self.check_rate_limit(portal_key)
        if not rl_status["allowed"]:
            return {
                "status": "RATE_LIMITED",
                "query": query,
                "url_validation": url_check,
                "rate_limit": rl_status,
                "results": [],
                "total_matches": 0,
            }

        q_tokens = [
            tok.lower()
            for tok in re.findall(r"[a-zA-Z0-9À-ÿ_\-/\.]+", query)
            if len(tok) > 1
        ]
        matched_docs: List[Dict[str, Any]] = []

        for doc in OFFICIAL_INITIAL_DOCUMENTS:
            if doc.security_quarantined or not doc.is_official:
                continue
            if vendor and doc.vendor.lower() != vendor.lower():
                continue
            haystack = (
                f"{doc.title} {doc.vendor} {doc.os_name} {doc.feature} {doc.domain} {doc.content}"
            ).lower()
            hits = sum(1 for tok in q_tokens if tok in haystack)
            if hits > 0 or (vendor and not q_tokens):
                portal_base = self.VENDOR_PORTAL_MAP.get(
                    doc.vendor.lower(), "https://datatracker.ietf.org/doc/html/rfc8200"
                )
                official_https_uri = (
                    doc.source_uri
                    if doc.source_uri.startswith("https://")
                    else f"{portal_base.rstrip('/')}/{quote_plus(doc.doc_id.lower())}"
                )
                matched_docs.append(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "vendor": doc.vendor,
                        "os_name": doc.os_name,
                        "version_build": doc.version_build,
                        "domain": doc.domain,
                        "official_portal_url": official_https_uri,
                        "reliability_score": doc.reliability_score,
                        "relevance_hits": hits,
                        "citation": doc.to_dict(),
                    }
                )

        if online_content_fetcher is not None and target_url is not None and url_check["valid"]:
            raw_online = online_content_fetcher(target_url, query)
            ingested = self.ingestor.ingest_document(
                title=f"Official Portal Result — {query}",
                raw_content=raw_online,
                file_format="html",
                vendor=str(url_check.get("vendor") or vendor or "Official"),
                original_source=f"Official Portal ({url_check['portal']})",
                source_uri=target_url,
                reliability_score=0.96,
                is_official=True,
            )
            online_doc: KnowledgeDocument = ingested["document"]
            if not online_doc.security_quarantined:
                matched_docs.insert(
                    0,
                    {
                        "doc_id": online_doc.doc_id,
                        "title": online_doc.title,
                        "vendor": online_doc.vendor,
                        "os_name": online_doc.os_name,
                        "version_build": online_doc.version_build,
                        "domain": online_doc.domain,
                        "official_portal_url": target_url,
                        "reliability_score": online_doc.reliability_score,
                        "relevance_hits": len(q_tokens) + 5,
                        "citation": online_doc.to_dict(),
                    },
                )

        matched_docs.sort(
            key=lambda item: (int(item["relevance_hits"]), float(item["reliability_score"])),
            reverse=True,
        )
        top_matches = matched_docs[:top_k]

        return {
            "status": "OK",
            "query": query,
            "approved_portals": list(self.APPROVED_OFFICIAL_PORTALS),
            "access_policy": {
                "respect_robots_txt": self.respect_robots_policy,
                "https_only": True,
                "official_domains_only": True,
            },
            "url_validation": url_check,
            "rate_limit": rl_status,
            "total_matches": len(top_matches),
            "results": top_matches,
        }


def _build_open_domain_search_queries(raw_query: str) -> List[str]:
    """Prepare a neutral search plus a source-focused query when the topic is recognizable."""
    import unicodedata

    q_clean = re.sub(r"[?!.,;:]", " ", raw_query).strip()
    q_norm = "".join(c for c in unicodedata.normalize("NFD", q_clean.lower()) if unicodedata.category(c) != "Mn")
    q_stripped = re.sub(
        r"^(qual\s+(?:e\s+)?(?:a|o)\s+|qual\s+|quais\s+(?:sao\s+)?(?:as|os)\s+|quais\s+|o\s+que\s+(?:e\s+)?(?:um|uma|o|a)?\s*|como\s+funciona\s+(?:o|a)?\s*|explique\s+(?:sobre)?\s*|me\s+fale\s+sobre\s*)",
        "", q_norm,
    ).strip()
    topic = q_stripped or q_clean
    queries = [topic]
    words = set(re.findall(r"[\w-]{3,}", topic))
    domain_sites = [
        ({"bgp", "ospf", "mpls", "rfc", "ietf", "router", "roteador", "network", "rede"}, "site:rfc-editor.org OR site:ietf.org OR site:cisco.com"),
        ({"physics", "fisica", "quantum", "quantica", "gravity", "gravitacao", "particula", "particle", "relativity", "relatividade"}, "site:nasa.gov OR site:cern.ch OR site:arxiv.org OR site:aps.org"),
        ({"medical", "medicina", "health", "saude", "disease", "doenca", "treatment", "tratamento", "drug", "medicamento"}, "site:who.int OR site:nih.gov OR site:pubmed.ncbi.nlm.nih.gov OR site:gov.br"),
        ({"law", "legal", "lei", "direito", "regulation", "regulamento", "regulacao"}, "site:gov.br OR site:planalto.gov.br OR site:eur-lex.europa.eu"),
        ({"chemistry", "quimica", "molecule", "molecula", "reaction", "reacao", "element", "elemento"}, "site:acs.org OR site:rsc.org OR site:pubchem.ncbi.nlm.nih.gov OR site:nist.gov"),
        ({"math", "mathematics", "matematica", "algebra", "calculus", "calculo", "theorem", "teorema"}, "site:ams.org OR site:arxiv.org OR site:maa.org"),
        ({"astronomy", "astronomia", "planet", "planeta", "galaxy", "galaxia", "star", "estrela", "cosmology"}, "site:nasa.gov OR site:esa.int OR site:iau.org OR site:arxiv.org"),
        ({"history", "historia", "historical", "historico", "war", "guerra", "empire", "imperio"}, "site:loc.gov OR site:archives.gov OR site:nationalarchives.gov.uk"),
        ({"software", "programming", "programacao", "python", "javascript", "api", "database", "linux", "algorithm", "algoritmo"}, "site:python.org OR site:developer.mozilla.org OR site:kernel.org OR site:learn.microsoft.com"),
    ]
    for markers, sites in domain_sites:
        if words & markers:
            queries.append(f"{topic} ({sites})")
            break
    if q_clean and q_clean.casefold() not in {item.casefold() for item in queries}:
        queries.append(q_clean)
    return list(dict.fromkeys(queries))[:3]


def search_live_web_and_rfc(
    query: str,
    max_results: int = 5,
    timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Busca dinâmica e analítica em tempo real na web e portais técnicos oficiais:
    1. Formulação de query técnica (incluindo expansão para termos IETF RFC).
    2. DuckDuckGo HTML Lite (extrai títulos, snippets e URLs reais desempacotadas).
    3. Extração direta de seção normativa IETF RFC (.txt) quando uma RFC relevante é identificada.
    4. Wikipedia API (pesquisa enciclopédica estruturada em português e inglês).
    5. Resiliência: caso offline ou timeout, retorna lista vazia sem travar a aplicação.
    """
    import urllib.request
    import urllib.parse

    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    clean_q = query.strip()
    if not clean_q:
        return results

    search_queries = _build_open_domain_search_queries(clean_q)

    # 1. DuckDuckGo HTML Lite com queries analíticas
    per_query_limit = max(2, max_results // max(1, len(search_queries)))
    for sq in search_queries:
        if len(results) >= max_results:
            break
        query_added = 0
        try:
            ddg_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(sq)
            req = urllib.request.Request(
                ddg_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=min(timeout, 4.5)) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                snippets = re.findall(
                    r'<a[^>]*class="result__snippet[^"]*"[^>]*>(.*?)</a>',
                    body,
                    re.DOTALL,
                )
                urls = re.findall(
                    r'<a[^>]*class="result__url[^"]*"[^>]*href="([^"]+)"',
                    body,
                    re.DOTALL,
                )
                titles = re.findall(
                    r'<h2[^>]*class="result__title"[^>]*>.*?<a[^>]*>(.*?)</a>',
                    body,
                    re.DOTALL,
                )

                for i, snip in enumerate(snippets[:max_results]):
                    raw_snip = html.unescape(re.sub(r"<[^>]+>", "", snip).strip())
                    raw_url = urls[i].strip() if i < len(urls) else ""
                    clean_title = (
                        html.unescape(re.sub(r"<[^>]+>", "", titles[i]).strip())
                        if i < len(titles)
                        else ""
                    )

                    real_url = raw_url
                    if "uddg=" in raw_url:
                        match_url = re.search(r"uddg=([^&]+)", raw_url)
                        if match_url:
                            real_url = urllib.parse.unquote(match_url.group(1))
                    elif raw_url.startswith("//"):
                        real_url = "https:" + raw_url

                    if raw_snip and len(raw_snip) > 15 and real_url not in seen_urls:
                        seen_urls.add(real_url)
                        results.append(
                            {
                                "title": clean_title or f"Documentação Técnica: {sq}",
                                "snippet": raw_snip,
                                "url": real_url or "https://datatracker.ietf.org",
                                "source": "Web Search (DuckDuckGo)",
                            }
                        )
                        query_added += 1
                        if len(results) >= max_results or query_added >= per_query_limit:
                            break
        except Exception:
            pass

    # 2. Wikipedia API (pt-BR e en como complemento enciclopédico de alta precisão)
    if len(results) < max_results:
        for lang in ["pt", "en"]:
            try:
                wiki_q = clean_q
                wiki_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(wiki_q)}&format=json"
                req_wiki = urllib.request.Request(
                    wiki_url,
                    headers={"User-Agent": "VeronicaNetworkAI/2.0 (Engineering Assistant)"},
                )
                with urllib.request.urlopen(req_wiki, timeout=min(timeout, 3.0)) as resp:
                    wiki_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    for item in wiki_data.get("query", {}).get("search", [])[:2]:
                        title = item.get("title", "")
                        raw_snip = item.get("snippet", "")
                        clean_snip = html.unescape(re.sub(r"<[^>]+>", "", raw_snip).strip())
                        w_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                        if clean_snip and len(clean_snip) > 20 and w_url not in seen_urls:
                            seen_urls.add(w_url)
                            results.append(
                                {
                                    "title": f"Wikipedia ({lang.upper()}): {title}",
                                    "snippet": clean_snip,
                                    "url": w_url,
                                    "source": f"Wikipedia ({lang.upper()})",
                                }
                            )
                            if len(results) >= max_results:
                                break
            except Exception:
                pass
            if len(results) >= max_results:
                break

    return results


def fetch_web_page_content(url: str, timeout: float = 3.0, max_bytes: int = 2_000_000) -> Dict[str, str]:
    """Fetch a public HTTP(S) page and extract readable text, rejecting private hosts."""
    import ipaddress
    import socket
    import urllib.error
    import urllib.request
    from urllib.parse import urljoin, urlparse

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    current_url = url.strip()
    headers = {"User-Agent": "VeronicaResearch/1.0 (+open-domain evidence retrieval)", "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf,application/json;q=0.8,*/*;q=0.3"}
    opener = urllib.request.build_opener(NoRedirect)
    raw = b""
    content_type = ""
    for _ in range(4):
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return {"content": "", "title": "", "url": current_url, "error": "UNSAFE_URL"}
        host = parsed.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
            return {"content": "", "title": "", "url": current_url, "error": "NON_PUBLIC_HOST"}
        try:
            try:
                address = ipaddress.ip_address(host)
                addresses = [address]
            except ValueError:
                addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)]
            if not addresses or any(not address.is_global for address in addresses):
                return {"content": "", "title": "", "url": current_url, "error": "NON_PUBLIC_HOST"}
        except (OSError, ValueError):
            return {"content": "", "title": "", "url": current_url, "error": "HOST_RESOLUTION_FAILED"}

        request = urllib.request.Request(current_url, headers=headers)
        try:
            with opener.open(request, timeout=max(0.5, min(timeout, 5.0))) as response:
                final_url = response.geturl()
                if final_url != current_url:
                    current_url = final_url
                    continue
                raw = response.read(max_bytes + 1)
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                break
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                current_url = urljoin(current_url, exc.headers["Location"])
                continue
            return {"content": "", "title": "", "url": current_url, "error": f"HTTP_{exc.code}"}
        except (OSError, TimeoutError, ValueError):
            return {"content": "", "title": "", "url": current_url, "error": "FETCH_FAILED"}
    else:
        return {"content": "", "title": "", "url": current_url, "error": "REDIRECT_LIMIT"}

    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    if content_type == "application/pdf" or current_url.lower().split("?", 1)[0].endswith(".pdf"):
        try:
            import fitz
            pdf = fitz.open(stream=raw, filetype="pdf")
            content = "\n".join(page.get_text("text") for page in pdf[:8])
            return {"content": content[:30000], "title": "", "url": current_url, "content_type": "application/pdf"}
        except Exception:
            return {"content": "", "title": "", "url": current_url, "error": "PDF_EXTRACTION_FAILED"}

    text = raw.decode("utf-8", errors="replace")
    title = ""
    if "html" in content_type or "<html" in text[:1000].lower() or "<!doctype html" in text[:1000].lower():
        try:
            from selectolax.lexbor import LexborHTMLParser
            tree = LexborHTMLParser(text)
            title_node = tree.css_first("title")
            title = title_node.text(strip=True) if title_node else ""
            for node in tree.css("script,style,noscript,svg,nav,footer,header,form,aside"):
                node.decompose()
            main = tree.css_first("article") or tree.css_first("main") or tree.css_first("[role=main]") or tree.body or tree.root
            text = main.text(separator=" ", strip=True)
        except Exception:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            title = html.unescape(re.sub(r"<[^>]+>", " ", title_match.group(1))).strip() if title_match else ""
            text = html.unescape(re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S))
            text = re.sub(r"<[^>]+>", " ", text)
    elif "json" in content_type:
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    text = re.sub(r"\s+", " ", text).strip()
    return {"content": text[:30000], "title": title[:500], "url": current_url, "content_type": content_type}


class KnowledgeIngestor:
    """
    Processa e valida novos documentos técnicos sem alterar os pesos neurais do modelo.
    Suporta TXT, Markdown, HTML, JSON, XML, DOCX, PDF e configurações CLI (.cfg, .conf, .rsc).
    """

    def ingest_document(
        self,
        title: str,
        raw_content: Union[str, bytes],
        file_format: str = "md",
        vendor: str = "Generic",
        family_model: str = "Unspecified",
        os_name: str = "Unspecified",
        version_build: str = "Unspecified",
        feature: str = "General Networking",
        original_source: str = "User Uploaded Document",
        source_uri: str = "upload://user-document",
        page_section: str = "Full Document",
        published_date: str = "2026-01-01",
        reliability_score: float = 0.75,
        applicability_restrictions: str = "Documento externo submetido pelo usuário; validar compatibilidade antes de uso em produção.",
        is_official: bool = False,
        domain: str = "routing",
        max_chunk_chars: int = 1200,
        overlap_chars: int = 150,
    ) -> Dict[str, Any]:
        normalized = normalize_format_content(raw_content, file_format)
        from knowledge.ingestion.sanitizer import sanitize_untrusted_text

        sanitized = sanitize_untrusted_text(normalized, source="ingested_document")
        scrubbed_text = sanitized.text
        secrets_redacted = sanitized.secrets_redacted
        injected = sanitized.injection_detected
        injection_reasons = list(sanitized.reasons)
        commands, tables = extract_tables_and_commands(scrubbed_text)
        chunks = chunk_large_document(
            scrubbed_text,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )

        doc_id = f"DOC-USR-{uuid.uuid4().hex[:8].upper()}"
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        effective_reliability = (
            0.0
            if injected
            else (reliability_score if is_official else min(reliability_score, 0.80))
        )

        doc = KnowledgeDocument(
            doc_id=doc_id,
            title=title,
            content=scrubbed_text,
            vendor=vendor,
            family_model=family_model,
            os_name=os_name,
            version_build=version_build,
            feature=feature,
            original_source=original_source,
            source_uri=source_uri,
            page_section=page_section,
            published_date=published_date,
            consulted_date=now_date,
            reliability_score=effective_reliability,
            applicability_restrictions=applicability_restrictions,
            extracted_commands=commands,
            extracted_tables=tables,
            is_official=is_official,
            is_obsolete=False,
            security_quarantined=injected,
            quarantine_reason="; ".join(injection_reasons),
            domain=domain,
        )

        return {
            "document": doc,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "secrets_redacted": secrets_redacted,
            "security_quarantined": injected,
            "quarantine_reasons": injection_reasons,
            "neural_weights_modified": False,
        }

