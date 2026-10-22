"""
Tokenizador Técnico Dinâmico e Poliglota BPE / Byte-Level da Veronica.
Suporta codificação e decodificação sem perdas (zero <unk>) para:
- Qualquer idioma humano (Português, Inglês, Espanhol, Chinês, Russo, Árabe, etc.) via Byte-Level UTF-8.
- Blocos de código-fonte poliglota com indentação precisa (Python, TypeScript, Rust, C++, Go, SQL).
- Notação e fórmulas matemáticas em LaTeX.
- Primitivas e identificadores de rede (IPv4, IPv6, CIDR, MAC, ASN, interfaces).
Preserva integralmente o contrato TechnicalTokenizer de INTERFACES.md.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
try:
    import regex as re
except ImportError:  # pragma: no cover
    import re
from typing import Dict, Iterable, List, Optional, Union


SPECIAL_TOKENS: List[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<sep>",
    "<cmd>",
    "</cmd>",
    "<fact>",
    "<infer>",
    "<hypo>",
    "[SYSTEM]", "[USER]", "[ASSISTANT]", "[THOUGHT]", "[/THOUGHT]",
    "[TOOL_CALL]", "[/TOOL_CALL]", "[TOOL_RESULT]", "[/TOOL_RESULT]",
    "[ARTIFACT]", "[/ARTIFACT]", "[CLAIM:FACT]", "[CLAIM:INFERENCE]",
    "[CLAIM:HYPOTHESIS]",
]

NETWORK_ATOM_BUCKETS = [f"[NET_ATOM_{index:04d}]" for index in range(64)]

# Lista vazia mantida para retrocompatibilidade de importações legadas
BASE_NETWORK_VOCAB: List[str] = []

# Primitivas de indentação e espaçamento de código-fonte
CODE_INDENT_TOKENS: List[str] = [
    "        ",  # 8 espaços
    "    ",      # 4 espaços
    "  ",        # 2 espaços
    "\t",
    "\r\n",
    "\n\n",
    "\n",
    " ",
]

# Primitivas poliglota de linguagens de programação (Python, TypeScript, Rust, C++, Go, SQL)
POLYGLOT_CODE_TOKENS: List[str] = [
    # Python
    "def", "class", "return", "import", "from", "if", "else", "elif",
    "for", "while", "try", "except", "finally", "with", "as", "yield",
    "lambda", "self", "None", "True", "False", "async", "await", "pass",
    "break", "continue", "raise", "assert", "print",
    # TypeScript / JavaScript
    "function", "const", "let", "var", "export", "default", "type",
    "interface", "extends", "implements", "null", "undefined", "typeof",
    "instanceof", "new", "this", "=>", "===", "!==", "console", "log",
    "string", "number", "boolean", "any",
    # Rust
    "fn", "pub", "struct", "impl", "enum", "match", "mut", "ref",
    "trait", "where", "unsafe", "mod", "use", "crate", "Self", "::", "->",
    "Option", "Result", "Some", "Ok", "Err",
    # C++ / C
    "int", "void", "char", "bool", "float", "double", "auto", "#include",
    "template", "typename", "namespace", "using", "public", "private",
    "protected", "std", "cout", "cin", "endl", "nullptr", "constexpr",
    "vector",
    # Go
    "package", "func", "go", "chan", "defer", "select", "fallthrough",
    "range", "make", "len", "cap", ":=", "nil", "error",
    # SQL
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "ON", "GROUP",
    "BY", "ORDER", "HAVING", "LIMIT", "CREATE", "TABLE", "DROP", "ALTER",
    "INDEX", "AND", "OR", "NOT", "NULL", "AS", "IN", "DISTINCT", "COUNT",
    # Operadores de sintaxe
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
    "%", "&", "|", "^", "~", "<<", ">>",
]

# Primitivas matemáticas em notação LaTeX
LATEX_MATH_TOKENS: List[str] = [
    # Operadores e funções de cálculo
    r"\frac", r"\sum", r"\int", r"\prod", r"\lim", r"\sqrt", r"\partial",
    r"\nabla", r"\infty", r"\times", r"\div", r"\pm", r"\leq", r"\geq",
    r"\neq", r"\approx", r"\equiv", r"\in", r"\notin", r"\subset", r"\cup",
    r"\cap", r"\to", r"\forall", r"\exists", r"\mathbf", r"\mathcal",
    r"\text", r"\left", r"\right", r"\begin", r"\end", r"\matrix",
    r"\cdots", r"\ldots", r"\quad",
    # Letras gregas
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta",
    r"\eta", r"\theta", r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu",
    r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\upsilon", r"\phi",
    r"\chi", r"\psi", r"\omega", r"\Gamma", r"\Delta", r"\Theta",
    r"\Lambda", r"\Xi", r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega",
]

# Caracteres ASCII imprimíveis
ASCII_CHARS: List[str] = [chr(c) for c in range(32, 127)]

SPECIAL_PATTERN = "|".join(re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True))

TOKEN_PATTERN = re.compile(
    rf"{SPECIAL_PATTERN}"
    r"|<\/?\w+>"
    r"|(?<![\w:])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w.])"
    r"|(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:/\d{1,3})?(?![\w:])"
    r"|\bAS\d{1,10}\b"
    r"|\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
    r"|\b(?:Gi|GigabitEthernet|Te|TenGigabitEthernet|Eth|Ethernet|Hu|HundredGigE|Fa|FastEthernet|ge-|xe-|et-|ether)"
    r"[A-Za-z0-9._/: -]*\d(?:[A-Za-z0-9._/:-]*\d)?"
    r"|\b(?:[A-Za-z]+(?:-[\w]+)+)\b"
    r"|\\[a-zA-Z]+"
    r"| {2,8}|\t|\r\n|\n\n|\n"
    r"|===|!==|:=|=>|->|::|!=|<=|>=|==|&&|\|\||\+\+|--|\+=|-=|\*=|/=|<<|>>"
    r"|[\w]+|\s+|[^\s\w]",
    re.UNICODE,
)

NETWORK_ATOM_PATTERN = re.compile(
    r"^(?:(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?"
    r"|(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:/\d{1,3})?"
    r"|AS\d{1,10}"
    r"|(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}"
    r"|(?:Gi|GigabitEthernet|Te|TenGigabitEthernet|Eth|Ethernet|Hu|HundredGigE|Fa|FastEthernet|ge-|xe-|et-|ether)[A-Za-z0-9._/: -]*\d(?:[A-Za-z0-9._/:-]*\d)?)$",
    re.IGNORECASE,
)


class TechnicalTokenizer:
    """
    Tokenizador técnico BPE / Byte-Level dinâmico e poliglota da Veronica.
    Suporta sem perdas qualquer idioma humano, blocos de código-fonte
    (indentação, Python, TypeScript, Rust, C++, Go, SQL), fórmulas LaTeX
    e primitivas de redes.
    """

    def __init__(self, extra_tokens: Optional[Iterable[str]] = None) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._special_tokens: List[str] = list(SPECIAL_TOKENS)

        all_initial: List[str] = []
        all_initial.extend(SPECIAL_TOKENS)
        all_initial.extend(CODE_INDENT_TOKENS)
        all_initial.extend(POLYGLOT_CODE_TOKENS)
        all_initial.extend(LATEX_MATH_TOKENS)
        all_initial.extend(ASCII_CHARS)
        # Byte fallback cobrindo todos os 256 bytes possíveis (zero <unk>)
        all_initial.extend(f"<0x{b:02X}>" for b in range(256))
        if extra_tokens:
            all_initial.extend(extra_tokens)

        for tok in all_initial:
            self._add_token(tok)

        # Slots reservados para átomos de rede intactos
        first_slot = len(self.token_to_id)
        for offset, token in enumerate(NETWORK_ATOM_BUCKETS):
            self.id_to_token[first_slot + offset] = token
        self._network_atom_ids = list(range(first_slot, first_slot + len(NETWORK_ATOM_BUCKETS)))
        self._network_atom_next = 0

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            return idx
        return self.token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def fit(self, texts: Iterable[str], max_new_tokens: int = 512, min_freq: int = 2) -> int:
        """Expande dinamicamente o vocabulário com termos e subwords frequentes encontrados no corpus."""
        counter: Counter[str] = Counter()
        for text in texts:
            for match in TOKEN_PATTERN.findall(text):
                low = match.lower()
                if len(low) > 1 and not low.isspace() and low not in self.token_to_id:
                    counter[low] += 1
                if len(match) > 1 and not match.isspace() and match not in self.token_to_id:
                    counter[match] += 1

        added = 0
        for tok, freq in counter.most_common():
            if freq < min_freq or added >= max_new_tokens:
                break
            if tok not in self.token_to_id:
                self._add_token(tok)
                added += 1
        return added

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for piece in TOKEN_PATTERN.findall(text):
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
            elif NETWORK_ATOM_PATTERN.fullmatch(piece):
                # Preserva átomos intactos usando slots pré-reservados
                while self._network_atom_next < len(self._network_atom_ids):
                    token_id = self._network_atom_ids[self._network_atom_next]
                    self._network_atom_next += 1
                    if self.id_to_token[token_id].startswith("[NET_ATOM_"):
                        placeholder = self.id_to_token[token_id]
                        self.token_to_id.pop(placeholder, None)
                        self.id_to_token[token_id] = piece
                        self.token_to_id[piece] = token_id
                        ids.append(token_id)
                        break
                else:
                    # Fallback byte-level sem perda caso esgote os slots
                    for b in piece.encode("utf-8"):
                        byte_tok = f"<0x{b:02X}>"
                        ids.append(self.token_to_id.get(byte_tok, self.unk_id))
            elif piece.lower() in self.token_to_id and piece.islower():
                ids.append(self.token_to_id[piece.lower()])
            else:
                # Decomposição em caracteres ou bytes UTF-8 (zero unk)
                for ch in piece:
                    if ch in self.token_to_id:
                        ids.append(self.token_to_id[ch])
                    else:
                        for b in ch.encode("utf-8"):
                            byte_tok = f"<0x{b:02X}>"
                            ids.append(self.token_to_id[byte_tok])

        if add_eos:
            ids.append(self.eos_id)

        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special_set = set(self._special_tokens) if skip_special_tokens else set()
        byte_buffer = bytearray()
        out_pieces: List[str] = []

        def flush_bytes() -> None:
            if byte_buffer:
                out_pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for idx in token_ids:
            tok = self.id_to_token.get(int(idx), "")
            if not tok:
                continue
            if tok in special_set or tok.startswith("[NET_ATOM_"):
                continue
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    b = int(tok[3:5], 16)
                    byte_buffer.append(b)
                    continue
                except ValueError:
                    pass
            byte_buffer.extend(tok.encode("utf-8"))

        flush_bytes()
        return "".join(out_pieces)

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": "Veronica",
            "vocab_size": self.vocab_size,
            "tokens": [self.id_to_token[i] for i in range(self.vocab_size)],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TechnicalTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.token_to_id.clear()
        tokenizer.id_to_token.clear()
        for tok in payload["tokens"]:
            tokenizer._add_token(tok)
        first_slot = len(tokenizer.token_to_id)
        for offset, token in enumerate(NETWORK_ATOM_BUCKETS):
            tokenizer.id_to_token[first_slot + offset] = token
        tokenizer._network_atom_ids = list(
            range(first_slot, first_slot + len(NETWORK_ATOM_BUCKETS))
        )
        tokenizer._network_atom_next = 0
        return tokenizer
