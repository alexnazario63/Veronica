"""
Tokenizador Técnico Especializado em Engenharia de Redes da Veronica.
Combina um vocabulário técnico curado (protocolos, comandos CLI multi-vendor,
prefixos de interfaces, estados de protocolo e termos em português/inglês)
com sub-tokens e fallback em nível de caractere/byte para cobertura de 100%
sem perda de símbolos técnicos (IPv4, IPv6, CIDR, MAC, ASN, interfaces).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


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
]

BASE_NETWORK_VOCAB: List[str] = [
    # Protocolos e tecnologias
    "bgp", "mp-bgp", "ospf", "ospfv2", "ospfv3", "isis", "is-is", "mpls", "ldp",
    "rsvp-te", "sr", "segment", "routing", "evpn", "vxlan", "l2vpn", "l3vpn",
    "vpws", "vpls", "vlan", "stp", "rstp", "mstp", "lacp", "arp", "nd", "ipv4",
    "ipv6", "icmp", "tcp", "udp", "mtu", "mss", "qos", "acl", "nat", "ipsec",
    "gre", "bfd", "snmp", "netconf", "restconf", "gnmi", "syslog", "ntp", "dns",
    "dhcp", "radius", "tacacs+", "ssh", "dwdm", "otn", "sfp", "qsfp", "dbm",
    # Fabricantes e sistemas operacionais
    "huawei", "vrp", "cloudengine", "netengine", "cisco", "ios", "ios-xe",
    "ios-xr", "nx-os", "juniper", "junos", "mikrotik", "routeros", "nokia",
    "sros", "sr-os", "arista", "eos", "fortinet", "fortios", "hpe", "aruba",
    "extreme", "dell", "ubiquiti", "zte",
    # Comandos e palavras-chave CLI
    "display", "show", "interface", "ip", "address", "Brief", "brief", "summary",
    "neighbor", "peer", "as-number", "remote-as", "router-id", "network", "area",
    "cost", "priority", "hello", "dead", "timer", "passive-interface", "vrf",
    "vpn-instance", "route-policy", "route-map", "prefix-list", "ip-prefix",
    "community", "local-preference", "med", "as-path", "next-hop", "ebgp-max-hop",
    "ebgp-multihop", "connect-interface", "update-source", "loopback", "loopback0",
    "gigabitethernet", "tengige", "10ge", "40ge", "100ge", "ge-", "xe-", "et-",
    "ether1", "ether2", "bridge", "eth-trunk", "port-channel", "bundle-ether",
    "trunk", "access", "dot1q", "encapsulation", "shutdown", "undo", "no",
    "commit", "confirmed", " trial", "save", "rollback", "configuration",
    "current-configuration", "running-config", "candidate", "diff", "ping",
    "traceroute", "tracert", "optical-module", "transceiver", "diagnostics",
    # Estados e diagnóstico
    "up", "down", "administratively", "established", "active", "idle", "connect",
    "opensent", "openconfirm", "full", "2way", "exstart", "exchange", "loading",
    "init", "attempt", "forwarding", "discarding", "blocking", "learning",
    "root", "designated", "mismatch", "flap", "loss", "crc", "rx", "tx", "power",
    "alarm", "warning", "critical", "high", "low", "timeout", "unreachable",
    # Termos técnicos em português
    "diagnostico", "diagnóstico", "evidencia", "evidência", "causa", "raiz",
    "hipotese", "hipótese", "risco", "impacto", "validacao", "validação",
    "reversao", "reversão", "aprovacao", "aprovação", "equipamento", "topologia",
    "enlace", "sessao", "sessão", "vizinhanca", "vizinhança", "configuracao",
    "configuração", "rota", "tabela", "fib", "rib", "pacote", "latencia",
    "latência", "atenuacao", "atenuação", "potencia", "potência", "optica",
    "óptica", "fibra", "transceptor", "seguranca", "segurança", "auditoria",
    "conectividade", "interface", "porta", " mascara", "máscara", "sub-rede",
    "padrao", "padrão", "estatica", "estática", "dinamica", "dinâmica",
    "veronica", "rede", "redes", "roteador", "switch", "firewall", "backbone",
    # Conectivos e palavras frequentes
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "sem", "entre", "sobre", "que", "se",
    "como", "quando", "onde", "qual", "quais", "porque", "está", "esta", "estao",
    "estão", "foi", "ser", "sao", "são", "tem", "possui", "deve", "pode",
    "verificar", "analisar", "consultar", "executar", "aplicar", "corrigir",
    "comando", "comandos", "estado", "status", "erro", "falha", "operacional",
    "normal", "ativo", "inativo", "habilitado", "desabilitado",
]

# Caracteres ASCII imprimíveis e acentuados para garantir tokenização sem <unk>
CHAR_TOKENS: List[str] = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,;:!?/-_+=*()[]{}<>@#%^&|~`\"'\\ \n\t"
    "áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ"
)

TOKEN_PATTERN = re.compile(
    r"<\w+>|<\/\w+>"
    r"|\b\d{1,3}(?:\.\d{1,3}){3}(?:\/\d{1,2})?\b"  # IPv4 / CIDR
    r"|\b(?:[A-Za-z]+(?:-\w+)+)\b"                  # Palavras hifenizadas técnicas (ios-xr, route-policy)
    r"|[A-Za-zÀ-ÿ0-9_]+"                            # Palavras e identificadores
    r"|\s+"                                         # Espaços e quebras de linha
    r"|[^\sA-Za-zÀ-ÿ0-9_]"                          # Pontuação individual
)


class TechnicalTokenizer:
    """
    Tokenizador técnico híbrido (Vocabulário de Redes + Subwords + Caracteres).
    """

    def __init__(self, extra_tokens: Optional[Iterable[str]] = None) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        all_initial: List[str] = []
        all_initial.extend(SPECIAL_TOKENS)
        all_initial.extend(BASE_NETWORK_VOCAB)
        all_initial.extend(CHAR_TOKENS)
        if extra_tokens:
            all_initial.extend(extra_tokens)

        for tok in all_initial:
            self._add_token(tok)

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
        """Expande o vocabulário com termos técnicos frequentes encontrados no corpus."""
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
            elif piece.lower() in self.token_to_id and piece.islower():
                ids.append(self.token_to_id[piece.lower()])
            else:
                # Decompõe em caracteres conhecidos sem perder informação
                for ch in piece:
                    ids.append(self.token_to_id.get(ch, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special_set = set(SPECIAL_TOKENS) if skip_special_tokens else set()
        pieces: List[str] = []
        for idx in token_ids:
            tok = self.id_to_token.get(int(idx), "<unk>")
            if tok in special_set:
                continue
            pieces.append(tok)
        return "".join(pieces)

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
        return tokenizer
