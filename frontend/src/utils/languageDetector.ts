export type SupportedLanguage =
  | 'python'
  | 'typescript'
  | 'javascript'
  | 'rust'
  | 'c++'
  | 'go'
  | 'sql'
  | 'json'
  | 'mermaid'
  | 'markdown'
  | 'network-cli'
  | 'diff'
  | 'html';

export interface LanguageMeta {
  id: SupportedLanguage;
  label: string;
  extension: string;
  monacoLanguage: string;
  badgeBg: string;
  badgeColor: string;
  iconName: string;
}

export const SUPPORTED_LANGUAGES: Record<SupportedLanguage, LanguageMeta> = {
  python: {
    id: 'python',
    label: 'Python 3.12',
    extension: '.py',
    monacoLanguage: 'python',
    badgeBg: '#1e3a5f',
    badgeColor: '#79c0ff',
    iconName: 'FileCode2',
  },
  typescript: {
    id: 'typescript',
    label: 'TypeScript 5.x',
    extension: '.ts',
    monacoLanguage: 'typescript',
    badgeBg: '#1f3c64',
    badgeColor: '#6cb6ff',
    iconName: 'FileCode',
  },
  javascript: {
    id: 'javascript',
    label: 'JavaScript ESNext',
    extension: '.js',
    monacoLanguage: 'javascript',
    badgeBg: '#443c16',
    badgeColor: '#e3b341',
    iconName: 'Code',
  },
  rust: {
    id: 'rust',
    label: 'Rust 1.80',
    extension: '.rs',
    monacoLanguage: 'rust',
    badgeBg: '#48241b',
    badgeColor: '#f0883e',
    iconName: 'Cpu',
  },
  'c++': {
    id: 'c++',
    label: 'C++ 20',
    extension: '.cpp',
    monacoLanguage: 'cpp',
    badgeBg: '#1b2d4f',
    badgeColor: '#58a6ff',
    iconName: 'Binary',
  },
  go: {
    id: 'go',
    label: 'Go 1.23',
    extension: '.go',
    monacoLanguage: 'go',
    badgeBg: '#143d4e',
    badgeColor: '#39c5bb',
    iconName: 'FastForward',
  },
  sql: {
    id: 'sql',
    label: 'SQL / Postgres',
    extension: '.sql',
    monacoLanguage: 'sql',
    badgeBg: '#341f4d',
    badgeColor: '#d2a8ff',
    iconName: 'Database',
  },
  json: {
    id: 'json',
    label: 'JSON Schema',
    extension: '.json',
    monacoLanguage: 'json',
    badgeBg: '#1f382a',
    badgeColor: '#7ee787',
    iconName: 'Braces',
  },
  mermaid: {
    id: 'mermaid',
    label: 'Mermaid Diagram',
    extension: '.mmd',
    monacoLanguage: 'markdown',
    badgeBg: '#471c3c',
    badgeColor: '#f778ba',
    iconName: 'Workflow',
  },
  markdown: {
    id: 'markdown',
    label: 'Markdown GFM',
    extension: '.md',
    monacoLanguage: 'markdown',
    badgeBg: '#21262d',
    badgeColor: '#c9d1d9',
    iconName: 'FileText',
  },
  'network-cli': {
    id: 'network-cli',
    label: 'Network CLI Multi-Vendor',
    extension: '.cli',
    monacoLanguage: 'shell',
    badgeBg: '#1b3d36',
    badgeColor: '#56d364',
    iconName: 'Terminal',
  },
  diff: {
    id: 'diff',
    label: 'Unified Diff',
    extension: '.diff',
    monacoLanguage: 'diff',
    badgeBg: '#362327',
    badgeColor: '#ff7b72',
    iconName: 'GitCompare',
  },
  html: {
    id: 'html',
    label: 'HTML5 Semantic',
    extension: '.html',
    monacoLanguage: 'html',
    badgeBg: '#442217',
    badgeColor: '#ffa657',
    iconName: 'Layout',
  },
};

/**
 * Automatically detects the language of a given code snippet or artifact filename.
 */
export function detectLanguage(filenameOrHint: string = '', content: string = ''): SupportedLanguage {
  const lowerHint = filenameOrHint.toLowerCase().trim();

  // 1. Check explicit hint or file extensions
  if (lowerHint.endsWith('.py') || lowerHint.includes('python')) return 'python';
  if (lowerHint.endsWith('.ts') || lowerHint.endsWith('.tsx') || lowerHint.includes('typescript')) return 'typescript';
  if (lowerHint.endsWith('.js') || lowerHint.endsWith('.jsx') || lowerHint.includes('javascript')) return 'javascript';
  if (lowerHint.endsWith('.rs') || lowerHint.includes('rust')) return 'rust';
  if (lowerHint.endsWith('.cpp') || lowerHint.endsWith('.hpp') || lowerHint.endsWith('.cc') || lowerHint.endsWith('.h') || lowerHint.includes('c++') || lowerHint === 'cpp') return 'c++';
  if (lowerHint.endsWith('.go') || lowerHint.includes('golang') || lowerHint === 'go') return 'go';
  if (lowerHint.endsWith('.sql') || lowerHint.includes('postgres') || lowerHint.includes('sqlite') || lowerHint === 'sql') return 'sql';
  if (lowerHint.endsWith('.json') || lowerHint.includes('json')) return 'json';
  if (lowerHint.endsWith('.mmd') || lowerHint.includes('mermaid')) return 'mermaid';
  if (lowerHint.endsWith('.md') || lowerHint.includes('markdown')) return 'markdown';
  if (lowerHint.endsWith('.diff') || lowerHint.endsWith('.patch') || lowerHint.includes('diff') || lowerHint.includes('patch')) return 'diff';
  if (lowerHint.endsWith('.html') || lowerHint.endsWith('.htm') || lowerHint.includes('html')) return 'html';
  if (lowerHint.includes('ios') || lowerHint.includes('junos') || lowerHint.includes('eos') || lowerHint.includes('bgp') || lowerHint.includes('cli') || lowerHint.includes('config_diff')) return 'network-cli';

  if (!content) return 'typescript';

  const trimmed = content.trim();

  // 2. Diff detection
  if (trimmed.startsWith('---') || trimmed.startsWith('+++') || trimmed.startsWith('diff --git') || trimmed.includes('@@ -')) {
    return 'diff';
  }

  // 3. Mermaid detection
  if (
    trimmed.startsWith('graph ') ||
    trimmed.startsWith('flowchart ') ||
    trimmed.startsWith('sequenceDiagram') ||
    trimmed.startsWith('stateDiagram') ||
    trimmed.startsWith('erDiagram') ||
    trimmed.startsWith('classDiagram')
  ) {
    return 'mermaid';
  }

  // 4. JSON detection
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      JSON.parse(trimmed);
      return 'json';
    } catch {
      // Not valid JSON, continue checks
    }
  }

  // 5. HTML detection
  if (/<!DOCTYPE\s+html/i.test(trimmed) || /<html[\s>]/i.test(trimmed) || (/<div[\s>]/i.test(trimmed) && /<\/div>/i.test(trimmed))) {
    return 'html';
  }

  // 6. Rust detection
  if (
    /\bfn\s+main\b/.test(trimmed) ||
    /\blet\s+mut\s+/.test(trimmed) ||
    /\bpub\s+struct\s+/.test(trimmed) ||
    /\bimpl\s+[A-Za-z0-9_]+\s+for\b/.test(trimmed) ||
    /\buse\s+std::/.test(trimmed) ||
    trimmed.includes('println!')
  ) {
    return 'rust';
  }

  // 7. Go detection
  if (
    /\bpackage\s+[a-z0-9_]+/.test(trimmed) ||
    /\bfunc\s+(?:\([a-zA-Z0-9_* ]+\)\s+)?[a-zA-Z0-9_]+\s*\(/.test(trimmed) ||
    trimmed.includes('fmt.Println')
  ) {
    return 'go';
  }

  // 8. C++ detection
  if (
    trimmed.includes('#include <') ||
    trimmed.includes('std::cout') ||
    trimmed.includes('std::vector') ||
    /\bint\s+main\s*\([^)]*\)\s*\{/.test(trimmed) ||
    trimmed.includes('namespace std')
  ) {
    return 'c++';
  }

  // 9. Python detection
  if (
    /\bdef\s+[a-zA-Z0-9_]+\s*\(/.test(trimmed) ||
    /\bimport\s+[a-zA-Z0-9_]+/.test(trimmed) ||
    /\bfrom\s+[a-zA-Z0-9_.]+\s+import\b/.test(trimmed) ||
    trimmed.includes('if __name__ == "__main__":') ||
    /\bclass\s+[A-Za-z0-9_]+(?:\([A-Za-z0-9_.]*\))?:/.test(trimmed) ||
    /\belif\s+/.test(trimmed)
  ) {
    return 'python';
  }

  // 10. SQL detection
  if (
    /\bSELECT\b[\s\S]+\bFROM\b/i.test(trimmed) ||
    /\bINSERT\s+INTO\b/i.test(trimmed) ||
    /\bCREATE\s+TABLE\b/i.test(trimmed) ||
    /\bALTER\s+TABLE\b/i.test(trimmed)
  ) {
    return 'sql';
  }

  // 11. Network CLI detection
  if (
    /\brouter\s+bgp\s+\d+/i.test(trimmed) ||
    /\binterface\s+(?:Gigabit|TenGig|Loopback|Ethernet)/i.test(trimmed) ||
    /\bshow\s+ip\s+bgp\b/i.test(trimmed) ||
    /\bset\s+protocols\s+bgp\b/i.test(trimmed) ||
    /\bneighbor\s+\d+\.\d+\.\d+\.\d+/i.test(trimmed)
  ) {
    return 'network-cli';
  }

  // 12. TypeScript vs JavaScript
  if (
    /\binterface\s+[A-Z][a-zA-Z0-9_]*\s*\{/.test(trimmed) ||
    /\btype\s+[A-Z][a-zA-Z0-9_]*\s*=/.test(trimmed) ||
    /:\s*(?:string|number|boolean|any|unknown|void|Record<)[\s;,)=]/.test(trimmed) ||
    trimmed.includes('as const')
  ) {
    return 'typescript';
  }

  if (
    /\bconst\s+/.test(trimmed) ||
    /\blet\s+/.test(trimmed) ||
    /\bfunction\s+/.test(trimmed) ||
    trimmed.includes('console.log')
  ) {
    return 'javascript';
  }

  // 13. Markdown fallback
  if (trimmed.startsWith('# ') || trimmed.startsWith('## ') || trimmed.includes('```')) {
    return 'markdown';
  }

  return 'python';
}
