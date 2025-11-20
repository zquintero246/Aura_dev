import React from 'react';

// Very small, safe-ish Markdown renderer for lists, bold, italics, code.
// Not full Markdown. Escapes HTML by default.

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMath(expr: string, display: boolean) {
  const katex = typeof window !== 'undefined' ? (window as WindowWithKatex).katex : undefined;
  if (katex?.renderToString) {
    try {
      return katex.renderToString(expr, { displayMode: display, throwOnError: false });
    } catch {}
  }
  return `<span class="rounded px-1.5 py-0.5 text-[13px] font-mono text-[#f5cd6a] bg-white/5 border border-white/10">${escapeHtml(
    expr
  )}</span>`;
}

function regexEscape(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function formatInline(s: string): string {
  const placeholders = new Map<string, string>();
  let working = s;
  const inlineExpr = /\\\(([\s\S]+?)\\\)/g;
  const displayExpr = /\\\[([\s\S]+?)\\\]/g;
  working = working.replace(inlineExpr, (_match, expr) => {
    const key = `@@MATH_INLINE_${placeholders.size}@@`;
    placeholders.set(key, renderMath(expr, false));
    return key;
  });
  working = working.replace(displayExpr, (_match, expr) => {
    const key = `@@MATH_DISPLAY_${placeholders.size}@@`;
    placeholders.set(key, renderMath(expr, true));
    return key;
  });
  let out = escapeHtml(working);
  out = out.replace(/`([^`]+)`/g, '<code class="px-1 rounded bg-white/10 text-white/90">$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>');
  out = out.replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, '$1<em class="italic">$2</em>');
  placeholders.forEach((value, key) => {
    out = out.replace(new RegExp(regexEscape(key), 'g'), value);
  });
  return out;
}

function extractMathPlaces(text: string) {
  let cursor = 0;
  let builder = '';
  const placeholders: Record<string, string> = {};
  let counter = 0;

  const renderMath = (expr: string, display: boolean): string => {
    const katex = typeof window !== 'undefined' ? (window as WindowWithKatex).katex : undefined;
    if (katex?.renderToString) {
      try {
        return katex.renderToString(expr, { displayMode: display, throwOnError: false });
      } catch {
        // fall through to fallback
      }
    }
    return `<span class="rounded px-1.5 py-0.5 text-[13px] font-mono text-[#f5cd6a] bg-white/5 border border-white/10">${escapeHtml(
      expr
    )}</span>`;
  };

  while (cursor < text.length) {
    const inlineIdx = text.indexOf('\\(', cursor);
    const displayIdx = text.indexOf('\\[', cursor);
    const candidates = [inlineIdx, displayIdx].filter((idx) => idx >= 0);
    if (!candidates.length) break;
    const start = Math.min(...candidates);
    const isDisplay = start === displayIdx;
    const close = isDisplay ? '\\]' : '\\)';
    const delimLen = 2;
    const end = text.indexOf(close, start + delimLen);
    if (end === -1) break;
    builder += text.slice(cursor, start);
    const expr = text.slice(start + delimLen, end);
    const placeholder = `@@MATH${counter}@@`;
    placeholders[placeholder] = renderMath(expr, isDisplay);
    counter += 1;
    builder += placeholder;
    cursor = end + delimLen;
  }

  builder += text.slice(cursor);
  return { processed: builder, placeholders };
}

interface WindowWithKatex extends Window {
  katex?: {
    renderToString: (
      expr: string,
      options: { displayMode?: boolean; throwOnError?: boolean }
    ) => string;
  };
}

function escapeRegexValue(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Light-theme formatting for table cells (remove literal ** and use dark text)
function formatTableCell(s: string): string {
  let out = escapeHtml(s);
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, '$1<em>$2</em>');
  out = out.replace(
    /`([^`]+)`/g,
    '<code style="background:#f1f5f9;color:#111;padding:0 4px;border-radius:4px;">$1</code>'
  );
  out = out.replace(/\*\*/g, '');
  return out;
}

// HTML table for clipboard with light colors (better for Docs/Excel)
function buildHtmlTableLight(header: string[], rows: string[][]): string {
  const cell = (t: string, tag: 'th' | 'td') =>
    `<${tag} style="border:1px solid #d1d5db;padding:6px 8px;text-align:left;">${escapeHtml(t)}</${tag}>`;
  const thead = `<thead style="background:#f1f5f9;"><tr>${header.map((h) => cell(h, 'th')).join('')}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map(
      (r, i) =>
        `<tr style="background:${i % 2 ? '#f8fafc' : 'transparent'};">${r.map((c) => cell(c, 'td')).join('')}</tr>`
    )
    .join('')}</tbody>`;
  return `<table style="border-collapse:collapse;min-width:600px;color:#111;background:#fff;">${thead}${tbody}</table>`;
}

function toTSV(header: string[], rows: string[][]): string {
  const esc = (v: string) => String(v).replace(/\t/g, ' ').replace(/\r?\n/g, ' ');
  const lines = [header.map(esc).join('\t'), ...rows.map((r) => r.map(esc).join('\t'))];
  return lines.join('\n');
}

function buildHtmlTable(header: string[], rows: string[][]): string {
  const cell = (t: string, tag: 'th' | 'td') =>
    `<${tag} style="border:1px solid rgba(255,255,255,0.15);padding:6px 8px;text-align:left;">${escapeHtml(t)}</${tag}>`;
  const thead = `<thead style="background:rgba(255,255,255,0.06);"><tr>${header
    .map((h) => cell(h, 'th'))
    .join('')}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map(
      (r, i) =>
        `<tr style="background:${i % 2 ? 'rgba(255,255,255,0.03)' : 'transparent'};">${r
          .map((c) => cell(c, 'td'))
          .join('')}</tr>`
    )
    .join('')}</tbody>`;
  return `<table style="border-collapse:collapse;min-width:600px;color:#eee;background:transparent;">${thead}${tbody}</table>`;
}

interface WindowWithKatex extends Window {
  katex?: {
    renderToString: (
      expr: string,
      options: { displayMode?: boolean; throwOnError?: boolean }
    ) => string;
  };

  ClipboardItem?: typeof ClipboardItem;
}

function TableBlock({ header, rows }: { header: string[]; rows: string[][] }) {
  const [copied, setCopied] = React.useState(false);
  const onCopy = async () => {
    try {
      const tsv = toTSV(header, rows);
      const html = buildHtmlTable(header, rows);
      const CI: any = (window as any).ClipboardItem;
      if (navigator?.clipboard && (navigator.clipboard as any).write && CI) {
        const data: Record<string, Blob> = {
          'text/plain': new Blob([tsv], { type: 'text/plain' }),
          'text/html': new Blob([html], { type: 'text/html' }),
        };
        await (navigator.clipboard as any).write([new CI(data)]);
      } else if (navigator?.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(tsv);
      } else {
        throw new Error('Clipboard API not available');
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (e) {
      setCopied(false);
    }
  };

  return (
    <div className="my-3 overflow-x-auto relative rounded-md ring-1 ring-white/10">
      <div className="absolute right-2 -top-2">
        <button
          onClick={onCopy}
          title="Copiar (compatible con Excel)"
          className="text-[12px] px-2 py-1 rounded bg-white/10 hover:bg-white/15 ring-1 ring-white/10 text-white/90"
        >
          {copied ? 'Copiado!' : 'Copiar'}
        </button>
      </div>
      <table className="min-w-full text-sm text-white/90 border-collapse table-auto">
        <thead className="bg-white/5">
          <tr>
            {header.map((h, j) => (
              <th
                key={j}
                className="px-3 py-2 text-left font-semibold border-b border-white/10 whitespace-nowrap"
                dangerouslySetInnerHTML={{ __html: formatInline(h) }}
              />
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, r) => (
            <tr key={r} className={r % 2 === 0 ? 'bg-white/0' : 'bg-white/[0.03]'}>
              {cells.map((c, k) => (
                <td
                  key={k}
                  className="px-3 py-2 align-top border-b border-white/10"
                  dangerouslySetInnerHTML={{ __html: formatInline(c) }}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MarkdownLite({ text }: { text: string }) {
  // Split into code blocks by triple backticks
  const parts = text.split(/```/g);
  const nodes: React.ReactNode[] = [];

  parts.forEach((chunk, idx) => {
    const isCode = idx % 2 === 1; // odd indexes are code blocks
    if (isCode) {
      nodes.push(<CodeBlock key={`code-${idx}`} raw={chunk} />);
      return;
    }

    // Parse text block into paragraphs and lists
    const lines = chunk.split(/\r?\n/);
    let i = 0;
    let listBuffer: { type: 'ul' | 'ol'; items: string[] } | null = null;
    let paraBuffer: string[] = [];

    const flushPara = () => {
      if (paraBuffer.length) {
        const p = paraBuffer.join(' ');
        nodes.push(
          <p
            key={`p-${idx}-${nodes.length}`}
            className="mt-2 mb-2 text-[15px] leading-relaxed text-white/90"
            dangerouslySetInnerHTML={{ __html: formatInline(p) }}
          />
        );
        paraBuffer = [];
      }
    };

    const flushList = () => {
      if (!listBuffer) return;
      const Tag: any = listBuffer.type;
      nodes.push(
        <Tag key={`list-${idx}-${nodes.length}`} className="mt-2 mb-2 list-outside pl-5 space-y-1">
          {listBuffer.items.map((li, j) => (
            <li
              key={j}
              className="text-[15px] leading-relaxed text-white/90"
              dangerouslySetInnerHTML={{ __html: formatInline(li) }}
            />
          ))}
        </Tag>
      );
      listBuffer = null;
    };

    const isTableSeparator = (s: string) =>
      /^(\s*\|)?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+(\|\s*)?$/.test(s);
    const splitTableRow = (s: string) =>
      s
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((c) => c.trim());

    const flushTable = (rows: string[]) => {
      if (rows.length < 2) return; // need header + separator and rows
      const headerCells = splitTableRow(rows[0]);
      const bodyRows = rows.slice(1).map(splitTableRow);
      nodes.push(
        <TableBlock key={`tbl-${idx}-${nodes.length}`} header={headerCells} rows={bodyRows} />
      );
    };

    while (i < lines.length) {
      const line = lines[i];
      const ul = /^\s*[-*+]\s+(.*)$/.exec(line);
      const ol = /^\s*(\d+)\.\s+(.*)$/.exec(line);
      const heading = /^(#{1,6})\s+(.*)$/.exec(line);

      if (!line.trim()) {
        flushPara();
        flushList();
        i++;
        continue;
      }

      // Markdown table (GFM-like): header row, separator row, then data rows
      // Detect pattern: row with pipes, followed by a separator (---|---|---)
      if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        flushPara();
        flushList();
        const tableRows: string[] = [];
        // Header row
        tableRows.push(line);
        // Skip separator line
        i += 2;
        // Collect subsequent pipe-rows until blank or non-table line
        while (i < lines.length) {
          const l = lines[i];
          if (!l.trim()) break;
          if (!l.includes('|')) break;
          tableRows.push(l);
          i++;
        }
        flushTable(tableRows);
        continue;
      }

      if (heading) {
        flushPara();
        flushList();
        const level = heading[1].length;
        const Tag: any = `h${Math.min(6, level)}`;
        nodes.push(
          <Tag
            key={`h-${idx}-${i}`}
            className="mt-2 mb-2 font-semibold text-white"
            dangerouslySetInnerHTML={{ __html: formatInline(heading[2]) }}
          />
        );
        i++;
        continue;
      }

      if (ul) {
        flushPara();
        if (!listBuffer || listBuffer.type !== 'ul') listBuffer = { type: 'ul', items: [] };
        listBuffer.items.push(ul[1]);
        i++;
        continue;
      }

      if (ol) {
        flushPara();
        if (!listBuffer || listBuffer.type !== 'ol') listBuffer = { type: 'ol', items: [] };
        listBuffer.items.push(ol[2]);
        i++;
        continue;
      }

      // Paragraph text
      paraBuffer.push(line.trim());
      i++;
    }

    flushPara();
    flushList();
  });

  return <>{nodes}</>;
}

/* ----------------------- Code highlighting ----------------------- */

type LangKey =
  | 'python'
  | 'javascript'
  | 'typescript'
  | 'js'
  | 'ts'
  | 'bash'
  | 'sh'
  | 'shell'
  | 'powershell'
  | 'php'
  | 'json'
  | 'html'
  | 'xml'
  | 'css'
  | 'sql'
  | 'yaml'
  | 'yml'
  | 'unknown';

function detectLangFromHeader(firstLine: string): LangKey | null {
  const id = (firstLine || '').trim().split(/\s+/)[0].toLowerCase();
  const known: Record<string, LangKey> = {
    python: 'python',
    py: 'python',
    javascript: 'javascript',
    js: 'js',
    node: 'javascript',
    typescript: 'typescript',
    ts: 'ts',
    bash: 'bash',
    sh: 'sh',
    shell: 'shell',
    zsh: 'bash',
    powershell: 'powershell',
    ps1: 'powershell',
    pwsh: 'powershell',
    php: 'php',
    json: 'json',
    html: 'html',
    xml: 'xml',
    css: 'css',
    sql: 'sql',
    yaml: 'yaml',
    yml: 'yml',
  };
  return (known as any)[id] || null;
}

function detectLangHeuristic(code: string): LangKey {
  const s = code.trim();
  if (/^\{[\s\S]*\}$/.test(s) && /"\w+"\s*:/.test(s)) return 'json';
  if (/^<(!DOCTYPE|html|\?xml)/i.test(s) || /<\/?[a-z][^>]*>/i.test(s)) return 'html';
  if (/^\s*#\!\/.+\bpython/.test(s) || /\bdef\s+\w+\s*\(|\bimport\s+\w+/.test(s)) return 'python';
  if (/\$\w+\s*=|<\?php|\becho\b/.test(s)) return 'php';
  if (/\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b/i.test(s)) return 'sql';
  if (/\bconst\b|\blet\b|\bfunction\b|=>/.test(s)) return 'javascript';
  if (/^\s*\w+\s*\{[^}]+:\s*[^;]+;/.test(s) || /\bcolor:\s*#?[0-9a-fA-F]{3,6}/.test(s))
    return 'css';
  if (/^\s*#!/.test(s) || /\b(?:echo|fi|esac|done)\b/.test(s)) return 'bash';
  if (/\bGet-\w+\b|\$\w+\b|\bWrite-Host\b/.test(s)) return 'powershell';
  if (/^\s*\w+:\s*[^\n]+(\n\w+:\s*)+/.test(s)) return 'yaml';
  return 'unknown';
}

function highlight(code: string, lang: LangKey): string {
  // Ensure HTML-escaped first
  let src = escapeHtml(code);

  const color = {
    kw: '#7B94FF', // keywords
    fn: '#64D2FF', // functions
    str: '#E5C07B', // strings
    com: '#6A737D', // comments
    num: '#A5E075', // numbers
    prop: '#56B6C2', // property/attr
    tag: '#E06C75', // tags
  } as const;

  const span = (cls: keyof typeof color, txt: string) =>
    `<span style="color:${color[cls]}">${txt}</span>`;

  const applyCommon = (
    text: string,
    options: { lineComments?: RegExp; blockComments?: RegExp } = {}
  ) => {
    // Strings
    text = text.replace(/(\"[^\"]*\"|\'[^\']*\'|`[^`]*`)/g, (m) => span('str', m));
    // Numbers
    text = text.replace(/\b(0x[0-9a-fA-F]+|\d+\.\d+|\d+)\b/g, (m) => span('num', m));
    // Line comments
    if (options.lineComments) text = text.replace(options.lineComments, (m) => span('com', m));
    // Block comments
    if (options.blockComments) text = text.replace(options.blockComments, (m) => span('com', m));
    return text;
  };

  switch (lang) {
    case 'python': {
      src = applyCommon(src, { lineComments: /#.*/g });
      src = src.replace(
        /\b(def|class|return|if|elif|else|for|while|try|except|with|as|from|import|pass|lambda|yield|raise|True|False|None|async|await|print)\b/g,
        (m) => span('kw', m)
      );
      src = src.replace(/\b([a-zA-Z_]\w*)\s*\(/g, (m, g1) => span('fn', g1) + '(');
      break;
    }
    case 'javascript':
    case 'typescript':
    case 'js':
    case 'ts': {
      src = applyCommon(src, { lineComments: /\/\/.*$/gm, blockComments: /\/\*[\s\S]*?\*\//g });
      src = src.replace(
        /\b(const|let|var|function|return|if|else|for|while|import|from|export|class|new|try|catch|finally|true|false|null|undefined|async|await|type|interface)\b/g,
        (m) => span('kw', m)
      );
      src = src.replace(/\b([a-zA-Z_$][\w$]*)\s*\(/g, (m, g1) => span('fn', g1) + '(');
      break;
    }
    case 'bash':
    case 'sh':
    case 'shell': {
      src = applyCommon(src, { lineComments: /#.*/g });
      src = src.replace(
        /\b(if|then|fi|for|in|do|done|case|esac|function|elif|else|echo|exit)\b/g,
        (m) => span('kw', m)
      );
      break;
    }
    case 'powershell': {
      src = applyCommon(src, { lineComments: /#.*/g });
      src = src.replace(
        /\b(function|param|if|elseif|else|for|foreach|return|try|catch|finally)\b/gi,
        (m) => span('kw', m)
      );
      src = src.replace(/\b[A-Z][A-Za-z0-9]*-[A-Za-z0-9*]+\b/g, (m) => span('fn', m)); // cmdlets
      src = src.replace(/\$[a-zA-Z_][\w-]*/g, (m) => span('prop', m));
      break;
    }
    case 'php': {
      src = applyCommon(src, {
        lineComments: /(\/\/.*$|#.*$)/gm,
        blockComments: /\/\*[\s\S]*?\*\//g,
      });
      src = src.replace(/&lt;\?php|\?&gt;/g, (m) => span('tag', m));
      src = src.replace(/\$[a-zA-Z_][\w]*/g, (m) => span('prop', m));
      src = src.replace(
        /\b(function|class|public|protected|private|static|use|namespace|if|else|elseif|return|echo|try|catch)\b/g,
        (m) => span('kw', m)
      );
      src = src.replace(/\b([a-zA-Z_][\w]*)\s*\(/g, (m, g1) => span('fn', g1) + '(');
      break;
    }
    case 'json': {
      src = src.replace(/(&quot;[^&]+?&quot;)(\s*:\s*)/g, (_, k, sep) => span('prop', k) + sep);
      src = src.replace(/\b(true|false|null)\b/g, (m) => span('kw', m));
      src = src.replace(/\b(0x[0-9a-fA-F]+|\d+\.\d+|\d+)\b/g, (m) => span('num', m));
      break;
    }
    case 'html':
    case 'xml': {
      // tags/attrs
      src = src.replace(
        /(&lt;\/?)([a-zA-Z][a-zA-Z0-9:-]*)([^&]*?)(\/?&gt;)/g,
        (_, open, tag, attrs, close) => {
          const attrColored = attrs.replace(
            /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(=)(&quot;[^&]*&quot;|[^\s>&]+)/g,
            (m: any, a: any, eq: any, v: any) => `${span('prop', a)}${eq}${span('str', v)}`
          );
          return `${open}${span('tag', tag)}${attrColored}${close}`;
        }
      );
      break;
    }
    case 'css': {
      src = applyCommon(src, { blockComments: /\/\*[\s\S]*?\*\//g });
      src = src.replace(
        /([a-zA-Z0-9_\-\.\#]+)(\s*\{[\s\S]*?\})/g,
        (m, sel, block) => `${span('tag', sel)}${block}`
      );
      src = src.replace(
        /([a-zA-Z-]+)(\s*:\s*)([^;]+)(;?)/g,
        (m, p, sep, v, end) => `${span('prop', p)}${sep}${span('str', v)}${end}`
      );
      break;
    }
    case 'sql': {
      src = applyCommon(src, { lineComments: /--.*$/gm, blockComments: /\/\*[\s\S]*?\*\//g });
      src = src.replace(
        /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|PRIMARY|KEY|NOT NULL|AND|OR|AS|LIMIT|OFFSET)\b/gi,
        (m) => span('kw', m)
      );
      break;
    }
    case 'yaml':
    case 'yml': {
      src = applyCommon(src, { lineComments: /#.*/g });
      src = src.replace(/(^|\n)\s*([\w-]+):/g, (m, p, k) => `${p}${span('prop', k)}:`);
      break;
    }
    default: {
      src = applyCommon(src, { lineComments: /#.*/g, blockComments: /\/\*[\s\S]*?\*\//g });
    }
  }

  return src;
}

function CodeBlock({ raw }: { raw: string }) {
  // Support ```lang\ncode... or just ```\ncode...
  const firstNl = raw.indexOf('\n');
  const firstLine = (firstNl >= 0 ? raw.slice(0, firstNl) : raw).trim();
  let lang = detectLangFromHeader(firstLine);
  let code = raw;
  if (lang) {
    code = raw.slice(firstNl + 1);
  } else {
    lang = detectLangHeuristic(raw);
  }
  const langLabel = lang && lang !== 'unknown' ? lang : 'texto';

  const html = highlight(code, lang || 'unknown');

  const [copied, setCopied] = React.useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard?.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 900);
    } catch {}
  };

  return (
    <div className="mt-2 mb-3 overflow-hidden rounded-md bg-black/30 ring-1 ring-white/10">
      <div className="flex items-center justify-between px-3 py-1.5 text-[12px] text-white/70 bg-white/5 border-b border-white/10">
        <span className="uppercase tracking-wide">{langLabel}</span>
        <button
          onClick={onCopy}
          className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/15 ring-1 ring-white/10"
        >
          {copied ? 'Copiado' : 'Copiar'}
        </button>
      </div>
      <pre className="overflow-auto p-3 text-[13px] leading-relaxed text-white/90">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}
