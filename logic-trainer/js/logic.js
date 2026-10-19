// Propositional logic engine: tokenizer, parser, evaluator, printer, readAloud.
// Pure functions, no DOM dependency, usable from both the browser (type="module") and Node (tests).

export const MAX_VARIABLES = 6;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

const ATOM_START = /[\p{L}_]/u;
const ATOM_PART = /[\p{L}\p{N}_]/u;

function isAtomStart(ch) {
  return ch !== undefined && ATOM_START.test(ch);
}
function isAtomPart(ch) {
  return ch !== undefined && ATOM_PART.test(ch);
}

// Tokenizes `input`, accepting both Unicode symbols (¬ ∧ ∨ →) and their ASCII
// equivalents (! ~, & &&, | ||, ->) directly, so error positions always refer
// to the original string.
export function tokenize(input) {
  const tokens = [];
  let i = 0;
  const n = input.length;
  while (i < n) {
    const ch = input[i];
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
      i++;
      continue;
    }
    if (ch === '(') {
      tokens.push({ type: 'LPAREN', value: '(', pos: i });
      i++;
      continue;
    }
    if (ch === ')') {
      tokens.push({ type: 'RPAREN', value: ')', pos: i });
      i++;
      continue;
    }
    if (ch === '¬' || ch === '!' || ch === '~') {
      tokens.push({ type: 'NOT', value: ch, pos: i });
      i++;
      continue;
    }
    if (ch === '∧') {
      tokens.push({ type: 'AND', value: ch, pos: i });
      i++;
      continue;
    }
    if (ch === '&') {
      const two = input.slice(i, i + 2) === '&&';
      tokens.push({ type: 'AND', value: two ? '&&' : '&', pos: i });
      i += two ? 2 : 1;
      continue;
    }
    if (ch === '∨') {
      tokens.push({ type: 'OR', value: ch, pos: i });
      i++;
      continue;
    }
    if (ch === '|') {
      const two = input.slice(i, i + 2) === '||';
      tokens.push({ type: 'OR', value: two ? '||' : '|', pos: i });
      i += two ? 2 : 1;
      continue;
    }
    if (ch === '→') {
      tokens.push({ type: 'IMPL', value: ch, pos: i });
      i++;
      continue;
    }
    if (ch === '-' && input[i + 1] === '>') {
      tokens.push({ type: 'IMPL', value: '->', pos: i });
      i += 2;
      continue;
    }
    if (ch === '⊥' || ch === '0') {
      tokens.push({ type: 'FALSUM', value: ch, pos: i });
      i++;
      continue;
    }
    if (isAtomStart(ch)) {
      let j = i + 1;
      while (j < n && isAtomPart(input[j])) j++;
      tokens.push({ type: 'ATOM', value: input.slice(i, j), pos: i });
      i = j;
      continue;
    }
    return { ok: false, error: { message: `Nieznany znak '${ch}' na pozycji ${i}.`, pos: i } };
  }
  tokens.push({ type: 'EOF', value: '', pos: n });
  return { ok: true, tokens };
}

// ---------------------------------------------------------------------------
// Parser (recursive descent, precedence: NOT > (AND/OR, equal) > IMPL)
// IMPL is right-associative. AND/OR chains of the SAME operator are
// left-associative; mixing AND and OR at the same level without explicit
// parentheses is rejected as ambiguous.
// ---------------------------------------------------------------------------

class ParseError extends Error {
  constructor(message, pos) {
    super(message);
    this.pos = pos;
  }
}

function parseImpl(ctx) {
  const left = parseAndOr(ctx);
  const tok = ctx.peek();
  if (tok.type === 'IMPL') {
    ctx.next();
    const right = parseImpl(ctx); // right-associative
    return { type: 'impl', left, right };
  }
  return left;
}

function parseAndOr(ctx) {
  let node = parseUnary(ctx);
  let chainOp = null; // 'AND' or 'OR', whichever this chain started with
  let chainStartPos = ctx.peek().pos;
  for (;;) {
    const tok = ctx.peek();
    if (tok.type !== 'AND' && tok.type !== 'OR') break;
    if (chainOp === null) {
      chainOp = tok.type;
    } else if (tok.type !== chainOp) {
      throw new ParseError(
        `Niejednoznaczne: dodaj nawiasy wokół wyrażenia zawierającego ∧ i ∨ (pozycja ${tok.pos}). ` +
          `Nie można mieszać ∧ i ∨ bez nawiasów, bo mają ten sam priorytet — zapisz np. (a ∨ b) ∧ c albo a ∨ (b ∧ c).`,
        tok.pos
      );
    }
    const opToken = ctx.next();
    const right = parseUnary(ctx);
    node = { type: opToken.type === 'AND' ? 'and' : 'or', left: node, right };
  }
  return node;
}

function parseUnary(ctx) {
  const tok = ctx.peek();
  if (tok.type === 'NOT') {
    ctx.next();
    const arg = parseUnary(ctx);
    return { type: 'not', arg };
  }
  return parseAtomOrGroup(ctx);
}

function parseAtomOrGroup(ctx) {
  const tok = ctx.peek();
  if (tok.type === 'ATOM') {
    ctx.next();
    return { type: 'atom', name: tok.value };
  }
  if (tok.type === 'FALSUM') {
    ctx.next();
    return { type: 'falsum' };
  }
  if (tok.type === 'LPAREN') {
    const openPos = tok.pos;
    ctx.next();
    if (ctx.peek().type === 'RPAREN') {
      throw new ParseError(`Brak formuły wewnątrz nawiasów otwartych na pozycji ${openPos}.`, openPos + 1);
    }
    const inner = parseImpl(ctx);
    const close = ctx.peek();
    if (close.type !== 'RPAREN') {
      throw new ParseError(
        `Brakuje nawiasu zamykającego ')' dla nawiasu otwartego na pozycji ${openPos}.`,
        close.pos
      );
    }
    ctx.next();
    return inner;
  }
  if (tok.type === 'EOF') {
    throw new ParseError(`Brak argumentu na pozycji ${tok.pos} — formuła jest niekompletna.`, tok.pos);
  }
  if (tok.type === 'RPAREN') {
    throw new ParseError(`Nieoczekiwany nawias ')' na pozycji ${tok.pos}.`, tok.pos);
  }
  throw new ParseError(`Nieoczekiwany znak na pozycji ${tok.pos}.`, tok.pos);
}

function makeCtx(tokens) {
  let i = 0;
  return {
    peek() {
      return tokens[i];
    },
    next() {
      return tokens[i++];
    },
  };
}

// parse(input) -> { ok: true, ast } | { ok: false, error: { message, pos } }
export function parse(input) {
  if (input == null || input.trim() === '') {
    return { ok: false, error: { message: 'Formuła jest pusta.', pos: 0 } };
  }
  const tok = tokenize(input);
  if (!tok.ok) return tok;
  const ctx = makeCtx(tok.tokens);
  try {
    const ast = parseImpl(ctx);
    const trailing = ctx.peek();
    if (trailing.type !== 'EOF') {
      throw new ParseError(`Nieoczekiwane dane po formule, na pozycji ${trailing.pos}.`, trailing.pos);
    }
    return { ok: true, ast };
  } catch (e) {
    if (e instanceof ParseError) {
      return { ok: false, error: { message: e.message, pos: e.pos } };
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------

export function variables(ast) {
  const set = new Set();
  (function walk(node) {
    if (node.type === 'atom') set.add(node.name);
    else if (node.type === 'falsum') {
      // nullary constant — contributes no variable
    } else if (node.type === 'not') walk(node.arg);
    else {
      walk(node.left);
      walk(node.right);
    }
  })(ast);
  return [...set].sort((a, b) => a.localeCompare(b, 'pl'));
}

export function unknownAtoms(ast, allowedAtoms) {
  const allowed = new Set(allowedAtoms);
  return variables(ast).filter((v) => !allowed.has(v));
}

export function evaluate(ast, valuation) {
  switch (ast.type) {
    case 'falsum':
      return false;
    case 'atom': {
      const v = valuation[ast.name];
      if (typeof v !== 'boolean') {
        throw new Error(`Brak wartości dla atomu '${ast.name}' w ocenianej walidacji.`);
      }
      return v;
    }
    case 'not':
      return !evaluate(ast.arg, valuation);
    case 'and':
      return evaluate(ast.left, valuation) && evaluate(ast.right, valuation);
    case 'or':
      return evaluate(ast.left, valuation) || evaluate(ast.right, valuation);
    case 'impl':
      return !evaluate(ast.left, valuation) || evaluate(ast.right, valuation);
    default:
      throw new Error(`Nieznany typ węzła AST: ${ast.type}`);
  }
}

// Enumerates all 2^n valuations in canonical order: first variable changes
// slowest, true before false (so the first row is all-true, the last all-false).
function enumerateValuations(vars) {
  const n = vars.length;
  const rows = [];
  const total = 2 ** n;
  for (let i = 0; i < total; i++) {
    const valuation = {};
    for (let j = 0; j < n; j++) {
      const bit = (i >> (n - 1 - j)) & 1;
      valuation[vars[j]] = bit === 0;
    }
    rows.push(valuation);
  }
  return rows;
}

export function truthTable(ast) {
  const vars = variables(ast);
  if (vars.length > MAX_VARIABLES) {
    return {
      ok: false,
      error: `Formuła ma ${vars.length} zmiennych — tabela prawdy obsługuje maksymalnie ${MAX_VARIABLES}.`,
    };
  }
  const rows = enumerateValuations(vars).map((valuation) => ({
    valuation,
    value: evaluate(ast, valuation),
  }));
  return { ok: true, variables: vars, rows };
}

export function classify(ast) {
  const table = truthTable(ast);
  if (!table.ok) return table;
  const allTrue = table.rows.every((r) => r.value === true);
  const allFalse = table.rows.every((r) => r.value === false);
  if (allTrue) return { ok: true, kind: 'tautology' };
  if (allFalse) return { ok: true, kind: 'contradiction' };
  return { ok: true, kind: 'satisfiable' };
}

export function equivalent(astA, astB) {
  const vars = [...new Set([...variables(astA), ...variables(astB)])].sort((a, b) => a.localeCompare(b, 'pl'));
  if (vars.length > MAX_VARIABLES) {
    return {
      ok: false,
      error: `Formuły mają łącznie ${vars.length} zmiennych — porównanie obsługuje maksymalnie ${MAX_VARIABLES}.`,
    };
  }
  for (const valuation of enumerateValuations(vars)) {
    if (evaluate(astA, valuation) !== evaluate(astB, valuation)) {
      return { ok: true, equivalent: false };
    }
  }
  return { ok: true, equivalent: true };
}

export function firstCounterexample(astA, astB) {
  const vars = [...new Set([...variables(astA), ...variables(astB)])].sort((a, b) => a.localeCompare(b, 'pl'));
  if (vars.length > MAX_VARIABLES) {
    return {
      ok: false,
      error: `Formuły mają łącznie ${vars.length} zmiennych — porównanie obsługuje maksymalnie ${MAX_VARIABLES}.`,
    };
  }
  for (const valuation of enumerateValuations(vars)) {
    const valueA = evaluate(astA, valuation);
    const valueB = evaluate(astB, valuation);
    if (valueA !== valueB) {
      return { ok: true, found: true, valuation, valueA, valueB, variables: vars };
    }
  }
  return { ok: true, found: false };
}

// ---------------------------------------------------------------------------
// Printer — minimal parentheses, round-trip safe under the grammar above.
// ---------------------------------------------------------------------------

function printNode(node) {
  switch (node.type) {
    case 'falsum':
      return '⊥';
    case 'atom':
      return node.name;
    case 'not': {
      const inner = printNode(node.arg);
      const needsParens = node.arg.type === 'and' || node.arg.type === 'or' || node.arg.type === 'impl';
      return '¬' + (needsParens ? `(${inner})` : inner);
    }
    case 'and':
    case 'or': {
      const op = node.type === 'and' ? '∧' : '∨';
      const leftNeedsParens =
        node.left.type === 'impl' || ((node.left.type === 'and' || node.left.type === 'or') && node.left.type !== node.type);
      const rightNeedsParens = node.right.type === 'impl' || node.right.type === 'and' || node.right.type === 'or';
      const leftStr = printNode(node.left);
      const rightStr = printNode(node.right);
      return `${leftNeedsParens ? `(${leftStr})` : leftStr} ${op} ${rightNeedsParens ? `(${rightStr})` : rightStr}`;
    }
    case 'impl': {
      const leftNeedsParens = node.left.type === 'impl';
      const leftStr = printNode(node.left);
      const rightStr = printNode(node.right);
      return `${leftNeedsParens ? `(${leftStr})` : leftStr} → ${rightStr}`;
    }
    default:
      throw new Error(`Nieznany typ węzła AST: ${node.type}`);
  }
}

export function print(ast) {
  return printNode(ast);
}

// ---------------------------------------------------------------------------
// readAloud — spoken Polish rendering, same minimal-parens strategy as print.
// ---------------------------------------------------------------------------

function readNode(node) {
  switch (node.type) {
    case 'falsum':
      return 'fałsz';
    case 'atom':
      return node.name;
    case 'not': {
      const inner = readNode(node.arg);
      const needsParens = node.arg.type === 'and' || node.arg.type === 'or' || node.arg.type === 'impl';
      return 'nie ' + (needsParens ? `(${inner})` : inner);
    }
    case 'and':
    case 'or': {
      const word = node.type === 'and' ? 'i' : 'lub';
      const leftNeedsParens =
        node.left.type === 'impl' || ((node.left.type === 'and' || node.left.type === 'or') && node.left.type !== node.type);
      const rightNeedsParens = node.right.type === 'impl' || node.right.type === 'and' || node.right.type === 'or';
      const leftStr = readNode(node.left);
      const rightStr = readNode(node.right);
      return `${leftNeedsParens ? `(${leftStr})` : leftStr} ${word} ${rightNeedsParens ? `(${rightStr})` : rightStr}`;
    }
    case 'impl': {
      const leftStr = readNode(node.left);
      const rightStr = readNode(node.right);
      return `jeśli ${leftStr}, to ${rightStr}`;
    }
    default:
      throw new Error(`Nieznany typ węzła AST: ${node.type}`);
  }
}

export function readAloud(ast) {
  return readNode(ast);
}
