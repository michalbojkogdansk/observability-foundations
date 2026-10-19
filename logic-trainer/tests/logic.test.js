import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  parse,
  tokenize,
  evaluate,
  variables,
  unknownAtoms,
  truthTable,
  classify,
  equivalent,
  firstCounterexample,
  print,
  readAloud,
  MAX_VARIABLES,
} from '../js/logic.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const tasks = JSON.parse(readFileSync(join(__dirname, '../tasks/tasks.json'), 'utf8'));

function ok(input) {
  const r = parse(input);
  assert.equal(r.ok, true, `expected '${input}' to parse, got error: ${r.ok ? '' : r.error.message}`);
  return r.ast;
}

function fails(input) {
  const r = parse(input);
  assert.equal(r.ok, false, `expected '${input}' to fail to parse`);
  return r.error;
}

// ---------------------------------------------------------------------------
// Tokenizer / ASCII normalization
// ---------------------------------------------------------------------------

test('tokenizer recognizes unicode operators', () => {
  const r = tokenize('¬p ∧ q ∨ r → s');
  assert.equal(r.ok, true);
  assert.deepEqual(
    r.tokens.map((t) => t.type),
    ['NOT', 'ATOM', 'AND', 'ATOM', 'OR', 'ATOM', 'IMPL', 'ATOM', 'EOF']
  );
});

test('tokenizer normalizes ASCII synonyms to the right token types', () => {
  const r = tokenize('!p & q | r -> s');
  assert.equal(r.ok, true);
  assert.deepEqual(
    r.tokens.map((t) => t.type),
    ['NOT', 'ATOM', 'AND', 'ATOM', 'OR', 'ATOM', 'IMPL', 'ATOM', 'EOF']
  );
});

test('tokenizer accepts && || and ~ too', () => {
  const r = tokenize('~p && q || r');
  assert.equal(r.ok, true);
  assert.deepEqual(
    r.tokens.map((t) => t.type),
    ['NOT', 'ATOM', 'AND', 'ATOM', 'OR', 'ATOM', 'EOF']
  );
});

test('ASCII and unicode formulas parse to the same AST', () => {
  const a = ok('!p & q -> r');
  const b = ok('¬p ∧ q → r');
  assert.deepEqual(a, b);
});

test('multi-character atoms like p1, q2, serwer are accepted', () => {
  const ast = ok('p1 ∧ q2 ∧ serwer');
  assert.deepEqual(variables(ast), ['p1', 'q2', 'serwer']);
});

// ---------------------------------------------------------------------------
// Precedence and associativity
// ---------------------------------------------------------------------------

test('NOT binds tighter than AND/OR', () => {
  assert.deepEqual(ok('¬p ∧ q'), { type: 'and', left: { type: 'not', arg: { type: 'atom', name: 'p' } }, right: { type: 'atom', name: 'q' } });
});

test('AND/OR bind tighter than IMPL', () => {
  assert.deepEqual(ok('p ∧ q → r'), {
    type: 'impl',
    left: { type: 'and', left: { type: 'atom', name: 'p' }, right: { type: 'atom', name: 'q' } },
    right: { type: 'atom', name: 'r' },
  });
});

test('IMPL is right-associative: a -> b -> c parses as a -> (b -> c)', () => {
  assert.deepEqual(ok('a → b → c'), {
    type: 'impl',
    left: { type: 'atom', name: 'a' },
    right: { type: 'impl', left: { type: 'atom', name: 'b' }, right: { type: 'atom', name: 'c' } },
  });
});

test('chained AND is left-associative', () => {
  assert.deepEqual(ok('a ∧ b ∧ c'), {
    type: 'and',
    left: { type: 'and', left: { type: 'atom', name: 'a' }, right: { type: 'atom', name: 'b' } },
    right: { type: 'atom', name: 'c' },
  });
});

test('chained OR is left-associative', () => {
  assert.deepEqual(ok('a ∨ b ∨ c'), {
    type: 'or',
    left: { type: 'or', left: { type: 'atom', name: 'a' }, right: { type: 'atom', name: 'b' } },
    right: { type: 'atom', name: 'c' },
  });
});

test('mixing AND and OR without parentheses is rejected as ambiguous', () => {
  const err = fails('a ∨ b ∧ c');
  assert.match(err.message, /Niejednoznaczne/);
  assert.equal(typeof err.pos, 'number');
});

test('mixing AND and OR the other way round is also rejected', () => {
  const err = fails('a ∧ b ∨ c');
  assert.match(err.message, /Niejednoznaczne/);
});

test('explicit parentheses resolve the ambiguity', () => {
  ok('(a ∨ b) ∧ c');
  ok('a ∨ (b ∧ c)');
});

test('double negation and parenthesized negation of a binary formula', () => {
  assert.deepEqual(ok('¬¬p'), { type: 'not', arg: { type: 'not', arg: { type: 'atom', name: 'p' } } });
  ok('¬(p ∧ q)');
});

// ---------------------------------------------------------------------------
// Syntax errors with position
// ---------------------------------------------------------------------------

test('empty formula is rejected', () => {
  const err = fails('');
  assert.match(err.message, /pusta/);
});

test('unknown character is rejected with its position', () => {
  const err = fails('p # q');
  assert.equal(err.pos, 2);
});

test('missing closing parenthesis is rejected', () => {
  const err = fails('(p ∧ q');
  assert.match(err.message, /nawiasu zamykającego/);
});

test('missing argument after a binary operator is rejected', () => {
  const err = fails('p ∧');
  assert.match(err.message, /Brak argumentu/);
});

test('missing argument after NOT is rejected', () => {
  const err = fails('¬');
  assert.match(err.message, /Brak argumentu/);
});

test('empty parentheses are rejected', () => {
  const err = fails('()');
  assert.equal(typeof err.pos, 'number');
});

test('trailing garbage after a valid formula is rejected', () => {
  const err = fails('p ∧ q)');
  assert.match(err.message, /Nieoczekiwane/);
});

test('parser returns an error object, never throws, on malformed input', () => {
  const samples = ['', '(', ')', '∧', 'p ∧ ∨ q', '→', 'p →', '((p)', 'p)(q'];
  for (const s of samples) {
    assert.doesNotThrow(() => parse(s), `parse('${s}') should not throw`);
  }
});

// ---------------------------------------------------------------------------
// evaluate / variables / unknownAtoms
// ---------------------------------------------------------------------------

test('evaluate: AND, OR, NOT, IMPL truth tables', () => {
  const and = ok('p ∧ q');
  assert.equal(evaluate(and, { p: true, q: true }), true);
  assert.equal(evaluate(and, { p: true, q: false }), false);

  const or = ok('p ∨ q');
  assert.equal(evaluate(or, { p: false, q: false }), false);
  assert.equal(evaluate(or, { p: true, q: false }), true);

  const not = ok('¬p');
  assert.equal(evaluate(not, { p: true }), false);
  assert.equal(evaluate(not, { p: false }), true);

  const impl = ok('p → q');
  assert.equal(evaluate(impl, { p: true, q: false }), false);
  assert.equal(evaluate(impl, { p: false, q: false }), true);
  assert.equal(evaluate(impl, { p: false, q: true }), true);
  assert.equal(evaluate(impl, { p: true, q: true }), true);
});

test('variables returns a sorted, deduplicated list', () => {
  assert.deepEqual(variables(ok('b ∧ a ∧ b ∧ c')), ['a', 'b', 'c']);
});

test('unknownAtoms reports atoms outside the allowed dictionary', () => {
  const ast = ok('p ∧ q ∧ r');
  assert.deepEqual(unknownAtoms(ast, ['p', 'q']), ['r']);
  assert.deepEqual(unknownAtoms(ast, ['p', 'q', 'r']), []);
});

// ---------------------------------------------------------------------------
// truthTable / classify
// ---------------------------------------------------------------------------

test('truthTable has 2^n rows and the canonical T..F ordering', () => {
  const table = truthTable(ok('p ∧ q'));
  assert.equal(table.ok, true);
  assert.equal(table.rows.length, 4);
  assert.deepEqual(table.rows[0].valuation, { p: true, q: true });
  assert.deepEqual(table.rows[3].valuation, { p: false, q: false });
});

test('truthTable rejects formulas with more than MAX_VARIABLES variables', () => {
  const ast = ok('a ∧ b ∧ c ∧ d ∧ e ∧ f ∧ g');
  assert.equal(variables(ast).length, 7);
  assert.ok(7 > MAX_VARIABLES);
  const table = truthTable(ast);
  assert.equal(table.ok, false);
});

test('classify recognizes tautology, contradiction, satisfiable', () => {
  assert.equal(classify(ok('p ∨ ¬p')).kind, 'tautology');
  assert.equal(classify(ok('p ∧ ¬p')).kind, 'contradiction');
  assert.equal(classify(ok('p ∧ q')).kind, 'satisfiable');
});

// ---------------------------------------------------------------------------
// equivalent / firstCounterexample
// ---------------------------------------------------------------------------

test('equivalent: De Morgan law holds', () => {
  const r = equivalent(ok('¬(p ∧ q)'), ok('¬p ∨ ¬q'));
  assert.equal(r.ok, true);
  assert.equal(r.equivalent, true);
});

test('equivalent: de Morgan is NOT the same as distributing negation over AND', () => {
  const r = equivalent(ok('¬(p ∧ q)'), ok('¬p ∧ ¬q'));
  assert.equal(r.equivalent, false);
});

test('equivalent: contrapositive of an implication', () => {
  const r = equivalent(ok('p → q'), ok('¬q → ¬p'));
  assert.equal(r.equivalent, true);
});

test('firstCounterexample finds a differing row and reports both values', () => {
  const r = firstCounterexample(ok('¬(s ∧ b)'), ok('¬s ∧ ¬b'));
  assert.equal(r.ok, true);
  assert.equal(r.found, true);
  assert.equal(evaluate(ok('¬(s ∧ b)'), r.valuation), r.valueA);
  assert.equal(evaluate(ok('¬s ∧ ¬b'), r.valuation), r.valueB);
  assert.notEqual(r.valueA, r.valueB);
});

test('firstCounterexample reports found:false for equivalent formulas', () => {
  const r = firstCounterexample(ok('p → q'), ok('¬p ∨ q'));
  assert.equal(r.ok, true);
  assert.equal(r.found, false);
});

// ---------------------------------------------------------------------------
// print: minimal parentheses, round-trip safe
// ---------------------------------------------------------------------------

const ROUND_TRIP_SAMPLES = [
  'p',
  '¬p',
  '¬¬p',
  'p ∧ q',
  'p ∨ q',
  'p ∧ q ∧ r',
  'p ∨ q ∨ r',
  '(p ∨ q) ∧ r',
  'p ∧ (q ∨ r)',
  'p → q',
  'a → b → c',
  '(a → b) → c',
  '¬(p ∧ q)',
  '¬(p ∨ q)',
  '(p ∧ q) → r',
  'p → (q ∧ r)',
  '(a ∨ u) ∧ k → d',
  'a ∨ (u ∧ k) → d',
];

test('print round-trips through parse for a battery of formulas', () => {
  for (const sample of ROUND_TRIP_SAMPLES) {
    const ast1 = ok(sample);
    const printed = print(ast1);
    const ast2 = ok(printed);
    assert.deepEqual(ast2, ast1, `round-trip failed for '${sample}' -> '${printed}'`);
  }
});

test('print uses minimal parentheses for a left-associative AND chain', () => {
  assert.equal(print(ok('a ∧ b ∧ c')), 'a ∧ b ∧ c');
});

test('print preserves grouping that changes meaning', () => {
  assert.equal(print(ok('(a ∨ b) ∧ c')), '(a ∨ b) ∧ c');
  assert.equal(print(ok('a ∨ (b ∧ c)')), 'a ∨ (b ∧ c)');
});

test('print omits parentheses for right-associative IMPL chains', () => {
  assert.equal(print(ok('a → b → c')), 'a → b → c');
});

test('print keeps parentheses when IMPL is nested on the left', () => {
  assert.equal(print(ok('(a → b) → c')), '(a → b) → c');
});

// ---------------------------------------------------------------------------
// readAloud
// ---------------------------------------------------------------------------

test('readAloud renders the documented example', () => {
  const ast = ok('(A ∨ B) ∧ ¬C → D');
  assert.equal(readAloud(ast), 'jeśli (A lub B) i nie C, to D');
});

test('readAloud renders simple connectives', () => {
  assert.equal(readAloud(ok('p ∧ q')), 'p i q');
  assert.equal(readAloud(ok('p ∨ q')), 'p lub q');
  assert.equal(readAloud(ok('¬p')), 'nie p');
});

// ---------------------------------------------------------------------------
// falsum (⊥) — a nullary constant, not a variable
// ---------------------------------------------------------------------------

test('⊥ and its ASCII alias 0 parse to the same falsum node', () => {
  assert.deepEqual(ok('⊥'), { type: 'falsum' });
  assert.deepEqual(ok('0'), { type: 'falsum' });
});

test('falsum combines with connectives like any other formula', () => {
  assert.deepEqual(ok('p → ⊥'), { type: 'impl', left: { type: 'atom', name: 'p' }, right: { type: 'falsum' } });
  assert.deepEqual(ok('(p ∧ q) → 0'), ok('(p ∧ q) → ⊥'));
  ok('¬⊥');
  ok('⊥ ∧ p');
});

test('evaluate: falsum is always false, regardless of valuation', () => {
  assert.equal(evaluate(ok('⊥'), {}), false);
  assert.equal(evaluate(ok('p ∨ ⊥'), { p: true }), true);
  assert.equal(evaluate(ok('p ∨ ⊥'), { p: false }), false);
});

test('variables() excludes falsum — it is a constant, not a propositional variable', () => {
  assert.deepEqual(variables(ok('⊥ → s')), ['s']);
  assert.deepEqual(variables(ok('⊥')), []);
});

test('unknownAtoms never flags ⊥, even against an empty dictionary', () => {
  assert.deepEqual(unknownAtoms(ok('⊥ → s'), ['s']), []);
  assert.deepEqual(unknownAtoms(ok('⊥'), []), []);
});

test('classify: ⊥ → p is a tautology (ex falso quodlibet)', () => {
  assert.equal(classify(ok('⊥ → p')).kind, 'tautology');
});

test('classify: p ∧ ⊥ is a contradiction', () => {
  assert.equal(classify(ok('p ∧ ⊥')).kind, 'contradiction');
});

test('print and readAloud render falsum as ⊥ / "fałsz"', () => {
  assert.equal(print(ok('p → ⊥')), 'p → ⊥');
  assert.equal(readAloud(ok('p → ⊥')), 'jeśli p, to fałsz');
});

test('print round-trips formulas containing falsum', () => {
  for (const sample of ['⊥', 'p → ⊥', '⊥ → p', '(p ∧ q) → ⊥', '¬⊥']) {
    const ast1 = ok(sample);
    const ast2 = ok(print(ast1));
    assert.deepEqual(ast2, ast1, `round-trip failed for '${sample}'`);
  }
});

// ---------------------------------------------------------------------------
// tasks.json integrity
// ---------------------------------------------------------------------------

test('every task pattern formula parses', () => {
  for (const task of tasks) {
    const r = parse(task.wzorzec);
    assert.equal(r.ok, true, `task ${task.id}: pattern '${task.wzorzec}' should parse`);
  }
});

test('every task pattern only uses atoms from its own dictionary', () => {
  for (const task of tasks) {
    const ast = ok(task.wzorzec);
    const unknown = unknownAtoms(ast, Object.keys(task.atomy));
    assert.deepEqual(unknown, [], `task ${task.id}: pattern uses undeclared atoms`);
  }
});

test('every typical-error formula parses and is NOT equivalent to the pattern', () => {
  for (const task of tasks) {
    const patternAst = ok(task.wzorzec);
    for (const bug of task.typoweBledy) {
      const r = parse(bug.formula);
      assert.equal(r.ok, true, `task ${task.id}: typical error '${bug.formula}' should parse`);
      const eq = equivalent(patternAst, r.ast);
      assert.equal(eq.ok, true);
      assert.equal(
        eq.equivalent,
        false,
        `task ${task.id}: typical error '${bug.formula}' should NOT be equivalent to pattern '${task.wzorzec}'`
      );
    }
  }
});

test('tasks.json has the expected exercises with required fields', () => {
  assert.equal(tasks.length, 11);
  for (const task of tasks) {
    assert.ok(task.id);
    assert.ok(task.zdanie);
    assert.ok(task.atomy && Object.keys(task.atomy).length > 0);
    assert.ok(task.wzorzec);
    assert.ok(task.uwaga);
    assert.ok(Array.isArray(task.typoweBledy));
  }
});
