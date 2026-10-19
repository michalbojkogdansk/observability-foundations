// DOM glue: symbol buttons, live parse status + tree.
// Depends only on js/logic.js — no framework.

import { parse } from './logic.js';

const SYMBOLS = ['¬', '∧', '∨', '→', '⊥', '(', ')'];

export function boolWord(b) {
  return b ? 'prawda' : 'fałsz';
}

export function insertAtCursor(el, text) {
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? el.value.length;
  if (typeof el.setRangeText === 'function') {
    el.setRangeText(text, start, end, 'end');
  } else {
    el.value = el.value.slice(0, start) + text + el.value.slice(end);
    const pos = start + text.length;
    el.setSelectionRange(pos, pos);
  }
  el.focus();
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

export function buildSymbolButtons(container, textareaEl) {
  container.innerHTML = '';
  for (const sym of SYMBOLS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = sym;
    btn.addEventListener('click', () => insertAtCursor(textareaEl, sym));
    container.appendChild(btn);
  }
}

function nodeLabel(node) {
  switch (node.type) {
    case 'atom':
      return node.name;
    case 'falsum':
      return '⊥';
    case 'not':
      return '¬';
    case 'and':
      return '∧';
    case 'or':
      return '∨';
    case 'impl':
      return '→';
    default:
      return '?';
  }
}

function nodeChildren(node) {
  switch (node.type) {
    case 'atom':
    case 'falsum':
      return [];
    case 'not':
      return [node.arg];
    default:
      return [node.left, node.right];
  }
}

function renderTreeNode(node) {
  const wrap = document.createElement('div');
  wrap.className = 'tree-node';
  const label = document.createElement('span');
  label.className = node.type === 'atom' || node.type === 'falsum' ? 'tree-label' : 'tree-label op';
  label.textContent = nodeLabel(node);
  wrap.appendChild(label);
  const children = nodeChildren(node);
  if (children.length) {
    const childrenWrap = document.createElement('div');
    childrenWrap.className = 'tree-children';
    for (const child of children) childrenWrap.appendChild(renderTreeNode(child));
    wrap.appendChild(childrenWrap);
  }
  return wrap;
}

export function renderTree(container, ast) {
  container.innerHTML = '';
  if (ast) container.appendChild(renderTreeNode(ast));
}

export function renderPosMarker(el, pos) {
  if (typeof pos !== 'number') {
    el.textContent = '';
    return;
  }
  el.textContent = ' '.repeat(pos) + '^';
}

// Wires one formula textarea to: symbol buttons, live parse status, position
// marker on error, parse tree, and an onResult(parseResult|null) callback.
export function createFormulaField({ textareaEl, symbolsEl, statusEl, posMarkerEl, treeEl, onResult }) {
  if (symbolsEl) buildSymbolButtons(symbolsEl, textareaEl);

  function refresh() {
    const value = textareaEl.value;
    const trimmedEmpty = value.trim() === '';
    const result = trimmedEmpty ? null : parse(value);

    if (trimmedEmpty) {
      statusEl.textContent = '';
      statusEl.className = 'status-line';
      if (posMarkerEl) posMarkerEl.textContent = '';
      if (treeEl) treeEl.innerHTML = '';
    } else if (result.ok) {
      statusEl.textContent = '✓ Składnia poprawna';
      statusEl.className = 'status-line ok';
      if (posMarkerEl) posMarkerEl.textContent = '';
      if (treeEl) renderTree(treeEl, result.ast);
    } else {
      statusEl.textContent = '✗ ' + result.error.message;
      statusEl.className = 'status-line err';
      if (posMarkerEl) renderPosMarker(posMarkerEl, result.error.pos);
      if (treeEl) treeEl.innerHTML = '';
    }

    onResult(result);
  }

  textareaEl.addEventListener('input', refresh);
  refresh();
  return { refresh };
}
