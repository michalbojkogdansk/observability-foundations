import { parse, variables, equivalent, firstCounterexample, print, readAloud } from './logic.js';
import { createFormulaField, boolWord } from './ui.js';

// ---------------------------------------------------------------------------
// Task mode
// ---------------------------------------------------------------------------

const taskSentenceEl = document.getElementById('task-sentence');
const taskDictBody = document.querySelector('#task-dict tbody');
const taskProgressEl = document.getElementById('task-progress');
const taskSelectEl = document.getElementById('task-select');
const taskPrevBtn = document.getElementById('task-prev');
const taskNextBtn = document.getElementById('task-next');
const taskResultEl = document.getElementById('task-result');

let tasks = [];
let currentTaskIndex = 0;
let lastTaskParse = null;

function clearResultPanel() {
  taskResultEl.className = 'result-panel';
  taskResultEl.innerHTML = '';
}

function paragraph(text, opts = {}) {
  const p = document.createElement('p');
  p.textContent = text;
  if (opts.className) p.className = opts.className;
  return p;
}

function describeValuation(valuation, varsOrder, atomsDict) {
  return varsOrder.map((v) => `${atomsDict ? atomsDict[v] : v} = ${boolWord(valuation[v])}`).join(', ');
}

function renderTaskResult(kind, nodes) {
  taskResultEl.className = `result-panel show ${kind}`;
  taskResultEl.innerHTML = '';
  for (const node of nodes) taskResultEl.appendChild(node);
}

function checkAnswer() {
  const task = tasks[currentTaskIndex];
  if (!task) return;

  if (!lastTaskParse) {
    renderTaskResult('err', [paragraph('Wpisz formułę przed sprawdzeniem.')]);
    return;
  }
  if (!lastTaskParse.ok) {
    renderTaskResult('err', [paragraph('Błąd składni: ' + lastTaskParse.error.message)]);
    return;
  }

  const studentAst = lastTaskParse.ast;
  const allowedAtoms = Object.keys(task.atomy);
  const unknown = variables(studentAst).filter((v) => !allowedAtoms.includes(v));
  if (unknown.length) {
    const nodes = [
      paragraph(`Nieznany atom: ${unknown.join(', ')}. Słownik tego zadania zawiera tylko: ${allowedAtoms.join(', ')}.`),
    ];
    renderTaskResult('err', nodes);
    return;
  }

  const patternAst = task._patternAst;
  const eq = equivalent(studentAst, patternAst);
  if (!eq.ok) {
    renderTaskResult('err', [paragraph(eq.error)]);
    return;
  }

  if (eq.equivalent) {
    const nodes = [
      paragraph('✅ Poprawnie! Twoja formuła jest równoważna wzorcowi.'),
      paragraph(`Odczyt Twojej formuły: „${readAloud(studentAst)}”.`),
      paragraph(`Formuła wzorcowa: ${print(patternAst)}`),
    ];
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = task.uwaga;
    nodes.push(note);
    renderTaskResult('ok', nodes);
    return;
  }

  const cex = firstCounterexample(studentAst, patternAst);
  const nodes = [];
  if (cex.ok && cex.found) {
    const desc = describeValuation(cex.valuation, cex.variables, task.atomy);
    const direction = cex.valueA
      ? 'Twoja formuła daje prawdę, a wymaganie tego przypadku nie dopuszcza (powinna wyjść fałsz).'
      : 'Twoja formuła daje fałsz, a wymaganie dopuszcza ten przypadek (powinna wyjść prawda).';
    nodes.push(paragraph(`Gdy: ${desc}.`));
    nodes.push(paragraph(direction));
  } else {
    nodes.push(paragraph('Twoja formuła nie jest równoważna wzorcowi.'));
  }

  for (const bug of task.typoweBledy) {
    const bugParse = parse(bug.formula);
    if (!bugParse.ok) continue;
    const bugEq = equivalent(studentAst, bugParse.ast);
    if (bugEq.ok && bugEq.equivalent) {
      const note = document.createElement('div');
      note.className = 'note';
      note.textContent = bug.komunikat;
      nodes.push(note);
      break;
    }
  }

  renderTaskResult('err', nodes);
}

const taskField = createFormulaField({
  textareaEl: document.getElementById('task-input'),
  symbolsEl: document.getElementById('task-symbols'),
  statusEl: document.getElementById('task-status'),
  posMarkerEl: document.getElementById('task-pos-marker'),
  treeEl: document.getElementById('task-tree'),
  onResult: (result) => {
    lastTaskParse = result;
  },
});

document.getElementById('task-check').addEventListener('click', checkAnswer);

function loadTask(index) {
  currentTaskIndex = Math.max(0, Math.min(tasks.length - 1, index));
  const task = tasks[currentTaskIndex];

  taskSentenceEl.textContent = task.zdanie;
  taskDictBody.innerHTML = '';
  for (const [symbol, meaning] of Object.entries(task.atomy)) {
    const tr = document.createElement('tr');
    const tdSym = document.createElement('td');
    tdSym.className = 'symbol';
    tdSym.textContent = symbol;
    const tdMeaning = document.createElement('td');
    tdMeaning.textContent = meaning;
    tr.appendChild(tdSym);
    tr.appendChild(tdMeaning);
    taskDictBody.appendChild(tr);
  }

  taskProgressEl.textContent = `Zadanie ${currentTaskIndex + 1} z ${tasks.length}`;
  taskSelectEl.value = String(currentTaskIndex);
  taskPrevBtn.disabled = currentTaskIndex === 0;
  taskNextBtn.disabled = currentTaskIndex === tasks.length - 1;

  const input = document.getElementById('task-input');
  input.value = '';
  taskField.refresh();
  clearResultPanel();
}

taskPrevBtn.addEventListener('click', () => loadTask(currentTaskIndex - 1));
taskNextBtn.addEventListener('click', () => loadTask(currentTaskIndex + 1));
taskSelectEl.addEventListener('change', () => loadTask(Number(taskSelectEl.value)));

async function loadTasks() {
  try {
    const response = await fetch('tasks/tasks.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    tasks = await response.json();
  } catch (err) {
    taskSentenceEl.textContent =
      'Nie udało się wczytać zadań. Uruchom stronę przez lokalny serwer (np. "python3 -m http.server" w katalogu logic-trainer), a nie przez otwarcie pliku index.html bezpośrednio.';
    console.error('Błąd wczytywania tasks/tasks.json:', err);
    return;
  }

  for (const task of tasks) {
    const r = parse(task.wzorzec);
    task._patternAst = r.ok ? r.ast : null;
    if (!r.ok) console.error(`Zadanie ${task.id}: wzorzec się nie parsuje:`, r.error);
  }

  taskSelectEl.innerHTML = '';
  tasks.forEach((task, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `Zadanie ${i + 1}: ${task.zdanie.length > 40 ? task.zdanie.slice(0, 40) + '…' : task.zdanie}`;
    taskSelectEl.appendChild(opt);
  });

  loadTask(0);
}

loadTasks();
