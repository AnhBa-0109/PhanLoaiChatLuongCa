'use strict';
const $ = id => document.getElementById(id);
let selected = null;
let previewURL = null;
let busy = false;
let method = 'yolo';
let methods = null;
try { if (['yolo', 'threshold'].includes(localStorage.getItem('tunanet-method'))) method = localStorage.getItem('tunanet-method'); } catch {}
const methodRadios = [...document.querySelectorAll('input[name="method"]')];
methodRadios.forEach(radio => { radio.checked = radio.value === method; });
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
let stageItems = [];
let zoomIndex = 0;
function showZoom(index) {
  zoomIndex = (index + stageItems.length) % stageItems.length;
  const item = stageItems[zoomIndex];
  $('zoom-title').textContent = `${zoomIndex + 1}/${stageItems.length}. ${item.label}`;
  $('zoom-image').src = item.image;
  $('zoom-image').alt = item.label;
  if (!$('zoom').open) $('zoom').showModal();
}
function updateStagesHint() { $('stages-hint').textContent = $('stages-toggle').open ? 'Thu gọn' : 'Mở ra'; }
try { if (localStorage.getItem('tunanet-stages') === 'closed') $('stages-toggle').open = false; } catch {}
updateStagesHint();
$('stages-toggle').addEventListener('toggle', () => {
  updateStagesHint();
  try { localStorage.setItem('tunanet-stages', $('stages-toggle').open ? 'open' : 'closed'); } catch {}
});
$('zoom-prev').addEventListener('click', () => showZoom(zoomIndex - 1));
$('zoom-next').addEventListener('click', () => showZoom(zoomIndex + 1));
$('zoom-close').addEventListener('click', () => $('zoom').close());
$('zoom').addEventListener('click', event => { if (event.target === $('zoom')) $('zoom').close(); });
$('zoom').addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') showZoom(zoomIndex - 1);
  if (event.key === 'ArrowRight') showZoom(zoomIndex + 1);
});
function renderStages(items, label) {
  stageItems = items;
  $('stage-list').replaceChildren();
  items.forEach((item, index) => {
    const row = document.createElement('li');
    const image = document.createElement('img');
    image.src = item.image; image.alt = item.label; image.loading = 'lazy';
    const open = document.createElement('button');
    open.type = 'button'; open.className = 'stage-open';
    open.setAttribute('aria-label', `Phóng to ảnh ${item.label}`);
    open.title = 'Phóng to';
    const icon = document.createElement('i');
    icon.textContent = '⤢'; icon.setAttribute('aria-hidden', 'true');
    open.append(image, icon);
    open.addEventListener('click', () => showZoom(index));
    const caption = document.createElement('span');
    caption.textContent = `${index + 1}. ${item.label}`;
    row.append(open, caption); $('stage-list').append(row);
  });
  $('stages-method').textContent = label;
  $('stages').hidden = !items.length;
}
function resetResult() {
  if ($('zoom').open) $('zoom').close();
  stageItems = [];
  $('stages').hidden = true;
  $('stage-list').replaceChildren();
  $('result-state').classList.remove('failed');
  $('result-content').hidden = true;
  $('result-empty').hidden = false;
  $('result-empty').querySelector('h3').textContent = 'Chưa có kết quả';
  $('result-empty').querySelector('p').textContent = 'Chọn ảnh và nhấn “Phân tích ảnh”. Kết quả dự đoán sẽ xuất hiện tại đây.';
  $('result-state').textContent = selected ? 'Chờ phân tích' : 'Chờ ảnh';
}
function chooseFile(file) {
  if (busy || !file) return;
  showError('');
  selected = null;
  $('submit').disabled = true;
  $('file-details').hidden = true;
  $('file-hint').hidden = false;
  $('preview').hidden = true;
  $('empty-upload').hidden = false;
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = null; }
  resetResult();
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) return showError('Vui lòng chọn ảnh JPG, PNG, WEBP hoặc BMP.');
  if (!file.size || file.size > 10 * 1024 * 1024) return showError('Ảnh phải có dung lượng từ 1 byte đến 10 MB.');
  selected = file;
  previewURL = URL.createObjectURL(file);
  $('preview').src = previewURL;
  $('preview').hidden = false;
  $('empty-upload').hidden = true;
  $('file-details').hidden = false;
  $('file-hint').hidden = true;
  $('file-name').textContent = file.name;
  $('file-size').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Sẵn sàng gửi lên server`;
  $('submit').disabled = false;
  resetResult();
}
$('preview').addEventListener('error', () => {
  selected = null; $('submit').disabled = true; $('preview').hidden = true;
  $('empty-upload').hidden = false; $('file-details').hidden = true;
  $('file-hint').hidden = false;
  resetResult(); showError('Trình duyệt không đọc được ảnh này. Hãy chọn ảnh khác.');
});
['choose', 'replace'].forEach(id => $(id).addEventListener('click', () => { $('file-input').value = ''; $('file-input').click(); }));
$('file-input').addEventListener('change', event => chooseFile(event.target.files[0]));
$('reset').addEventListener('click', () => {
  if (busy) return;
  selected = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $('preview').hidden = true;
  $('preview').removeAttribute('src');
  $('empty-upload').hidden = false;
  $('file-details').hidden = true;
  $('file-hint').hidden = false;
  $('file-input').value = '';
  $('submit').disabled = true;
  $('processed-image').removeAttribute('src');
  $('result-content').querySelector('details').open = false;
  resetResult(); showError(''); health();
});
const zone = $('drop-zone');
['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); if (!busy) zone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', event => { if (event.dataTransfer.files.length !== 1) showError('Vui lòng chọn mỗi lần một ảnh.'); else chooseFile(event.dataTransfer.files[0]); });
const num = (value, digits = 3) => value.toFixed(digits).replace('.', ',');
const stat = item => `${num(item.mean)} ± ${num(item.std)}`;
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function makeTable(caption, headers, rows) {
  const table = node('table', undefined, 'data-table');
  table.append(node('caption', caption));
  const head = node('tr');
  headers.forEach(text => { const cell = node('th', text); cell.scope = 'col'; head.append(cell); });
  const thead = node('thead'); thead.append(head);
  const tbody = node('tbody');
  rows.forEach(row => { const line = node('tr'); row.forEach(text => line.append(node('td', text))); tbody.append(line); });
  table.append(thead, tbody);
  return table;
}
function makeConfusion(title, run) {
  const box = node('figure', undefined, 'confusion');
  box.append(node('figcaption', `${title} · ${run.name} · Accuracy ${num(run.accuracy)}`));
  const table = node('table');
  const head = node('tr');
  head.append(node('th', 'Thực tế \\ Dự đoán'));
  run.classes.forEach(name => head.append(node('th', name)));
  table.append(head);
  const max = Math.max(...run.confusion.flat());
  run.confusion.forEach((row, i) => {
    const line = node('tr');
    line.append(node('th', run.classes[i]));
    row.forEach(value => {
      const cell = node('td', String(value));
      const level = value / max;
      cell.style.background = `rgba(34,34,34,${(level * 0.85).toFixed(2)})`;
      cell.style.color = level > 0.45 ? '#fff' : '#222';
      line.append(cell);
    });
    table.append(line);
  });
  box.append(table);
  return box;
}
let dataLoaded = false;
async function loadData() {
  if (dataLoaded) return;
  $('data-notice').hidden = false;
  $('data-notice').textContent = 'Đang đọc kết quả…';
  try {
    const response = await fetch('/api/experiments');
    if (!response.ok) throw new Error();
    const data = await response.json();
    const content = $('data-content');
    content.replaceChildren();
    const ready = data.pipelines.filter(item => item.runs.length);
    if (ready.length) {
      const classes = ready[0].runs[0].classes;
      const rows = [['Accuracy', ...ready.map(item => stat(item.accuracy))], ['Macro-F1', ...ready.map(item => stat(item.macro_f1))],
        ...classes.map(name => [`F1 ${name}`, ...ready.map(item => stat(item.f1[name]))])];
      content.append(makeTable('So sánh hai phương án (trung bình ± độ lệch chuẩn, 3 seed)', ['Độ đo', ...ready.map(item => item.label)], rows));
      content.append(makeTable('Accuracy từng lần chạy', ['Phương án', 'Seed 1', 'Seed 2', 'Seed 3'], ready.map(item => [item.label, ...item.runs.map(run => num(run.accuracy))])));
      const grid = node('div', undefined, 'confusion-grid');
      ready.forEach(item => grid.append(makeConfusion(item.label, item.best)));
      content.append(node('h2', 'Ma trận nhầm lẫn của lần chạy tốt nhất', 'data-title'), grid);
    }
    if (data.iou.length) {
      content.append(makeTable('Độ khớp mask với nhãn thật', ['Tập', 'Số ảnh', 'IoU ngưỡng màu', 'IoU YOLO-seg', 'IoU < 0,8 ngưỡng màu', 'IoU < 0,8 YOLO-seg'],
        data.iou.map(row => [row.split, String(row.images), num(row.threshold), num(row.yolo), String(row.threshold_low), String(row.yolo_low)])));
    }
    if (data.white_balance.length) {
      const names = { wb_off: 'Không cân bằng trắng', wb_on: 'Có cân bằng trắng' };
      content.append(makeTable('Ảnh hưởng của cân bằng trắng (mỗi cấu hình chạy 1 lần)', ['Cấu hình', 'Accuracy', 'Macro-F1'],
        data.white_balance.map(run => [names[run.name], num(run.accuracy), num(run.macro_f1)])));
    }
    if (!ready.length && !data.iou.length && !data.white_balance.length) throw new Error();
    $('data-notice').hidden = true;
    dataLoaded = true;
  } catch {
    $('data-notice').textContent = 'Không đọc được kết quả. Kiểm tra thư mục outputs/runs rồi thử lại.';
  }
}
function showView(name) {
  $('analyze-view').hidden = name !== 'analyze';
  $('settings-view').hidden = name !== 'settings';
  $('data-view').hidden = name !== 'data';
  $('nav-analyze').classList.toggle('nav-active', name === 'analyze');
  $('nav-settings').classList.toggle('nav-on', name === 'settings');
  $('nav-data').classList.toggle('nav-on', name === 'data');
  if (name === 'analyze') $('nav-analyze').setAttribute('aria-current', 'page'); else $('nav-analyze').removeAttribute('aria-current');
  if (name === 'data') loadData();
}
$('nav-analyze').addEventListener('click', event => { event.preventDefault(); showView('analyze'); });
$('nav-settings').addEventListener('click', () => showView('settings'));
$('nav-data').addEventListener('click', () => showView('data'));
methodRadios.forEach(radio => radio.addEventListener('change', () => {
  if (busy) { methodRadios.forEach(item => { item.checked = item.value === method; }); return; }
  method = radio.value;
  try { localStorage.setItem('tunanet-method', method); } catch {}
  resetResult(); showError(''); renderStatus();
}));
function renderStatus() {
  if (!methods) return;
  for (const name of ['yolo', 'threshold']) {
    const element = $(`status-${name}`);
    element.textContent = methods[name].ready ? 'Sẵn sàng' : methods[name].message;
    element.classList.toggle('unavailable', !methods[name].ready);
  }
  const current = methods[method];
  $('connection').replaceChildren(document.createElement('i'), document.createTextNode(current.ready ? `${current.label} sẵn sàng` : `${current.label} chưa sẵn sàng`));
  $('connection').classList.toggle('ready', current.ready);
  $('model-notice').textContent = current.message || '';
  $('model-notice').hidden = current.ready;
}
async function health() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error();
    const data = await response.json();
    methods = data.methods;
    renderStatus();
  } catch {
    $('connection').textContent = 'Mất kết nối server';
    $('connection').classList.remove('ready');
    $('model-notice').textContent = 'Không kết nối được server. Kiểm tra server đang chạy rồi tải lại trang.';
    $('model-notice').hidden = false;
  }
}
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!selected || busy) return;
  busy = true; showError(''); resetResult();
  $('results-panel').hidden = false;
  document.body.classList.add('busy');
  ['submit', 'choose', 'replace', 'file-input', 'reset'].forEach(id => $(id).disabled = true);
  $('results-panel').setAttribute('aria-busy', 'true');
  $('submit-label').textContent = 'Đang phân tích…';
  $('result-state').textContent = 'Đang xử lý';
  $('result-empty').querySelector('h3').textContent = 'TunaNet đang phân tích';
  $('result-empty').querySelector('p').textContent = 'Ảnh đang được gửi và xử lý trên server. Vui lòng chờ.';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const body = new FormData(); body.append('image', selected); body.append('method', method);
    const response = await fetch('/api/predict', { method: 'POST', body, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('Server trả về phản hồi không hợp lệ. Hãy thử lại.'); }
    if (!response.ok) throw new Error(data.error || 'Không thể phân tích ảnh.');
    $('prediction').textContent = data.prediction;
    $('method-used').textContent = data.method_label;
    $('timings').textContent = data.timings.map(item => `${item.label} ${item.ms} ms`).join(' · ');
    $('timing-row').hidden = !data.timings.length;
    renderStages(data.stages, data.method_label);
    $('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    $('elapsed').textContent = `${(data.elapsed_ms / 1000).toFixed(2)} giây`;
    $('probabilities').replaceChildren();
    for (const item of data.probabilities) {
      const row = document.createElement('tr');
      const label = document.createElement('td'); label.textContent = item.label;
      const value = document.createElement('td'); value.textContent = `${(item.probability * 100).toFixed(1)}%`;
      if (item.label === data.prediction) {
        row.className = 'predicted';
        const badge = document.createElement('small'); badge.textContent = 'Dự đoán'; label.append(badge);
      }
      row.append(label, value); $('probabilities').append(row);
    }
    $('processed-image').src = data.processed_image;
    $('meat-ratio').textContent = `${(data.meat_ratio * 100).toFixed(1)}% diện tích`;
    $('warning').textContent = data.warning || ''; $('warning').hidden = !data.warning;
    $('result-content').hidden = false; $('result-empty').hidden = true;
    $('result-state').textContent = 'Hoàn tất';
  } catch (error) {
    showError(error.name === 'AbortError' ? 'Xử lý quá thời gian chờ. Vui lòng kiểm tra server và thử lại.' : (error instanceof TypeError ? 'Mất kết nối server. Vui lòng kiểm tra và thử lại.' : error.message));
    resetResult(); $('result-state').textContent = 'Không thành công';
    $('result-state').classList.add('failed');
  } finally {
    clearTimeout(timeout); busy = false; document.body.classList.remove('busy');
    ['choose', 'replace', 'file-input', 'reset'].forEach(id => $(id).disabled = false);
    $('submit').disabled = !selected;
    $('submit-label').textContent = 'Phân tích ảnh';
    $('results-panel').setAttribute('aria-busy', 'false');
    health();
  }
});
health();
