(() => {
  const root = document.querySelector('.evidence-diagnostics[data-observations]');
  if (!root) return;
  const number = value => typeof value !== 'number' ? String(value) : Math.abs(value) > 1e8 ? value.toPrecision(17) : value !== 0 && Math.abs(value) < 1e-4 ? value.toExponential(3) : value.toLocaleString(undefined, {maximumFractionDigits: 5});
  const percent = value => (100 * value).toFixed(1) + '%';
  const select = (values, change) => {
    const element = document.createElement('select');
    for (const [key, label] of values) element.add(new Option(label, key));
    element.onchange = change;
    return element;
  };
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  const metric = (label, value) => {
    const element = node('div', undefined, 'explorer-metric');
    element.append(node('strong', value), node('span', label));
    return element;
  };
  function records(box, data) {
    box.append(node('h4', 'Inspect a captured prediction'));
    box.append(node('p', 'Examples use the first captured coordinates in execution order. Answers are shown for explanation; the proof hides its targets when predicting.', 'explorer-note'));
    const controls = node('div', undefined, 'explorer-controls');
    const condition = select(data.conditions.map((value, i) => [String(i), value.label]), () => {populate(); update();});
    condition.setAttribute('aria-label', 'Captured condition');
    const example = select([], update);
    example.setAttribute('aria-label', 'Captured example coordinate');
    controls.append(condition, example);
    const values = node('div', undefined, 'explorer-metrics');
    values.setAttribute('aria-live', 'polite');
    const details = node('details', undefined, 'captured-record');
    details.append(node('summary', 'Captured source record, with evaluator answers'));
    const record = node('pre'); details.append(record);
    const status = node('p', undefined, 'explorer-note');
    box.append(controls, values, status, details);
    function populate() {
      example.replaceChildren();
      const selected = data.conditions[Number(condition.value)];
      selected.records.forEach((item, i) => example.add(new Option('Coordinate ' + item.coordinate.join('/'), String(i))));
    }
    function update() {
      const selected = data.conditions[Number(condition.value)];
      const item = selected.records[Number(example.value)];
      values.replaceChildren(metric('Actual hidden answer', number(item.actual)), metric(item.positive_probability !== undefined ? 'Predicted positive probability' : 'Captured model prediction', item.positive_probability !== undefined ? percent(item.positive_probability) : number(item.predicted)));
      if (item.positive_probability !== undefined) values.append(metric('Probability scope', 'One observed answer'));
      else if (typeof item.actual === 'number' && typeof item.predicted === 'number') values.append(metric('Absolute error', number(Math.abs(item.actual - item.predicted))));
      else values.append(metric('Exact answer match', item.actual === item.predicted ? 'Correct' : 'Incorrect'));
      status.textContent = `${selected.n.toLocaleString()} coordinates captured for this condition.`;
      if (item.matches !== undefined) status.textContent += ` This record has ${item.matches} matching keys; found probability ${percent(item.found_probability)}.`;
      details.hidden = item.input === null || item.input === undefined;
      record.textContent = details.hidden ? '' : JSON.stringify(item.input, null, 2);
    }
    populate(); update();
  }
  function fusion(box, data) {
    box.append(node('h4', 'Follow one held-out record'));
    box.append(node('p', 'Toggle sensors to select an actual captured visibility condition. The first sixteen example IDs are shown in their original order.', 'explorer-note'));
    const controls = node('div', undefined, 'explorer-controls');
    const example = select(Array.from({length: 16}, (_, i) => [String(i), 'Example ' + i]), update);
    example.setAttribute('aria-label', 'Sensor example ID');
    const target = node('span'); controls.append(example, target);
    const inputs = node('div', undefined, 'sensor-inputs');
    const boxes = ['first', 'second', 'third'].map(name => {
      const card = node('div', undefined, 'sensor');
      const label = node('label');
      const input = document.createElement('input'); input.type = 'checkbox'; input.checked = true; input.onchange = update;
      label.append(input, document.createTextNode(name));
      const reading = node('strong'); card.append(label, reading); inputs.append(card);
      return {card, input, reading};
    });
    const summary = node('p'); summary.setAttribute('aria-live', 'polite');
    const chart = node('div', undefined, 'paired-chart');
    box.append(controls, inputs, summary, chart, node('p', 'Green dot: model prediction. Blue tick and band: the known conditional mean and central 95% interval from the generator. Dashed line: the actual hidden target. The blue interval is not learned model uncertainty. Pattern 101 was withheld from training.', 'explorer-note'));
    function update() {
      const id = Number(example.value), pattern = boxes.map(value => value.input.checked ? '1' : '0').join('');
      const d = data.patterns[pattern], actual = d.target[id];
      boxes.forEach((value, i) => {value.card.classList.toggle('off', !value.input.checked); value.reading.textContent = number(d.readings[id][i]);});
      target.textContent = 'Hidden target: ' + number(actual);
      summary.textContent = `${pattern}: model ${number(d.prediction[id])} · conditional mean ${number(d.oracle_mean[id])}`;
      const order = ['111', '110', '101', '011', '100', '010', '001', '000'];
      const bound = Math.max(3.5, Math.abs(actual) + .5, ...order.map(k => Math.abs(data.patterns[k].prediction[id]) + .5), ...order.map(k => Math.abs(data.patterns[k].oracle_mean[id]) + 1.96 * Math.sqrt(data.patterns[k].oracle_variance) + .3));
      const width = Math.min(800, chart.clientWidth), left = width < 600 ? 116 : 180, span = width - left - 20;
      const x = value => left + (value + bound) / (2 * bound) * span;
      let svg = `<svg viewBox="0 0 ${width} 420" role="img" aria-label="Paired predictions for eight visibility conditions"><line class="target-mark" x1="${x(actual)}" x2="${x(actual)}" y1="22" y2="366" stroke-dasharray="5 5"/>`;
      order.forEach((key, i) => {
        const y = 40 + i * 43, value = data.patterns[key], mean = value.oracle_mean[id], sd = Math.sqrt(value.oracle_variance);
        if (key === pattern) svg += `<rect class="selected-pattern" x="0" y="${y - 18}" width="${width - 5}" height="38" rx="5"/>`;
        svg += `<text x="8" y="${y + 5}">${key}${key === '101' ? ' (withheld)' : ''}</text><rect class="conditional-band" x="${x(mean - 1.96 * sd)}" y="${y - 7}" width="${x(mean + 1.96 * sd) - x(mean - 1.96 * sd)}" height="14" rx="3"/><line class="oracle-mark" x1="${x(mean)}" x2="${x(mean)}" y1="${y - 12}" y2="${y + 12}" stroke-width="2"/><circle class="prediction-mark" cx="${x(value.prediction[id])}" cy="${y}" r="5"/>`;
      });
      for (let tick = Math.ceil(-bound); tick <= bound; tick++) svg += `<text x="${x(tick)}" y="393" text-anchor="middle">${tick}</text>`;
      svg += `<text x="${left + span / 2}" y="416" text-anchor="middle">Signal value</text></svg>`;
      chart.innerHTML = svg;
    }
    update(); window.addEventListener('resize', update);
  }
  function anomaly(box, data) {
    box.append(node('h4', 'Explore the threshold'));
    box.append(node('p', 'Start at the threshold chosen from clean validation data. Moving the slider explores the captured test scores; the proof’s reported operating point stays fixed.', 'explorer-note'));
    const controls = node('div', undefined, 'explorer-controls');
    const label = node('label', 'Residual threshold '), value = node('strong'); label.append(value);
    const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0; slider.max = Math.max(...data.anomalous, 3); slider.step = 'any'; slider.value = data.threshold;
    slider.setAttribute('aria-label', 'Residual warning threshold'); slider.oninput = update;
    const reset = node('button', 'Use validation threshold'); reset.onclick = () => {slider.value = data.threshold; update();};
    controls.append(label, slider, reset);
    const metrics = node('div', undefined, 'explorer-metrics'); metrics.setAttribute('aria-live', 'polite');
    const counts = node('p', undefined, 'explorer-note');
    const chart = node('div', undefined, 'live-roc');
    box.append(controls, metrics, counts, chart);
    const normal = [...data.normal].sort((a,b)=>a-b), anomalous = [...data.anomalous].sort((a,b)=>a-b), tail = [...data.tail].sort((a,b)=>a-b);
    function above(sorted, threshold) {
      let lo = 0, hi = sorted.length;
      while (lo < hi) {const mid = (lo + hi) >> 1; if (sorted[mid] <= threshold) lo = mid + 1; else hi = mid;}
      return sorted.length - lo;
    }
    const thresholds = [...new Set([...normal, ...anomalous])].sort((a,b)=>b-a);
    const path = [[0, 0], ...thresholds.map(t => [above(normal, t) / normal.length, above(anomalous, t) / anomalous.length]), [1,1]].map(([x,y], i) => `${i ? 'L' : 'M'} ${50 + x * 290} ${300 - y * 260}`).join(' ');
    function update() {
      const threshold = Number(slider.value), n = above(normal, threshold), a = above(anomalous, threshold), t = above(tail, threshold);
      value.textContent = threshold.toFixed(3);
      metrics.replaceChildren(metric('Anomalies detected', percent(a/anomalous.length)), metric('Normal false alarms', percent(n/normal.length)), metric('Rare valid false alarms', percent(t/tail.length)));
      counts.textContent = `${a}/${anomalous.length} anomalies detected; ${n}/${normal.length} normal records and ${t}/${tail.length} rare valid records flagged.`;
      chart.innerHTML = `<svg viewBox="0 0 390 355" role="img" aria-label="Measured ROC curve with the explored threshold"><path d="M 50 40 V 300 H 340" fill="none" class="target-mark"/><path d="${path}" fill="none" stroke="var(--plot-green)" stroke-width="2"/><circle cx="${50 + n/normal.length * 290}" cy="${300 - a/anomalous.length * 260}" r="6" fill="var(--plot-plum)"/><text x="50" y="321">0%</text><text x="314" y="321">100%</text><text x="12" y="45">100%</text><text x="24" y="300">0%</text><text x="175" y="347" text-anchor="middle">Normal false-alarm rate</text><text x="12" y="170" transform="rotate(-90 12 170)" text-anchor="middle">Anomaly recall</text></svg>`;
    }
    update();
  }
  fetch(root.dataset.observations).then(response => {
    if (!response.ok) throw new Error('Captured example data could not be loaded');
    return response.json();
  }).then(data => {
    const interactive = data.interactive, box = root.querySelector('.proof-explorer');
    if (!box || !interactive) return;
    if (interactive.kind === 'numeric' || interactive.kind === 'classification') records(box, interactive);
    else if (interactive.kind === 'fusion') fusion(box, interactive);
    else if (interactive.kind === 'anomaly') anomaly(box, interactive);
    box.dataset.ready = 'true';
  }).catch(error => {
    const box = root.querySelector('.proof-explorer');
    if (box) box.textContent = error.message + '. The full run record remains available below.';
    console.error(error);
  });
})();
