/* Director mode engine.
 *
 * Listens ONLY to director.* cue events routed here by app.js.  Data
 * rendering stays untouched — this module does camera moves (smooth
 * scroll), spotlighting, chapter cards, the caption bar, the progress
 * bar, page-tab switching and the results data card.
 *
 * Inert unless the page is opened with ?director=1.
 */
(function () {
  'use strict';
  var ON = new URLSearchParams(location.search).get('director') === '1';
  var REHEARSAL = new URLSearchParams(location.search).get('rehearsal') === '1';
  window.DIRECTOR_MODE = ON;

  var built = false;
  function $(id) { return document.getElementById(id); }
  function esc(v) { var d = document.createElement('div'); d.textContent = String(v == null ? '' : v); return d.innerHTML; }

  function buildUI() {
    if (built) return;
    built = true;
    document.body.classList.add('director-mode');
    if (REHEARSAL) document.body.classList.add('rehearsal-mode');
    var host = document.createElement('div');
    host.id = 'dir-root';
    host.innerHTML =
      '<div id="dir-progress"><div id="dir-progress-fill"></div><span id="dir-progress-label"></span>' +
      '<span id="dir-mode-badge">排练回放 · MOCK EVENT TAPE</span></div>' +
      '<div id="dir-caption"><span id="dir-caption-text"></span></div>' +
      '<div id="dir-beat"><span id="dir-beat-text"></span></div>' +
      '<div id="dir-stinger"><span id="dir-stinger-title"></span>' +
      '<small id="dir-stinger-sub"></small></div>' +
      '<div id="dir-fade"></div>' +
      '<div id="dir-card"><div class="dir-card-inner">' +
      '<span id="dir-card-eyebrow"></span>' +
      '<h2 id="dir-card-title"></h2>' +
      '<p id="dir-card-sub"></p>' +
      '<p id="dir-card-closing"></p>' +
      '<div id="dir-card-tags"></div>' +
      '</div></div>' +
      '<div id="dir-data-card"><div class="dir-data-inner">' +
      '<span id="dir-data-kicker"></span>' +
      '<h2 id="dir-data-title"></h2>' +
      '<div id="dir-results-overview"></div>' +
      '<div id="dir-evidence-tracks"></div>' +
      '<div id="dir-recovery-proof"></div>' +
      '</div></div>';
    document.body.appendChild(host);
  }

  function switchTab(pageName) {
    var tab = $('tab-' + (pageName || 'local'));
    if (tab && !tab.classList.contains('active')) tab.click();
  }

  function topLevelBlockOf(sel) {
    var page = document.querySelector('.dashboard-page.active');
    if (!page || !sel) return null;
    var el = document.querySelector(sel);
    while (el && el.parentElement && el.parentElement !== page) el = el.parentElement;
    return (el && el.parentElement === page) ? el : null;
  }

  function spotlight(camera, extra) {
    var page = document.querySelector('.dashboard-page.active');
    if (!page) return;
    clearSpotlight();
    var selectors = [camera].concat(extra || []).filter(Boolean);
    var keep = [], exact = [];
    selectors.forEach(function (sel) {
      var selected = document.querySelector(sel);
      if (selected) exact.push(selected);
      var b = topLevelBlockOf(sel);
      if (b) keep.push(b);
    });
    Array.prototype.forEach.call(page.children, function (block) {
      block.classList.toggle('dir-dim', keep.length > 0 && keep.indexOf(block) < 0);
    });
    exact.forEach(function (el, index) {
      el.classList.add('dir-panel-focus');
      if (exact.length > 1) el.classList.add(index === 0 ? 'dir-split-primary' : 'dir-split-extra');
    });
  }

  function clearSpotlight() {
    document.querySelectorAll('.dir-dim').forEach(function (el) { el.classList.remove('dir-dim'); });
    document.querySelectorAll('.dir-panel-focus').forEach(function (el) {
      el.classList.remove('dir-panel-focus', 'dir-split-primary', 'dir-split-extra');
    });
  }

  var CAM = { x: 0, y: 0, s: 1 };
  var VIEW_W_FRAC = 0.52, VIEW_H_FRAC = 0.62, MAX_SCALE = 1.18;

  function shellEl() { return document.querySelector('.shell'); }

  function layoutRect(el) {
    // getBoundingClientRect in current camera space -> untransformed coords
    var r = el.getBoundingClientRect();
    return { left: (r.left - CAM.x) / CAM.s, top: (r.top - CAM.y) / CAM.s,
             width: r.width / CAM.s, height: r.height / CAM.s };
  }

  function unionRect(selectors) {
    var lo = { x: Infinity, y: Infinity }, hi = { x: -Infinity, y: -Infinity };
    var found = false;
    selectors.forEach(function (sel) {
      var el = sel && document.querySelector(sel);
      if (!el) return;
      found = true;
      var r = layoutRect(el);
      lo.x = Math.min(lo.x, r.left); lo.y = Math.min(lo.y, r.top);
      hi.x = Math.max(hi.x, r.left + r.width);
      hi.y = Math.max(hi.y, r.top + r.height);
    });
    if (!found) return null;
    var pad = 20;
    return { left: lo.x - pad, top: lo.y - pad,
             width: hi.x - lo.x + 2 * pad, height: hi.y - lo.y + 2 * pad };
  }

  function applyCamera() {
    var sh = shellEl();
    if (sh) sh.style.transform =
      'translate(' + CAM.x + 'px,' + CAM.y + 'px) scale(' + CAM.s + ')';
  }

  function moveCamera(d) {
    spotlight(d.camera, d.spotlight_extra);
    var sh = shellEl();
    if (!sh) return;
    var rect = d.camera
      ? unionRect([d.camera].concat(d.spotlight_extra || []))
      : null;
    if (!rect) {                       // panorama: top of page, no zoom
      CAM = { x: 0, y: 0, s: 1 };
      applyCamera();
      return;
    }
    var vw = window.innerWidth, vh = window.innerHeight;
    var s = Math.min(VIEW_W_FRAC * vw / rect.width,
                     VIEW_H_FRAC * vh / rect.height, MAX_SCALE);
    s = Math.max(s, 1.0);
    var cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
    var tx = vw / 2 - s * cx, ty = vh * 0.47 - s * cy;
    var shW = sh.scrollWidth, shH = sh.scrollHeight;
    tx = Math.min(0, Math.max(tx, vw - s * shW));
    ty = Math.min(0, Math.max(ty, vh - s * shH));
    CAM = { x: tx, y: ty, s: s };
    applyCamera();
  }

  var cardTimer = null;
  function showChapterCard(card) {
    var box = $('dir-card');
    $('dir-card-eyebrow').textContent = card.eyebrow || '';
    $('dir-card-title').textContent = card.title || '';
    $('dir-card-sub').textContent = card.subtitle || '';
    $('dir-card-closing').textContent = card.closing || '';
    $('dir-card-tags').innerHTML = (card.tags || []).map(function (tag) {
      return '<span class="dir-card-tag">' + esc(tag) + '</span>';
    }).join('');
    box.classList.toggle('fullscreen', !!card.fullscreen);
    box.classList.toggle('persistent', !!card.persistent);
    box.classList.add('show');
    if (cardTimer) clearTimeout(cardTimer);
    if (card.persistent) return;
    cardTimer = setTimeout(function () { box.classList.remove('show'); },
                           Math.max((card.hold || 2.5) * 1000 - 400, 400));
  }

  var stingerTimers = [];
  function clearStingerTimers() {
    stingerTimers.forEach(clearTimeout); stingerTimers = [];
  }

  function setLabel(text) {
    var span = $('dir-progress-label');
    if (text === '') { span.textContent = ''; return; }
    span.classList.remove('swap'); void span.offsetWidth;
    span.textContent = text; span.classList.add('swap');
  }

  function showStinger(st, label) {
    var box = $('dir-stinger');
    $('dir-stinger-title').textContent = st.title || '';
    $('dir-stinger-sub').textContent = st.subtitle || '';
    var variant = st.variant || 'chapter';
    box.className = '';                       // reset variant/show/collapse
    if (variant !== 'chapter') box.classList.add('variant-' + variant);
    void box.offsetWidth;
    box.classList.add('show');
    clearStingerTimers();
    var hold = (st.hold || 2.2) * 1000;
    stingerTimers.push(setTimeout(function () {
      box.classList.add('collapse');
      if (variant === 'chapter' && label) setLabel(label);
      stingerTimers.push(setTimeout(function () {
        box.classList.remove('show', 'collapse');
      }, 450));
    }, hold));
  }

  function sourcePill(source) {
    return source
      ? '<small class="dir-source-pill">' + esc(source) + '</small>'
      : '';
  }

  function renderOverview(overview) {
    var token = overview.token || {};
    var cost = overview.cost || {};
    var cache = overview.cache || {};

    $('dir-results-overview').innerHTML =
      '<section class="dir-overview-primary">' +
        '<div>' +
          '<span>' + esc(token.label) + '</span>' +
          '<strong>' + esc(token.value) + '</strong>' +
          sourcePill(token.source) +
        '</div>' +
        '<div>' +
          '<span>' + esc(cost.label) + '</span>' +
          '<strong>' + esc(cost.value) + '</strong>' +
          sourcePill(cost.source) +
        '</div>' +
      '</section>' +
      '<section class="dir-overview-cache">' +
        '<span>' + esc(cache.label) + '</span>' +
        '<strong>' +
          esc(cache.from) +
          '<i>→</i>' +
          esc(cache.to) +
        '</strong>' +
        sourcePill(cache.source) +
      '</section>';
  }

  function renderEvidenceTracks(items) {
    $('dir-evidence-tracks').innerHTML = (items || []).map(function (item, index) {
      return '<article class="dir-evidence-track" ' +
        'style="animation-delay:' + (0.15 * index) + 's">' +
        '<header>' +
          '<b>' + esc(item.scene) + '</b>' +
          sourcePill(item.source) +
        '</header>' +
        '<div>' +
          '<span>Token <strong>' + esc(item.token) + '</strong></span>' +
          '<span>费用 <strong>' + esc(item.cost) + '</strong></span>' +
        '</div>' +
        '<em>' + esc(item.note) + '</em>' +
      '</article>';
    }).join('');
  }

  function renderRecoveryProof(recovery) {
    recovery = recovery || {};

    $('dir-recovery-proof').innerHTML =
      '<span>' + esc(recovery.label) + '</span>' +
      '<strong>' + esc(recovery.runtime) + '</strong>' +
      '<b>对比</b>' +
      '<i>' + esc(recovery.native) + '</i>' +
      sourcePill(recovery.source) +
      '<p>' + esc(recovery.conclusion) + '</p>';
  }

  function showDataCard(card) {
    var box = $('dir-data-card');
    $('dir-data-kicker').textContent = card.kicker || '';
    $('dir-data-title').textContent = card.title || '';

    renderOverview(card.overview || {});
    renderEvidenceTracks(card.evidence || []);
    renderRecoveryProof(card.recovery || {});

    box.classList.add('show');
  }

  function hideDataCard() { var b = $('dir-data-card'); if (b) b.classList.remove('show'); }

  function setCaption(text) {
    var bar = $('dir-caption'), span = $('dir-caption-text');
    if (!text) { bar.classList.remove('show'); return; }
    bar.classList.add('show');
    span.classList.remove('swap'); void span.offsetWidth;
    span.textContent = text; span.classList.add('swap');
  }

  function setProgress(d) {
    var fill = $('dir-progress-fill');
    fill.style.transitionDuration = '0s';
    fill.style.width = (d.start / d.show_total * 100) + '%';
    void fill.offsetWidth;
    fill.style.transitionTimingFunction = 'linear';
    fill.style.transitionDuration = d.duration + 's';
    fill.style.width = ((d.start + d.duration) / d.show_total * 100) + '%';
    if (typeof d.label === 'string') {
      var deferred = d.stinger && (d.stinger.variant || 'chapter') === 'chapter';
      if (!deferred) setLabel(d.label);       // stinger 收缩时再换章节标签
    } else {
      var title = (d.chapter_card && d.chapter_card.title) || d.scene;
      setLabel((d.index + 1) + ' / ' + d.scene_count + ' · ' + title);
    }
  }

  var beatTimer = null;
  function onBeat(d) {
    var box = $('dir-beat');
    var span = $('dir-beat-text');
    if (!box || !span) return;
    span.textContent = d.banner || '';
    box.classList.remove('show'); void box.offsetWidth;
    box.classList.add('show');
    if (beatTimer) clearTimeout(beatTimer);
    var hold = (d.hold || 3) * 1000;
    beatTimer = setTimeout(function () { box.classList.remove('show'); }, hold);
    (d.pulse || []).forEach(function (sel) {
      try {
        var el = document.querySelector(sel);
        if (!el) return;
        el.classList.remove('dir-pulse'); void el.offsetWidth;
        el.classList.add('dir-pulse');
        setTimeout(function () { el.classList.remove('dir-pulse'); }, hold);
      } catch (e) { /* invalid selector — ignore */ }
    });
  }

  var fadeTimer = null;
  function onSceneStart(d) {
    buildUI();
    switchTab(d.page);
    moveCamera(d);
    setCaption(d.caption);
    if (d.data_card) showDataCard(d.data_card); else hideDataCard();
    setProgress(d);
    if (d.stinger) showStinger(d.stinger, d.label);
    if (fadeTimer) { clearTimeout(fadeTimer); fadeTimer = null; }
    if (typeof d.fade_out === 'number') {
      var fade = $('dir-fade');
      fade.style.transitionDuration = d.fade_out + 's';
      fadeTimer = setTimeout(function () { fade.classList.add('show'); },
                             Math.max((d.duration - d.fade_out) * 1000, 0));
    }
  }

  function handle(evt) {
    if (!ON) return true;             // consume silently outside director mode
    var d = evt.data || {};
    if (evt.event_type === 'director.scene_start') onSceneStart(d);
    else if (evt.event_type === 'director.beat') onBeat(d);
    else if (evt.event_type === 'director.show_end') {
      clearSpotlight();
      document.body.classList.add('director-ended');
      document.title = 'DIRECTOR_DONE';
    }
    return true;
  }

  window.Director = { active: ON, handle: handle };
})();
