/* Runs from a local file, with no server, CDN or network requests. */
(() => {
  'use strict';
  const content = document.getElementById('help-content');
  const sidebar = document.getElementById('help-sidebar');
  const outline = document.getElementById('help-outline');
  const results = document.getElementById('help-results');
  const search = document.getElementById('help-search');
  const status = document.getElementById('search-status');
  const toggle = document.getElementById('contents-toggle');
  const hint = status.textContent.trim();
  const headings = [...content.querySelectorAll('h1,h2,h3,h4,h5,h6')]
    .filter(heading => !heading.closest('.inline-contents'));
  const links = new Map([...outline.querySelectorAll('a[href^="#"]')]
    .map(link => [link.getAttribute('href').slice(1), link]));
  let active = null;

  function navigation(open) {
    document.body.classList.toggle('nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  }
  toggle.addEventListener('click', () => navigation(!document.body.classList.contains('nav-open')));
  sidebar.addEventListener('click', event => {
    if (event.target.closest('a')) navigation(false);
  });

  function current(heading) {
    const link = links.get(heading.id);
    if (!link || link === active) return;
    if (active) active.removeAttribute('aria-current');
    link.setAttribute('aria-current', 'location');
    active = link;
    for (let parent = link.parentElement; parent && parent !== outline; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
    }
  }

  let scheduled = false;
  function track() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      let selected = headings[0];
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top > 135) break;
        selected = heading;
      }
      if (selected) current(selected);
    });
  }
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
    const target = document.getElementById(id);
    if (!target) return;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
    }
    target.scrollIntoView({block: 'start'});
    current(target);
  }
  window.addEventListener('scroll', track, {passive: true});
  window.addEventListener('resize', track);
  window.addEventListener('hashchange', revealHash);
  window.addEventListener('load', () => { revealHash(); track(); });

  const sections = headings.map((heading, index) => {
    const range = document.createRange();
    range.setStartAfter(heading);
    if (headings[index + 1]) range.setEndBefore(headings[index + 1]);
    else range.setEndAfter(content.lastElementChild);
    const fragment = range.cloneContents();
    fragment.querySelectorAll('.inline-contents').forEach(node => node.remove());
    const text = fragment.textContent.replace(/\s+/g, ' ').trim();
    return {heading, text, searchable: `${heading.textContent} ${text}`.toLocaleLowerCase()};
  });
  function searchDocument() {
    const query = search.value.trim().toLocaleLowerCase();
    const words = query.split(/\s+/).filter(Boolean);
    outline.hidden = Boolean(query);
    results.hidden = !query;
    results.replaceChildren();
    if (!query) { status.textContent = hint; return; }
    const matches = sections.filter(section => words.every(word => section.searchable.includes(word)));
    status.textContent = status.dataset.count.replace('{count}', String(matches.length));
    for (const section of matches) {
      const link = document.createElement('a');
      link.href = `#${section.heading.id}`;
      const title = document.createElement('strong');
      title.textContent = section.heading.textContent;
      const excerpt = document.createElement('span');
      const found = section.text.toLocaleLowerCase().indexOf(words[0]);
      const start = Math.max(0, found - 35);
      excerpt.textContent = (start ? '…' : '') + section.text.slice(start, start + 150)
        + (section.text.length > start + 150 ? '…' : '');
      link.append(title, excerpt);
      results.append(link);
    }
  }
  let timer;
  search.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(searchDocument, 120);
  });
  search.addEventListener('keydown', event => {
    if (event.key === 'Escape') { search.value = ''; clearTimeout(timer); searchDocument(); }
    if (event.key === 'Enter') {
      event.preventDefault();
      clearTimeout(timer);
      searchDocument();
      if (results.firstElementChild) results.firstElementChild.click();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)
        && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault(); navigation(true); search.focus();
    }
  });
  document.body.classList.add('script-ready');
  document.querySelector('.search-box').hidden = false;
  revealHash();
  track();
})();
