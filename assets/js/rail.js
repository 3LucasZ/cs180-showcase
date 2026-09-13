/* The section rail — a film edge in the left margin listing the writeup's
   sections, with the one you are reading marked in amber.

   Built from the headings rather than authored into the HTML, so it cannot
   drift out of sync with the prose and so any project page gets one from a
   single script tag. Without this file the page is unchanged: the rail is
   navigation, not structure.

   It also gives every heading an id, which is what makes a section linkable. */
(() => {
  const prose = document.querySelector("main .prose");
  if (!prose) return;

  const heads = Array.from(prose.querySelectorAll("h2"));
  if (!heads.length) return;

  const slug = (text) =>
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");

  // Anchor targets, and the thing that makes a deep link possible. Existing ids
  // win; a collision gets a numeric suffix rather than stealing the name.
  const seen = new Set();
  heads.forEach((h, i) => {
    if (!h.id) {
      let id = slug(h.textContent) || "section-" + (i + 1);
      while (seen.has(id)) id += "-" + i;
      h.id = id;
    }
    seen.add(h.id);
  });

  const nav = document.createElement("nav");
  nav.className = "rail";
  nav.setAttribute("aria-label", "Sections");

  // Wrapper mirrors the prose column's box so the list can be positioned off its
  // left edge without hard-coding a pixel offset.
  const inner = document.createElement("div");
  inner.className = "rail-in";

  const list = document.createElement("ol");
  list.className = "rail-list";

  const links = heads.map((h) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.className = "rail-item";
    a.href = "#" + h.id;
    a.textContent = h.textContent.trim();
    li.appendChild(a);
    list.appendChild(li);
    return a;
  });

  inner.appendChild(list);
  nav.appendChild(inner);
  document.body.appendChild(nav);

  // Which section you are reading: the last heading to have crossed the line.
  // Sections above it are behind you, sections below it are still ahead, so the
  // rail doubles as a progress readout through the pipeline.
  const LINE = 0.4;

  const update = () => {
    const mark = window.innerHeight * LINE;
    let now = -1;

    for (let i = 0; i < heads.length; i++) {
      if (heads[i].getBoundingClientRect().top <= mark) now = i;
    }
    // Above the first heading, the first section is still the one you are in.
    if (now < 0) now = 0;

    for (let i = 0; i < links.length; i++) {
      const a = links[i];
      const isNow = i === now;
      a.classList.toggle("is-now", isNow);
      a.classList.toggle("is-past", i < now);
      if (isNow) {
        a.setAttribute("aria-current", "true");
      } else {
        a.removeAttribute("aria-current");
      }
    }
  };

  // rAF-throttled: the page is long and already has scroll handlers.
  let queued = false;
  const onScroll = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      update();
    });
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", update);
  update();
})();
