(() => {
  // Exposure meter — an amber bar at the bottom that fills with scroll progress.
  const fill = document.querySelector('.exposure-fill');
  const updateExposure = () => {
    if (!fill) return;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const p = max > 0 ? window.scrollY / max : 0;
    fill.style.width = (p * 100).toFixed(2) + '%';
  };

  // Scroll reveals — sections fade up once, then stop watching.
  const reveals = document.querySelectorAll('.reveal');
  if (!('IntersectionObserver' in window)) {
    reveals.forEach(el => el.classList.add('in'));
  } else {
    const io = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    reveals.forEach(el => io.observe(el));
  }

  // Hand-place the prints — a seeded random tilt (±3.2°) and a random nudge off
  // each grid slot (up to ±46px, y and x), like prints dropped on a table.
  // Seeded so the arrangement is the same on every reload.
  const prints = document.querySelectorAll('.print');
  const rand = (() => {
    let s = 0x2f6e2b1;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  })();
  prints.forEach(el => {
    el.style.setProperty('--tilt', ((rand() * 2 - 1) * 3.2).toFixed(2) + 'deg');
    el.style.setProperty('--drift', ((rand() * 2 - 1) * 46).toFixed(0) + 'px');
    el.style.setProperty('--drop', ((rand() * 2 - 1) * 46).toFixed(0) + 'px');
  });

  // Develop — scrolling develops each print into its frame. As a print's top
  // rises from near the bottom of the viewport to mid-screen, `--dev` goes
  // 0 → 1, sweeping the mask open and riding the developer wash across. Scroll
  // back up and the print re-develops — scrubbing both ways. The window ends
  // at mid-screen so every print finishes developing before the page runs out
  // of scroll (the last row can't rise past ~a third of the viewport).
  const devPhase = () => {
    const vh = window.innerHeight;
    for (const el of prints) {
      const t = el.getBoundingClientRect().top;
      let p = (vh * 0.9 - t) / (vh * 0.4);
      p = Math.max(0, Math.min(1, p));
      el.style.setProperty('--dev', p.toFixed(3));
      el.style.setProperty('--wash', (p > 0.05 && p < 0.95 ? 1 : 0).toFixed(0));
    }
  };
  devPhase();
  window.addEventListener('scroll', devPhase, { passive: true });
  window.addEventListener('resize', devPhase);

  // Safelight — a warm pool of light that trails the cursor over the table.
  const safelight = document.querySelector('.safelight');
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  if (safelight && finePointer) {
    let tX = 0, tY = 0, cX = 0, cY = 0, raf = 0, shown = false;
    const loop = () => {
      cX += (tX - cX) * 0.14;
      cY += (tY - cY) * 0.14;
      safelight.style.setProperty('--sx', cX + 'px');
      safelight.style.setProperty('--sy', cY + 'px');
      raf = requestAnimationFrame(loop);
    };
    window.addEventListener('pointermove', (e) => {
      tX = e.clientX;
      tY = e.clientY;
      if (!shown) { shown = true; safelight.classList.add('is-on'); }
    }, { passive: true });
    document.documentElement.addEventListener('mouseleave', () => {
      shown = false;
      safelight.classList.remove('is-on');
    });
    raf = requestAnimationFrame(loop);
  }

  // Sprockets — the film strip drifts like a running reel. Hover pauses it, like
  // catching the strip with a finger. Project pages run it in reverse, so the
  // strip always leads back toward the index.
  const sprockets = document.querySelectorAll('.sprockets');
  sprockets.forEach(s => {
    s.addEventListener('mouseenter', () => s.classList.add('paused'));
    s.addEventListener('mouseleave', () => s.classList.remove('paused'));
    if (document.body.classList.contains('project-page')) s.classList.add('reverse');
  });

  // The room light — flip between darkroom (amber glow) and room-lit (paper & ink).
  const switchBtn = document.querySelector('.light-switch');
  const setLit = lit => {
    document.documentElement.classList.toggle('room-lit', lit);
    if (switchBtn) switchBtn.setAttribute('aria-pressed', String(lit));
    try { localStorage.setItem('cs180-room-lit', lit ? '1' : '0'); } catch {}
  };
  if (switchBtn) {
    setLit(localStorage.getItem('cs180-room-lit') === '1');
    switchBtn.addEventListener('click', () => {
      setLit(!document.documentElement.classList.contains('room-lit'));
    });
  }

  window.addEventListener('scroll', updateExposure, { passive: true });
  window.addEventListener('resize', updateExposure);
  updateExposure();
})();
