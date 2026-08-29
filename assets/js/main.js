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

  // Hand-place the prints — each gets an alternating tilt and a nudge off its
  // grid slot, like prints dropped on a table. Both roll back on hover.
  const prints = document.querySelectorAll('.print');
  const tilts = [-1.9, 1.7, -1.5, 1.9, -1.7, 1.5];
  const drifts = [-10, 8, -7, 9, -6, 10];
  const drops = [-6, 7, -5, 6, -8, 5];
  prints.forEach((el, i) => {
    el.style.setProperty('--tilt', (tilts[i % tilts.length]) + 'deg');
    el.style.setProperty('--drift', (drifts[i % drifts.length]) + 'px');
    el.style.setProperty('--drop', (drops[i % drops.length]) + 'px');
  });

  // Develop — scrolling develops each print into its frame. As a print's top
  // rises through the viewport (entry → fully entered) `--dev` goes 0 → 1,
  // sweeping the mask open and riding the developer wash across. Scroll back up
  // and the print re-develops — scrubbing both ways.
  const devPhase = () => {
    const vh = window.innerHeight;
    for (const el of prints) {
      const t = el.getBoundingClientRect().top;
      let p = (vh * 0.9 - t) / (vh * 0.7);
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
