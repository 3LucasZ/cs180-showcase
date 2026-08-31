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

  // Hand-place the prints — a curated scatter. Each print gets a fixed tilt and
  // a fixed drop below its grid slot (y only), picked so every row breathes
  // slightly and the prints feel dropped by hand, not aligned by machine.
  // Hover straightens a print but leaves it where it landed.
  const prints = document.querySelectorAll('.print');
  const tilts = [-1.9, 1.6, -1.7, 1.8, -1.5, 1.9];
  const drops = [-14, 12, 20, -10, -18, 8];
  prints.forEach((el, i) => {
    el.style.setProperty('--tilt', (tilts[i % tilts.length]) + 'deg');
    el.style.setProperty('--drop', (drops[i % drops.length]) + 'px');
  });

  // Develop — scrolling develops each print into its frame. Progress runs from
  // the print's bottom entering the viewport to its bottom reaching mid-screen,
  // so the image rises up out of the bottom as the card climbs. Every print can
  // reach mid-screen, so every one fully develops before the page ends. Each
  // print latches: it develops once, up to its peak, and stays developed —
  // scrolling back up never re-masks it.
  const devPhase = () => {
    const vh = window.innerHeight;
    const ends = vh * 0.5;
    for (const el of prints) {
      const r = el.getBoundingClientRect();
      let p = (vh - r.bottom) / (vh - ends);
      p = Math.max(0, Math.min(1, p));
      if (p > parseFloat(el.dataset.dev || 0)) {
        el.dataset.dev = String(p);
        el.style.setProperty('--dev', p.toFixed(3));
        el.style.setProperty('--wash', (p > 0.05 && p < 0.95 ? 1 : 0).toFixed(0));
      }
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
  // Pulling the dangling bulb swings it, then the room flips.
  // The darkroom is the default: every load starts dark; the switch is a
  // session flourish and the choice is not persisted.
  const switchBtn = document.querySelector('.light-switch');
  const setLit = lit => {
    document.documentElement.classList.toggle('room-lit', lit);
    if (switchBtn) switchBtn.setAttribute('aria-pressed', String(lit));
  };
  if (switchBtn) {
    setLit(false);

    // Wall switch — flick it and the room flips between darkroom and room-lit.
    const switchIcon = switchBtn.querySelector('.light-switch-icon');
    switchBtn.addEventListener('click', () => {
      if (switchIcon) {
        switchIcon.classList.remove('flick');
        void switchIcon.offsetWidth; // restart the click press
        switchIcon.classList.add('flick');
      }
      setLit(!document.documentElement.classList.contains('room-lit'));
    });
  }

  window.addEventListener('scroll', updateExposure, { passive: true });
  window.addEventListener('resize', updateExposure);
  updateExposure();
})();
