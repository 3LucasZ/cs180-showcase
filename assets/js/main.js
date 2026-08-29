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

  // Develop — each print develops into its frame as it enters view.
  // Cards queue in sequence: each starts a beat after the previous, so on load
  // the prints develop one by one like a tray coming up, and any card that
  // scrolls into view later joins the queue at the next beat.
  const prints = document.querySelectorAll('.print');
  const develop = el => el.classList.add('develop');

  // Hand-place the prints — each gets a small alternating tilt.
  const tilts = [-0.6, 0.5, -0.45, 0.55, -0.5, 0.45];
  prints.forEach((el, i) => el.style.setProperty('--tilt', (tilts[i % tilts.length]) + 'deg'));
  if (!('IntersectionObserver' in window)) {
    prints.forEach(develop);
  } else {
    let seq = 0;
    const devIO = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.style.setProperty('--d', (seq * 0.5) + 's');
          seq++;
          develop(entry.target);
          devIO.unobserve(entry.target);
        }
      }
    }, { threshold: 0.15 });
    prints.forEach(el => devIO.observe(el));
  }

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
