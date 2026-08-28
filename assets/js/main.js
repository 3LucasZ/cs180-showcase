(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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
  if (reduceMotion || !('IntersectionObserver' in window)) {
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
  // Cards develop in order: each needs to be a bit more visible than the last,
  // so the cascade reads as a sequence instead of one batch.
  const prints = document.querySelectorAll('.print');
  const develop = el => el.classList.add('develop');
  if (!('IntersectionObserver' in window)) {
    prints.forEach(develop);
  } else {
    const n = prints.length;
    prints.forEach((el, i) => {
      const t = i === 0 ? 0.45 : i === n - 1 ? 1.0 : 0.45 + (0.55 * i) / (n - 2);
      const devIO = new IntersectionObserver(entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            develop(entry.target);
            devIO.unobserve(entry.target);
          }
        }
      }, { threshold: t });
      devIO.observe(el);
    });
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

  window.addEventListener('scroll', updateExposure, { passive: true });
  window.addEventListener('resize', updateExposure);
  updateExposure();
})();
