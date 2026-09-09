/* Maxwell's color matcher — an interactive recreation of the classic
   color-matching experiment. A test light shines on one half of a split
   field; three lamps (imperfect on purpose) light the other. The observer
   mixes the lamps until the halves match. Some test lights can only be
   matched by driving a lamp NEGATIVE — physically, that lamp swings over and
   lights the test side instead of the mixture. That is Maxwell's negative
   primary, and it is why the CIE matching curves dip below zero. */

(function () {
  const svg = document.getElementById("exp-svg");
  if (!svg) return;

  // The three lamps, as additive lights in linear-light sRGB. The "green"
  // lamp leaks red on purpose: real primaries are never pure, and the leak is
  // exactly what forces a negative red for the cyan-greens.
  const LAMPS = [
    { name: "red", linear: [0.85, 0, 0] },
    { name: "green", linear: [0.28, 1, 0] },
    { name: "blue", linear: [0, 0, 0.9] },
  ];

  // Display stand-ins for the spectral test lights: a smooth sweep of
  // saturated linear-RGB anchors from violet to red. (A monitor can't emit
  // a monochromatic light, so each wavelength is shown by its closest vivid
  // screen color — near the display's gamut rim.)
  const SPECTRUM = [
    [400, [0.5, 0, 0.9]],
    [420, [0.38, 0, 0.92]],
    [440, [0.2, 0, 0.95]],
    [460, [0.1, 0.02, 0.9]],
    [480, [0, 0.3, 0.95]],
    [495, [0, 0.7, 0.9]],
    [510, [0, 1, 0.45]],
    [520, [0, 1, 0.15]],
    [540, [0.1, 1, 0]],
    [560, [0.35, 1, 0]],
    [580, [0.7, 1, 0]],
    [600, [1, 0.8, 0]],
    [620, [1, 0.4, 0]],
    [650, [1, 0.1, 0]],
    [700, [0.8, 0, 0]],
  ];

  const target = document.getElementById("exp-target");
  const mix = document.getElementById("exp-mix");
  const status = document.getElementById("exp-status");
  const lamSlider = document.getElementById("exp-lam");
  const lamVal = document.getElementById("exp-lam-val");
  const lampCtl = [0, 1, 2].map((i) => ({
    row: document.querySelector('[data-lamp="' + i + '"]'),
    input: document.getElementById("exp-lamp-" + i),
    out: document.getElementById("exp-lamp-val-" + i),
    swatch: document.querySelector('[data-lamp="' + i + '"] [data-swatch]'),
  }));
  const amounts = [0, 0, 0];
  const state = { lam: parseFloat(lamSlider.value) };

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // linear-light RGB -> sRGB hex, for SVG fills and swatches
  const lin2srgb = (c) =>
    c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  const hex = (lin) =>
    "#" +
    lin
      .map((c) => Math.round(clamp(lin2srgb(c), 0, 1) * 255).toString(16).padStart(2, "0"))
      .join("");

  // color of the test light at a wavelength
  function spectral(lam) {
    lam = clamp(lam, 400, 700);
    for (let i = 0; i < SPECTRUM.length - 1; i++) {
      const [a, ca] = SPECTRUM[i];
      const [b, cb] = SPECTRUM[i + 1];
      if (lam <= b) {
        const t = (lam - a) / (b - a);
        return ca.map((v, j) => v + (cb[j] - v) * t);
      }
    }
    return SPECTRUM[SPECTRUM.length - 1][1].slice();
  }

  // normalize so the brightest channel is full on — every test light is shown
  // at its most saturated, most visible setting
  function toRim(c) {
    const m = Math.max.apply(null, c);
    return m > 0 ? c.map((v) => v / m) : c;
  }

  // the exact amounts that reproduce a target color — three lamps vs three
  // channels, so the match is unique. A negative amount is a lamp that must
  // light the test side instead of the mixture.
  function idealFor(targetLin) {
    const g = targetLin[1];
    const b = targetLin[2] / LAMPS[2].linear[2];
    const r = (targetLin[0] - LAMPS[1].linear[0] * g) / LAMPS[0].linear[0];
    return [r, g, b];
  }

  // the two halves, as seen: positive lamps build the mixture, negative lamps
  // swing over and add to the test light
  function halves(targetLin) {
    const mixLin = [0, 0, 0];
    const testLin = targetLin.slice();
    for (let i = 0; i < 3; i++) {
      const a = amounts[i];
      const lamp = LAMPS[i].linear;
      if (a >= 0) for (let j = 0; j < 3; j++) mixLin[j] += a * lamp[j];
      else for (let j = 0; j < 3; j++) testLin[j] -= a * lamp[j];
    }
    return [
      mixLin.map((v) => clamp(v, 0, 1)),
      testLin.map((v) => clamp(v, 0, 1)),
    ];
  }

  function fmt(v) {
    const s = v.toFixed(2);
    return v >= 0 ? "+" + s : s;
  }

  function render() {
    const targetLin = toRim(spectral(state.lam));
    const ideal = idealFor(targetLin);
    const [mixLin, testLin] = halves(targetLin);

    target.setAttribute("fill", hex(testLin));
    // an unlit half shows the empty well behind it, so the field stays a disc
    mix.setAttribute("fill", mixLin.every((c) => c < 0.015) ? "none" : hex(mixLin));

    // how far the two halves are, in displayed linear light
    let d2 = 0;
    for (let j = 0; j < 3; j++) {
      const d = mixLin[j] - testLin[j];
      d2 += d * d;
    }
    const delta = Math.sqrt(d2 / 3);
    const matched = delta < 0.02;

    const needsNegative = ideal.some((v) => v < -0.002);
    const negNames = LAMPS.filter((_, i) => ideal[i] < -0.002)
      .map((l) => l.name)
      .join(" and ");

    status.classList.toggle("is-match", matched);
    if (matched) {
      status.textContent = "matched — the two halves agree.";
    } else if (needsNegative) {
      status.textContent =
        "this test light can't be matched from positives alone — the exact mix wants a negative " +
        negNames +
        ". drag that lamp below zero and it swings over onto the test light.";
    } else {
      status.textContent =
        "move the lamps until the two halves look the same.";
    }

    // lamp swatches & readouts
    for (let i = 0; i < 3; i++) {
      const neg = amounts[i] < 0;
      lampCtl[i].row.classList.toggle("is-negative", neg);
      lampCtl[i].out.textContent = fmt(amounts[i]);
    }
    lamVal.textContent = Math.round(state.lam) + " nm";
  }

  // lamp swatch colors, once
  for (let i = 0; i < 3; i++) {
    const sw = lampCtl[i].swatch;
    const h = hex(LAMPS[i].linear);
    sw.style.background = h;
    sw.style.setProperty("--glow", h);
  }

  lamSlider.addEventListener("input", () => {
    state.lam = parseFloat(lamSlider.value);
    render();
  });
  for (let i = 0; i < 3; i++) {
    lampCtl[i].input.addEventListener("input", () => {
      amounts[i] = parseFloat(lampCtl[i].input.value) / 100;
      render();
    });
  }

  render();
})();
