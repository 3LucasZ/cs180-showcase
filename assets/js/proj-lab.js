/* P0 projection lab — a side-view pinhole camera. Drag the focal length
   (zoom) or the subject distance and watch the rays, the angle the subject
   subtends, and the size of the inverted image change together. Enable the
   "dolly" lock and dragging the zoom dollies the camera back so the subject
   keeps its size — the mechanism behind the vertigo shot. */

(function () {
  const svg = document.getElementById("proj-svg");
  if (!svg) return;

  const NS = "http://www.w3.org/2000/svg";
  const fmt = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1));

  // Geometry (viewBox 960x600). The lens is the center of projection; the
  // subject stands to its right, the sensor hangs at focal length behind it.
  const LENS_X = 300;
  const AXIS_Y = 300; // optical axis
  const GROUND_Y = 560;
  const HEAD_TOP = 198; // subject's head top (above the axis)
  const FEET_Y = 560; // subject's feet (below the axis)
  const SUBJ_H = (HEAD_TOP - AXIS_Y) * -1 + (FEET_Y - AXIS_Y); // 362

  const MIN_FOCAL = 18;
  const MAX_FOCAL = 70;
  const MIN_DIST = 140;
  const MAX_DIST = 615;

  const scene = { focal: 40, dist: 400 };

  // ---- build the static skeleton once ----
  svg.innerHTML =
    "" +
    '<line id="pg-ground" x1="40" y1="' + GROUND_Y + '" x2="940" y2="' + GROUND_Y + '" class="proj-static"/>' +
    '<g id="pg-ticks"></g>' +
    '<g id="pg-sensor"></g>' +
    '<g id="pg-subject"></g>' +
    '<g id="pg-camera"></g>' +
    '<g id="pg-wedge"></g>' +
    '<g id="pg-rays"></g>' +
    '<g id="pg-image"></g>' +
    '<g id="pg-labels"></g>';

  const $ = (id) => svg.getElementById(id);
  const mk = (tag, attrs, text) => {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (text != null) el.textContent = text;
    return el;
  };

  // sensor plane x for a given focal length (behind the lens)
  const sensorX = () => LENS_X - scene.focal;
  const subjectX = () => LENS_X + scene.dist;

  // project a subject point onto the sensor plane through the lens
  function project(px, py) {
    const sx = sensorX();
    const t = (sx - LENS_X) / (px - LENS_X); // negative: behind the lens
    return { x: sx, y: AXIS_Y + t * (py - AXIS_Y) };
  }
  const imgHeight = () => (SUBJ_H * scene.focal) / scene.dist;
  const subtended = () =>
    (Math.atan((AXIS_Y - HEAD_TOP) / scene.dist) + Math.atan((FEET_Y - AXIS_Y) / scene.dist)) *
    180 / Math.PI;

  function drawTicks() {
    const g = $("pg-ticks");
    g.innerHTML = "";
    for (let d = 200; d <= 600; d += 100) {
      const x = LENS_X + d;
      const t = mk("line", { x1: x, y1: GROUND_Y, x2: x, y2: GROUND_Y + 6, class: "proj-static" });
      const n = mk("text", { x, y: GROUND_Y + 20, "text-anchor": "middle", class: "proj-lab" }, d / 10 + "");
      g.append(t, n);
    }
  }

  function drawSubject() {
    const g = $("pg-subject");
    g.innerHTML = "";
    const cx = subjectX();
    const body = mk("path", {
      d:
        "M " + (cx - 34) + " 278 L " + (cx + 34) + " 278 L " + (cx + 24) + " " + GROUND_Y +
        " L " + (cx - 24) + " " + GROUND_Y + " Z",
      class: "proj-subject-body",
    });
    const head = mk("ellipse", { cx, cy: 238, rx: 38, ry: 40, class: "proj-subject-head" });
    const nose = mk("circle", { cx: cx + 30, cy: 240, r: 9, class: "proj-nose" });
    const headTop = mk("circle", { cx, cy: HEAD_TOP, r: 7, class: "proj-point" });
    const feet = mk("circle", { cx, cy: FEET_Y, r: 7, class: "proj-point" });
    g.append(body, head, nose, headTop, feet);
  }

  function drawCamera() {
    const g = $("pg-camera");
    g.innerHTML = "";
    const lens = mk("circle", { cx: LENS_X, cy: AXIS_Y, r: 22, class: "proj-lens" });
    const cop = mk("circle", { cx: LENS_X, cy: AXIS_Y, r: 7, class: "proj-cop" });
    const barrel = mk("rect", { x: LENS_X - 18, y: AXIS_Y - 28, width: 18, height: 56, class: "proj-cam-body" });
    const label = mk("text", { x: LENS_X, y: AXIS_Y - 46, "text-anchor": "middle", class: "proj-cop" }, "center of projection");
    g.append(lens, cop, barrel, label);
  }

  function drawSensor() {
    const g = $("pg-sensor");
    g.innerHTML = "";
    const sx = sensorX();
    const plane = mk("line", { x1: sx, y1: AXIS_Y - 150, x2: sx, y2: AXIS_Y + 150, class: "proj-sensor-plane" });
    const label = mk("text", { x: sx - 16, y: AXIS_Y - 158, "text-anchor": "end", class: "proj-lab" }, "sensor at focal length");
    g.append(plane, label);
    const dim = mk("line", { x1: LENS_X, y1: AXIS_Y - 72, x2: sx, y2: AXIS_Y - 72, class: "proj-dim" });
    g.append(dim);
    const fLab = mk("text", { x: (LENS_X + sx) / 2, y: AXIS_Y - 80, "text-anchor": "middle", class: "proj-f" }, "f = " + fmt(scene.focal));
    g.append(fLab);
  }

  function drawRays() {
    const g = $("pg-rays");
    g.innerHTML = "";
    const cx = subjectX();
    const head = project(cx, HEAD_TOP);
    const feet = project(cx, FEET_Y);
    const mkRay = (px, py, end) => {
      const seg = mk("line", { x1: px, y1: py, x2: LENS_X, y2: AXIS_Y, class: "proj-ray" });
      const ext = mk("line", { x1: LENS_X, y1: AXIS_Y, x2: end.x, y2: end.y, class: "proj-ray-ext" });
      g.append(seg, ext);
    };
    mkRay(cx, HEAD_TOP, head);
    mkRay(cx, FEET_Y, feet);
  }

  function drawImage() {
    const g = $("pg-image");
    g.innerHTML = "";
    const head = project(subjectX(), HEAD_TOP);
    const feet = project(subjectX(), FEET_Y);
    const top = Math.min(head.y, feet.y);
    const bot = Math.max(head.y, feet.y);
    const strip = mk("rect", {
      x: sensorX() - 9, y: top, width: 18, height: bot - top,
      class: "proj-img",
    });
    const label = mk("text", { x: sensorX() - 20, y: (top + bot) / 2, "text-anchor": "end", class: "proj-img-lab" }, "image " + fmt(bot - top) + "px");
    g.append(strip, label);
  }

  function drawWedge() {
    const g = $("pg-wedge");
    g.innerHTML = "";
    const aHead = -Math.atan((AXIS_Y - HEAD_TOP) / scene.dist);
    const aFeet = Math.atan((FEET_Y - AXIS_Y) / scene.dist);
    const R = 90;
    const p0 = { x: LENS_X + R * Math.cos(aHead), y: AXIS_Y + R * Math.sin(aHead) };
    const p1 = { x: LENS_X + R * Math.cos(aFeet), y: AXIS_Y + R * Math.sin(aFeet) };
    const arc = mk("path", {
      d: "M " + p0.x + " " + p0.y + " A " + R + " " + R + " 0 0 1 " + p1.x + " " + p1.y,
      class: "proj-arc",
    });
    const wedge = mk("path", {
      d: "M " + LENS_X + " " + AXIS_Y + " L " + p0.x + " " + p0.y + " L " + p1.x + " " + p1.y + " Z",
      class: "proj-wedge-fill",
    });
    g.append(wedge, arc);
    const ang = subtended();
    const aMid = (aHead + aFeet) / 2;
    const lx = LENS_X + (R + 26) * Math.cos(aMid);
    const ly = AXIS_Y + (R + 26) * Math.sin(aMid);
    const lab = mk("text", { x: lx, y: ly, "text-anchor": "middle", class: "proj-ang" }, fmt(ang) + "°");
    g.append(lab);
  }

  function drawLabels() {
    const g = $("pg-labels");
    g.innerHTML = "";
    const cx = subjectX();
    const dLab = mk("text", { x: cx, y: GROUND_Y + 34, "text-anchor": "middle", class: "proj-lab" }, "subject " + fmt(scene.dist / 10) + "m");
    g.append(dLab);
  }

  function drawAll() {
    drawTicks();
    drawSensor();
    drawSubject();
    drawCamera();
    drawWedge();
    drawRays();
    drawImage();
    drawLabels();
  }

  // ---- readout ----
  function setReadout() {
    const el = document.getElementById("proj-readout");
    if (el) {
      el.textContent =
        "focal " + fmt(scene.focal) + "mm · distance " + fmt(scene.dist / 10) + "m · " +
        "subject subtends " + fmt(subtended()) + "° · image " + fmt(imgHeight()) + "px";
    }
  }

  function update() {
    drawAll();
    setReadout();
  }

  // ---- sliders ----
  const focalSlider = document.getElementById("proj-focal");
  const distSlider = document.getElementById("proj-dist");
  const focalVal = document.getElementById("proj-focal-val");
  const distVal = document.getElementById("proj-dist-val");

  function setDistUI() {
    if (distSlider) {
      distSlider.value = scene.dist;
      distVal.textContent = fmt(scene.dist / 10) + "m";
    }
  }
  function setFocalUI() {
    if (focalSlider) {
      focalSlider.value = scene.focal;
      focalVal.textContent = fmt(scene.focal) + "mm";
    }
  }

  if (focalSlider) {
    focalSlider.min = MIN_FOCAL; focalSlider.max = MAX_FOCAL; focalSlider.value = scene.focal;
    focalSlider.addEventListener("input", () => {
      scene.focal = parseFloat(focalSlider.value);
      setFocalUI(); update();
    });
  }
  if (distSlider) {
    distSlider.min = MIN_DIST; distSlider.max = MAX_DIST; distSlider.value = scene.dist;
    distSlider.addEventListener("input", () => {
      scene.dist = parseFloat(distSlider.value);
      setDistUI(); update();
    });
  }

  update();
})();
