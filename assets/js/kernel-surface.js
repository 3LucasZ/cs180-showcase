/* <kernel-surface> — a convolution kernel as a surface you can turn over.

     <kernel-surface kind="log"><img src="…" /></kernel-surface>

   One implementation, three kernels, picked by `kind`:

     log       the Laplacian of Gaussian (2.1) — a pit in a shallow ring
     gaussian  the smoothing kernel (1.3) — a plain bell
     dog       its derivative along x (1.3) — one lobe up, one lobe down

   The <img> is the whole widget until the first WebGL frame is drawn, so the
   figure is never empty: no JS, no three.js, no WebGL, and it is still the
   original plot. It turns on its own, and stops turning the moment you drag
   it; arrow keys tilt it while it has focus, and the slider moves sigma.

   Pinned to three r147 — the last release that still ships a classic
   (non-module) examples/js/controls/OrbitControls.js. r148 deleted that tree,
   so this pair has no upgrade path short of a bundler. */
(function () {
  // Every kernel here is written with its sigma-dependent prefactor dropped, so
  // the shape is invariant under sigma and only its width moves. That is what
  // lets one fixed vertical frame hold a whole slider range. It also hides the
  // real amplitude falloff — 1/sigma^2 for the Gaussian, 1/sigma for the DoG,
  // 1/sigma^4 for the LoG — which is a thing to say out loud rather than a
  // thing these plots can show.
  const P = Math.exp(-0.5); // the DoG's lobe height, at x = sigma
  const E2 = Math.exp(-2); // the LoG's ridge, at r = 2*sigma

  const KINDS = {
    // The LoG's shape is its own: the Laplacian of a Gaussian is negative at the
    // centre, where the surface bottoms out at -1, and positive in a ridge at
    // r = 2*sigma, where it tops out at 1/e^2. That 7.4:1 ratio is why the plot
    // is a deep narrow pit in a shallow ring rather than a dome.
    //
    // Isotropic, so the turn it takes is not showing the reader anything new —
    // it is there to say, without a word of instruction, that the figure moves
    // and can be taken hold of.
    log: {
      label: "Laplacian of Gaussian",
      half: 12,
      vz: 13,
      // Highest and lowest points of the surface, as multiples of vz. Together
      // with the domain they set the camera aim below.
      top: 0,
      bot: -1,
      lo: -1,
      hi: E2,
      sig: [1.5, 3, 2.5],
      value: (x, r2, s) => {
        const u = r2 / (2 * s * s);
        return (u - 1) * Math.exp(-u);
      },
    },
    // A plain bell, 1 at the centre and decayed to nothing well inside the
    // window across the whole slider: at sigma = 5 the rim is at
    // exp(-15^2/50) = 0.011, so the plate reads flat rather than as a bowl.
    //
    gaussian: {
      label: "Gaussian kernel",
      half: 15,
      vz: 8,
      top: 1,
      bot: 0,
      lo: 0,
      hi: 1,
      sig: [2, 5, 2],
      value: (x, r2, s) => Math.exp(-r2 / (2 * s * s)),
    },
    // dG/dx. The 1/sigma^2 prefactor is dropped and one factor of sigma kept on
    // the x, which is what holds the lobes at a constant height as sigma moves.
    // Antisymmetric in x, symmetric in y — so `x` is the first argument here and
    // r2 the second, the only kernel of the three that needs the direction. A
    // quarter turn of this one is not decoration: it shows the same kernel
    // differentiated along y, which is the pair of images sitting under it.
    dog: {
      label: "Derivative of Gaussian",
      // Tighter than the Gaussian's window, and a tighter sigma range to match.
      // The lobes sit at x = +-sigma and are spent by 3 sigma, so a window sized
      // for the Gaussian's sigma = 5 leaves them a sixth of the radius at
      // sigma = 2 — a bump and a dimple adrift on a plate, which is the one
      // thing this kernel is not. This sizing tracks the LoG's instead.
      half: 12,
      vz: 9,
      top: P,
      bot: -P,
      lo: -P,
      hi: P,
      sig: [2, 4, 3],
      value: (x, r2, s) => (-x / s) * Math.exp(-r2 / (2 * s * s)),
    },
  };

  // Viridis, sampled at nine points, as sRGB hex. The still the LoG widget
  // replaced is a matplotlib surface and this is that plot's colormap, so the
  // live figure and its fallback agree — and unlike a ramp mixed from the page
  // tokens it runs dark-to-light the same way in either room, which is what
  // keeps a pit reading as depth rather than as a hole punched through to the
  // background. All three kernels share it, so they read as one family rather
  // than as three plots that happen to be near each other.
  const RAMP_HEX = [
    "#440154",
    "#482878",
    "#3e4a89",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6dcd59",
    "#fde725",
  ];

  class KernelSurface extends HTMLElement {
    connectedCallback() {
      if (this.dataset.built) return;
      this.dataset.built = "1";

      // OrbitControls.js is a bare IIFE that extends THREE.EventDispatcher at
      // parse time, so if three.min.js never arrived it throws during its own
      // evaluation — before any of this runs — and checking THREE alone would
      // miss it. Both globals, or the <img> stays.
      if (!window.THREE || !THREE.OrbitControls) return;

      const fallback = this.querySelector("img");
      if (!fallback) return;
      if (!("IntersectionObserver" in window)) return;

      // Building a GL context and compiling shaders for a figure most readers
      // may never scroll to is wasted work, so nothing is created until it is
      // nearly in view and until then the <img> is the whole widget.
      const io = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            io.disconnect();
            this._build();
          }
        },
        { rootMargin: "400px 0px" },
      );
      io.observe(this);
    }

    _build() {
      const k = KINDS[this.getAttribute("kind")] || KINDS.log;
      const HALF = k.half;
      // Four cells per unit of radius in every kernel, so a facet is the same
      // size whichever figure it is in and the three match at any zoom. Odd, so
      // a vertex lands exactly on r = 0 — at even N none does, and the peak is
      // never sampled.
      const N = HALF * 4 + 1;
      const TURN = Math.PI / 30; // one arrow-key nudge

      let renderer;
      try {
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      } catch (err) {
        return; // no WebGL — leave the plot up and say nothing
      }
      if (!renderer.getContext()) return;

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputEncoding = THREE.sRGBEncoding; // r147 has no outputColorSpace
      // Transparent, so the page background shows through and the canvas can
      // never be caught out of step with the room during a theme flip.
      renderer.setClearColor(0x000000, 0);

      const scene = new THREE.Scene();

      // Non-indexed, so every triangle owns its corners and can carry one flat
      // colour. That is the whole trick behind the faceted look — on shared
      // vertices the colour would interpolate across each quad and the mesh
      // would come out smooth.
      const geo = new THREE.PlaneGeometry(HALF * 2, HALF * 2, N - 1, N - 1).toNonIndexed();
      geo.rotateX(-Math.PI / 2); // lay the plane down as ground

      const pos = geo.attributes.position;
      const count = pos.count;
      pos.setUsage(THREE.DynamicDrawUsage);

      // x is kept signed as well as r2 because the DoG needs the direction; the
      // two isotropic kernels ignore it.
      const vx = new Float32Array(count);
      const vr2 = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        const x = pos.getX(i);
        const z = pos.getZ(i);
        vx[i] = x;
        vr2[i] = x * x + z * z;
      }

      // One value per quad, taken at the quad's centre so both of its triangles
      // land on the same colour and the mesh reads as quads rather than as
      // triangles. The six corners are averaged rather than indexed because that
      // stays right whatever winding order three puts inside a cell; the two
      // triangles of a quad always share the same six positions.
      const quads = count / 6;
      const qx = new Float32Array(quads);
      const qr2 = new Float32Array(quads);
      for (let q = 0; q < quads; q++) {
        let mx = 0;
        let mz = 0;
        for (let c = 0; c < 6; c++) {
          mx += pos.getX(q * 6 + c);
          mz += pos.getZ(q * 6 + c);
        }
        mx /= 6;
        mz /= 6;
        qx[q] = mx;
        qr2[q] = mx * mx + mz * mz;
      }

      const col = new Float32Array(count * 3);
      geo.setAttribute(
        "color",
        new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage),
      );

      // r147 still runs ColorManagement.legacyMode, so these sRGB hexes land in
      // the Color unconverted while outputEncoding applies the sRGB transfer
      // again on the way out. Without the conversion the surface renders about a
      // stop too bright and desaturated.
      const ramp = RAMP_HEX.map((hex) =>
        new THREE.Color().setStyle(hex).convertSRGBToLinear(),
      );

      const mix = new THREE.Color();
      const rampAt = (v) => {
        const t = v * (ramp.length - 1);
        const i = Math.min(ramp.length - 2, Math.max(0, Math.floor(t)));
        return mix.copy(ramp[i]).lerp(ramp[i + 1], t - i);
      };

      // Height maps straight onto the ramp, each kernel autoscaled over its own
      // range rather than rescaled about zero. That is the whole difference
      // between this and a diverging map. For the LoG it is why the skirt sits
      // at 0.88 of the ramp, not at its midpoint, and the plot reads as a broad
      // pale plateau with a narrow dark spike rather than as a ring glowing out
      // of a black plate. For the DoG the range is symmetric, so its plate lands
      // mid-ramp — which is honest, because the plate there really is the zero
      // the two lobes are measured from.
      //
      // Recoloured on every sigma change: the colour is a function of height,
      // and the height moves. Colouring by position instead leaves the ramp
      // sitting where sigma was when it was built, so the features and the ramp
      // drift apart as the slider moves.
      const shade = (s) => {
        const span = k.hi - k.lo;
        for (let q = 0; q < quads; q++) {
          const v = (k.value(qx[q], qr2[q], s) - k.lo) / span;
          const c = rampAt(v);
          for (let j = 0; j < 6; j++) {
            const i = (q * 6 + j) * 3;
            col[i] = c.r;
            col[i + 1] = c.g;
            col[i + 2] = c.b;
          }
        }
        geo.attributes.color.needsUpdate = true;
      };

      // Crops a material to the disc r <= HALF, which is the circle inscribed in
      // the square the grid is built on — so the domain is round and the last
      // row of cells lands exactly on the rim. A fragment discard rather than a
      // mesh actually built to the circle, because the square grid is the look
      // here and a square grid trimmed cell by cell is a staircase; discarding
      // keeps every facet the same size and puts the edge on the exact radius.
      // Both the surface and its edge grid are cropped, or the grid would carry
      // on across the cut as a floating net.
      const cropToDisc = (material) => {
        const rad2 = (HALF * HALF).toFixed(1);
        material.onBeforeCompile = (shader) => {
          shader.vertexShader = shader.vertexShader
            .replace("void main() {", "varying vec2 vDisc;\nvoid main() {")
            .replace(
              "#include <begin_vertex>",
              "#include <begin_vertex>\n\tvDisc = position.xz;",
            );
          shader.fragmentShader = shader.fragmentShader
            .replace("void main() {", "varying vec2 vDisc;\nvoid main() {")
            .replace(
              "#include <clipping_planes_fragment>",
              "#include <clipping_planes_fragment>\n\tif ( dot( vDisc, vDisc ) > " +
                rad2 +
                " ) discard;",
            );
        };
        // What onBeforeCompile injects is invisible to the program cache, so two
        // materials that differ only in their injected source would share one.
        material.customProgramCacheKey = () => "disc" + rad2;
      };

      const surfaceMat = new THREE.MeshLambertMaterial({
        vertexColors: true,
        // One normal per facet, so each polygon shades as the flat plane it
        // is. Without it the lighting smooths the facets back out and the
        // coarse grid buys nothing but a blunter silhouette.
        flatShading: true,
        // Mandatory now that the polar angle is unbounded: seen from below this
        // surface is nothing but backfaces, and the default FrontSide culls
        // every one of them. Under DOUBLE_SIDED the flat-shaded normal is flipped
        // by gl_FrontFacing, so the underside lights as a surface rather than
        // going black.
        side: THREE.DoubleSide,
        // Nudges the surface back so the mesh lines drawn over it land on top
        // instead of tearing through it wherever they cross a facet.
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      cropToDisc(surfaceMat);

      const mesh = new THREE.Mesh(geo, surfaceMat);
      // flatShading reads its normals from screen derivatives and never touches
      // the attribute, so nothing here refreshes the bounds — and a mesh that
      // starts small could be culled mid-orbit as sigma grows.
      mesh.frustumCulled = false;
      scene.add(mesh);

      // The polygon edges, as a line grid over the same heightfield. Flat
      // shading alone gives the facets, but they only separate where the light
      // happens to catch them, and the mesh is the point here — so the grid is
      // drawn in and follows the surface.
      const seg = N - 1;
      const step = (HALF * 2) / seg;
      // Two vertices per segment, so a row of `seg` cells is `seg` segments and
      // each grid line is walked as its own short polyline. x and z are fixed;
      // only y is rewritten when sigma moves.
      const linePos = [];
      const lineX = [];
      const lineR2 = [];
      const push = (i, j) => {
        const x = -HALF + i * step;
        const z = -HALF + j * step;
        linePos.push(x, 0, z);
        lineX.push(x);
        lineR2.push(x * x + z * z);
      };
      for (let j = 0; j <= seg; j++) {
        for (let i = 0; i < seg; i++) {
          push(i, j);
          push(i + 1, j);
        }
      }
      for (let i = 0; i <= seg; i++) {
        for (let j = 0; j < seg; j++) {
          push(i, j);
          push(i, j + 1);
        }
      }
      const lx = new Float32Array(lineX);
      const lr2 = new Float32Array(lineR2);
      const lpos = new Float32Array(linePos);
      // White, matching the plots these replace: it reads against a dark spike
      // and a green plateau, fades out on a yellow rim, and is the only value
      // that works over both ends of the ramp and in either room.
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.24,
      });
      cropToDisc(lineMat);
      const lines = new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute(
          "position",
          new THREE.Float32BufferAttribute(lpos, 3).setUsage(
            THREE.DynamicDrawUsage,
          ),
        ),
        lineMat,
      );
      lines.frustumCulled = false;
      scene.add(lines);

      // Lit for the form rather than for the colour: a high key on a matte
      // ground, which is what makes each facet read as a plane and the skirt
      // read as a surface at all. Kept under 1.1 total, because the top of the
      // ramp is a near-white yellow and anything brighter clips it to red.
      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const key = new THREE.DirectionalLight(0xffffff, 0.6);
      key.position.set(6, 9, 7);
      const fill = new THREE.DirectionalLight(0xffffff, 0.22);
      fill.position.set(-7, 4, -6);
      scene.add(key, fill);

      // A long lens rather than a wide one. At 38 degrees the disc is close
      // enough to the camera that its near rim, magnified for being nearest,
      // projects lower on screen than the pit floor does — so the spike hides
      // behind the plate and the figure reads as a bowl. Pulling back and
      // narrowing flattens the projection until the pit clears the rim, which is
      // also how the plots these replace are drawn. Distance is a multiple of
      // HALF so the disc fills the same fraction of the frame at any domain
      // size.
      const camera = new THREE.PerspectiveCamera(20, 1, 0.1, 400);
      // Three-quarter view, deliberately not OrbitControls' default: a polar
      // angle of 90 sits level with the horizon, where a heightfield is a sliver.
      // Low on purpose — at a steeper angle you look down into the bowl and the
      // depth cancels itself out, and the relief is the whole figure.
      const phi = THREE.MathUtils.degToRad(58);
      const theta = THREE.MathUtils.degToRad(-38);
      const dist = HALF * 5.2;
      // Aimed at the midpoint of what actually decides the vertical extent, in
      // screen terms. A rim sits down on the plate but a tilted disc throws it
      // HALF*cot up and down the screen from the centre, so the top of the
      // figure is whichever reaches further of that and the peak; likewise the
      // bottom against the trough. Reading only the surface and not the rim
      // (or only the rim and not the surface) pushes a tall kernel off the top
      // of the frame — a flat plate's own near edge still hangs below it.
      const rim = HALF * Math.tan(Math.PI / 2 - phi);
      const target = new THREE.Vector3(
        0,
        (Math.max(k.vz * k.top, rim) + Math.min(k.vz * k.bot, -rim)) / 2,
        0,
      );
      // Measured from the target, not the origin — OrbitControls orbits the
      // target, so placing the camera from the origin would tilt the view.
      camera.position.copy(target).add(
        new THREE.Vector3(
          dist * Math.sin(phi) * Math.sin(theta),
          dist * Math.cos(phi),
          dist * Math.sin(phi) * Math.cos(theta),
        ),
      );

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.085;
      controls.enablePan = false;
      // Turning only. Zooming would let the reader put the camera where the
      // framing no longer works — and there is nothing to zoom in on.
      controls.enableZoom = false;
      // The whole sphere: stand the figure on edge, or drop under the plate and
      // read the underside. The arrow keys clamp to these, so this is also what
      // opens the tilt up.
      controls.minPolarAngle = 0;
      controls.maxPolarAngle = Math.PI;
      // A turntable, not a carousel: three of these turn at once on this page,
      // and the turn is there to say the figure can be taken hold of, not to
      // hold attention.
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.2;
      controls.target.copy(target);
      controls.update();
      // The constructor sets touch-action:none. On a phone that turns a canvas
      // this size into a hole the reader cannot scroll past — worse than the
      // still it replaces. Vertical swipe goes back to the page; horizontal drag
      // and pinch still orbit.
      renderer.domElement.style.touchAction = "pan-y";

      const canvas = renderer.domElement;
      canvas.tabIndex = 0;
      canvas.setAttribute("role", "img");
      canvas.setAttribute(
        "aria-label",
        k.label +
          " surface, turning on its own. Drag to turn it, or tilt it with the up and down arrow keys.",
      );

      let raf = 0;
      let down = false;
      // autoRotate makes update() report movement on every frame, so this loop
      // no longer settles on its own the way it did when the only motion was the
      // reader's own damping tail. Two things have to be able to stop it: the
      // reader taking hold of the figure, who has asked for one angle and would
      // not thank us for turning it away, and the figure leaving the screen.
      let held = false;
      let onScreen = false;
      const render = () => renderer.render(scene, camera);
      const loop = () => {
        const moving = controls.update();
        render();
        raf = moving || down ? requestAnimationFrame(loop) : 0;
      };
      const wake = () => {
        if (!raf) raf = requestAnimationFrame(loop);
      };
      // 'change' rather than 'start' — it also covers the wheel, which has no
      // pointerdown, and it is dispatched from inside update() when the camera
      // actually moved, which is the only honest "still settling" signal.
      controls.addEventListener("change", wake);
      const grab = () => {
        held = true;
        controls.autoRotate = false;
      };
      // A press is not a grab until it moves. On a phone every scroll past the
      // figure starts with a pointerdown on it, and reading that as "the reader
      // has taken hold" would stop the turn for the rest of the visit.
      let downX = 0;
      let downY = 0;
      canvas.addEventListener("pointerdown", (e) => {
        down = true;
        downX = e.clientX;
        downY = e.clientY;
        wake();
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!down) return;
        if (Math.abs(e.clientX - downX) > 4 || Math.abs(e.clientY - downY) > 4) {
          grab();
        }
      });
      // Cancel as well as up: touch-action:pan-y hands a vertical swipe to the
      // page, which cancels the pointer — and a cancelled pointer never fires
      // pointerup, so watching only for that leaves the loop running for the
      // rest of the visit.
      const release = () => {
        down = false;
      };
      window.addEventListener("pointerup", release);
      window.addEventListener("pointercancel", release);

      // Off screen or in a background tab there is nobody to draw for, and a
      // throttled rAF still keeps the reader's GPU warm. The build above already
      // happened, so this second observer is about the running loop, not about
      // whether to create the context — hence no rootMargin: the figure stops
      // the moment it is out of sight.
      const resume = () => {
        controls.autoRotate = !held && onScreen && !document.hidden;
        if (controls.autoRotate) wake();
      };
      const vis = new IntersectionObserver((entries) => {
        onScreen = entries[0].isIntersecting;
        resume();
      });
      vis.observe(this);
      document.addEventListener("visibilitychange", resume);

      // rotateUp is closure-private in r147 and there are no setters for the
      // angles, so the camera is moved directly here. Going around damping is
      // the right feel for a keypress anyway.
      const tilt = (dPhi) => {
        const offset = camera.position.clone().sub(controls.target);
        const sph = new THREE.Spherical().setFromVector3(offset);
        sph.phi = THREE.MathUtils.clamp(
          sph.phi + dPhi,
          controls.minPolarAngle,
          controls.maxPolarAngle,
        );
        sph.makeSafe();
        camera.position
          .copy(controls.target)
          .add(new THREE.Vector3().setFromSpherical(sph));
        camera.lookAt(controls.target);
        render();
      };
      canvas.addEventListener("keydown", (e) => {
        if (e.key === "ArrowUp") tilt(-TURN);
        else if (e.key === "ArrowDown") tilt(TURN);
        else return;
        grab();
        e.preventDefault();
      });

      const setSigma = (s) => {
        for (let i = 0; i < count; i++) {
          pos.setY(i, k.vz * k.value(vx[i], vr2[i], s));
        }
        pos.needsUpdate = true;
        for (let i = 0; i < lx.length; i++) {
          lpos[i * 3 + 1] = k.vz * k.value(lx[i], lr2[i], s);
        }
        lines.geometry.attributes.position.needsUpdate = true;
        shade(s); // the colour is a function of height, so it moves with it
        // No computeVertexNormals: flat shading takes its normals from screen
        // derivatives in the fragment shader and never reads the attribute, so
        // the surface re-lights itself as it deforms.
        render();
      };

      const stage = document.createElement("div");
      stage.className = "kernel-surface-stage";
      stage.append(canvas);

      const slider = document.createElement("input");
      slider.type = "range";
      slider.className = "kernel-surface-range";
      slider.min = String(k.sig[0]);
      slider.max = String(k.sig[1]);
      slider.step = "0.05";
      slider.value = String(k.sig[2]);
      slider.setAttribute("aria-label", "Gaussian sigma");

      const readout = document.createElement("output");
      readout.className = "kernel-surface-readout";
      readout.textContent = "σ = " + k.sig[2].toFixed(2);

      // A drag fires input faster than the display refreshes, and each one is a
      // full position rewrite plus a colour rebuild — so coalesce to one per
      // frame, reading the latest value rather than queueing every step.
      let queued = false;
      slider.addEventListener("input", () => {
        readout.textContent = "σ = " + Number(slider.value).toFixed(2);
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => {
          queued = false;
          setSigma(Number(slider.value));
        });
      });

      const bar = document.createElement("div");
      bar.className = "kernel-surface-controls";
      bar.append(slider, readout);

      this.append(stage, bar);

      const resize = () => {
        const w = stage.clientWidth;
        const h = stage.clientHeight;
        if (!w || !h) return;
        // updateStyle stays false so three leaves the CSS size alone — an inline
        // height would beat the stylesheet and pin the canvas at one pixel size.
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        render();
      };

      // A ResizeObserver rather than window.resize: it also catches the column
      // changing width for reasons that are not a window resize.
      if ("ResizeObserver" in window) new ResizeObserver(resize).observe(stage);

      canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        this.classList.remove("is-live"); // fall back to the plot, permanently
      });

      // The class goes on first so the stage is in layout and has a measurable
      // width; both happen in one task, so nothing paints in between.
      this.classList.add("is-live");
      setSigma(k.sig[2]);
      resize();
    }
  }

  customElements.define("kernel-surface", KernelSurface);
})();
