/* <image-compare> — a drag-to-wipe before/after. Give it two asset links and it
   builds the figure itself:

     <image-compare before="a.jpg" after="b.jpg" caption="…"></image-compare>

   The range input carries the drag, the touch, and the arrow keys; we hand its
   value to CSS as --pos, which drives the clip and the handle together. */
(function () {
  const CHEVRON =
    '<path d="M9.5 7 L4.5 12 L9.5 17"></path>' +
    '<path d="M14.5 7 L19.5 12 L14.5 17"></path>';

  class ImageCompare extends HTMLElement {
    connectedCallback() {
      if (this.dataset.built) return;
      this.dataset.built = "1";

      const before = this.getAttribute("before");
      const after = this.getAttribute("after");
      if (!before || !after) return;

      const figure = document.createElement("figure");
      figure.className = "compare-figure";

      const box = document.createElement("div");
      box.className = "compare";

      const image = (cls, src, alt) => {
        const img = document.createElement("img");
        img.className = "compare-img " + cls;
        img.src = src;
        img.alt = alt || "";
        img.loading = "lazy";
        img.decoding = "async";
        return img;
      };

      box.append(
        image("compare-after", after, this.getAttribute("after-alt")),
        image("compare-before", before, this.getAttribute("before-alt")),
      );

      const line = document.createElement("span");
      line.className = "compare-line";
      line.setAttribute("aria-hidden", "true");
      line.innerHTML =
        '<span class="compare-knob"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">' +
        CHEVRON +
        "</svg></span>";

      const range = document.createElement("input");
      range.className = "compare-range";
      range.type = "range";
      range.min = "0";
      range.max = "100";
      range.step = "0.1";
      range.value = "50";
      range.setAttribute(
        "aria-label",
        this.getAttribute("label") || "Drag to compare the two images",
      );

      const apply = () => box.style.setProperty("--pos", range.value + "%");
      range.addEventListener("input", apply);
      apply();

      box.append(line, range);
      figure.append(box);

      const caption = this.getAttribute("caption");
      if (caption) {
        const figcaption = document.createElement("figcaption");
        figcaption.textContent = caption;
        figure.append(figcaption);
      }

      this.append(figure);
    }
  }

  customElements.define("image-compare", ImageCompare);
})();
