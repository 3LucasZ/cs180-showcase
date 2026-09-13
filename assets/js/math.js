/* Renders the \( … \) math in a page with KaTeX. katex.min.js and
   auto-render.min.js are loaded just before this, so the global is there.

   Only \( … \) is enabled — auto-render's default also claims $ … $, which
   would eat a dollar sign in ordinary prose. */
(function () {
  if (typeof renderMathInElement !== "function") return;

  renderMathInElement(document.body, {
    delimiters: [{ left: "\\(", right: "\\)", display: false }],
    throwOnError: false,
  });
})();
