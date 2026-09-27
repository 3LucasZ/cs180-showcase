/* Renders the \( … \) math in a page with KaTeX. katex.min.js and
   auto-render.min.js are loaded just before this, so the global is there.

   Single $ … $ is deliberately not enabled — auto-render's default claims it,
   and it would eat a dollar sign in ordinary prose. The two display forms are
   both here: \[ … \], and $$ … $$ for the LaTeX habit that types it. Neither
   can be produced by accident the way a lone $ can. */
(function () {
  if (typeof renderMathInElement !== "function") return;

  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
      { left: "$$", right: "$$", display: true },
    ],
    throwOnError: false,
  });
})();
