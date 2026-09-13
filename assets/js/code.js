/* Colours the Python quoted in the writeup's code blocks.

   Each block ships as plain text inside its <pre>, so the page reads correctly
   with this file missing — it only wraps tokens in spans. The rebuild happens
   from textContent and goes back in as text nodes, so nothing in a snippet can
   inject markup. */
(function () {
  var KEYWORDS = /^(?:def|return|if|elif|else|for|while|in|not|and|or|import|from|as|None|True|False|lambda|with|pass|break|continue|class|try|except|finally|raise|is|global|yield|assert|del|async|await)$/;

  // Comment first, then strings, numbers, identifiers — so a # or a digit
  // inside a string is consumed by the string it belongs to.
  var TOKEN = /(#[^\n]*)|('''[\s\S]*?'''|"""[\s\S]*?"""|'[^'\n]*'|"[^"\n]*")|(\b\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

  var blocks = document.querySelectorAll("pre.code-src");

  for (var b = 0; b < blocks.length; b++) {
    var pre = blocks[b];
    var src = pre.textContent;
    var frag = document.createDocumentFragment();
    var last = 0;
    var m;

    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(src)) !== null) {
      if (m.index > last) {
        frag.appendChild(document.createTextNode(src.slice(last, m.index)));
      }
      var cls = m[1] ? "tok-com" : m[2] ? "tok-str" : m[3] ? "tok-num" : KEYWORDS.test(m[4]) ? "tok-kw" : "";
      if (cls) {
        var span = document.createElement("span");
        span.className = cls;
        span.textContent = m[0];
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(m[0]));
      }
      last = m.index + m[0].length;
    }

    if (last < src.length) {
      frag.appendChild(document.createTextNode(src.slice(last)));
    }

    pre.textContent = "";
    pre.appendChild(frag);
  }
})();
