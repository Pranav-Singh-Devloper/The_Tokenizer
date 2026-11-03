import { $, $$ } from "./util.js";
import { initLearn } from "./learn.js";
import { initPlayground } from "./playground.js";
import { initTrainLab } from "./trainlab.js";
import { initEncodeLab, showEncodeLab } from "./encodelab.js";
import { initExperiments } from "./experiments.js";

const TABS = {
  learn: { init: initLearn },
  playground: { init: initPlayground },
  train: { init: initTrainLab },
  encode: { init: initEncodeLab, onShow: showEncodeLab },
  experiments: { init: initExperiments },
};
const inited = new Set();

async function show() {
  const name = location.hash.slice(1) in TABS ? location.hash.slice(1) : "learn";
  for (const el of $$(".tab")) el.hidden = el.id !== name;
  for (const a of $$("nav a")) a.classList.toggle("on", a.dataset.tab === name);
  const tab = TABS[name], el = $("#" + name);
  if (!inited.has(name)) { inited.add(name); await tab.init(el); }
  else tab.onShow?.(el);
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", show);
show();

// theme toggle (persisted when storage is available)
const root = document.documentElement;
try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch {}
$("#theme").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = root.dataset.theme ?? (dark ? "dark" : "light");
  root.dataset.theme = cur === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});
