// Small DOM + display helpers shared by all views.
import { symbolsToBytes, EOW } from "./core.js";

/** Tiny hyperscript: h("div.card", {id:"x"}, "text", childNode, ...) */
export function h(spec, attrs = {}, ...kids) {
  const [tag, ...classes] = spec.split(".");
  const el = document.createElement(tag || "div");
  if (classes.length) el.className = classes.join(" ");
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.classList.add(...String(v).split(" ").filter(Boolean));
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const clear = (el) => { el.replaceChildren(); return el; };
export const fmt = (n, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : "–");
export const pct = (n, d = 1) => (n * 100).toFixed(d) + "%";
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const decoder = new TextDecoder("utf-8", { fatal: true });
/** Human-readable form of a token. Returns {text, kind} with kind in
 *  "text" | "byte" (a partial UTF-8 sequence, shown as hex) | "special". */
export function display(tok, mode) {
  if (mode === "bpe-byte") {
    if (/^<\|.*\|>$/.test(tok) || /^\[[A-Z]+\]$/.test(tok)) return { text: tok, kind: "special" };
    try {
      return { text: visible(decoder.decode(symbolsToBytes(tok))), kind: "text" };
    } catch {
      return { text: [...symbolsToBytes(tok)].map((b) => b.toString(16).padStart(2, "0")).join(" "), kind: "byte" };
    }
  }
  if (/^<[a-z|]+>$/.test(tok) || /^\[[A-Z]+\]$/.test(tok)) return { text: tok, kind: "special" };
  return { text: visible(tok), kind: "text" };
}
/** Make whitespace visible. */
export const visible = (s) => s.replaceAll(" ", "␣").replaceAll("\n", "↵").replaceAll("\t", "⇥");

const hue = (key) => {
  let x = 0;
  for (const c of String(key)) x = (x * 31 + c.codePointAt(0)) >>> 0;
  return (x * 47) % 360;
};

/** A coloured token chip. opts: id, mode, cls, title. */
export function chip(tok, { mode = "", id = null, cls = "", title = "" } = {}) {
  const d = display(tok, mode);
  const el = h("span.chip", { class: cls, title: title || (id != null ? `id ${id}` : tok), style: { "--h": hue(id ?? tok) } });
  if (d.kind === "byte") el.classList.add("partial");
  if (d.kind === "special") el.classList.add("special");
  if (d.text.endsWith(EOW) && mode === "bpe-char") {
    el.append(d.text.slice(0, -EOW.length), h("span.eow", {}, EOW));
  } else el.append(d.text);
  if (id != null) el.append(h("sub", {}, id));
  return el;
}

/** Inline SVG sparkline. */
export function sparkline(values, { w = 220, hgt = 46, label = "" } = {}) {
  if (values.length < 2) return h("div.muted", {}, "…");
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * (w - 4) + 2},${hgt - 4 - ((v - lo) / span) * (hgt - 8)}`);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${hgt}`); svg.setAttribute("class", "spark");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", label);
  svg.innerHTML = `<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${pts.join(" ")}"/>`;
  return svg;
}
