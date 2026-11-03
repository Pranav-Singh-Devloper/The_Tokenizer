// State shared between tabs (e.g. the Training lab feeds the Encoding lab).
export const shared = { trainer: null, sample: null, sampleP: null, demoModels: new Map() };

export const PRESETS = {
  toy: {
    label: "Sennrich et al. toy corpus",
    text: "low ".repeat(5) + "lower ".repeat(2) + "newest ".repeat(6) + "widest ".repeat(3),
  },
  morph: {
    label: "Morphology (play/walk/talk + ing/ed/er/s)",
    text: "playing played player plays play walking walked walker walks walk talking talked talker talks talk "
        + "playing walking talking play walk talk played walked talked",
  },
  lines: {
    label: "A few Shakespeare lines",
    text: "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
        + "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
        + "And by opposing end them. To die: to sleep; No more; and by a sleep to say we end "
        + "The heart-ache and the thousand natural shocks That flesh is heir to.",
  },
};

export function loadSample() {
  shared.sampleP ??= fetch("data/shakespeare_sample.txt").then((r) => r.text()).then((t) => (shared.sample = t));
  return shared.sampleP;
}
