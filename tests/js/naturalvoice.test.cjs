// Run with: node --test tests/js/naturalvoice.test.cjs   (tests/test_naturalvoice_js.py runs it as part of pytest)
//
// The chat page's "Naturalise tone" option: finding a natural browser voice and speaking with it without getting stuck.
// The browser's speechSynthesis is replaced by fakes that can be driven event by event, and time by a clock that only moves
// when a test says so (no sleeps). What this cannot show is how a real voice behaves or sounds.
const test = require("node:test");
const assert = require("node:assert/strict");
const NaturalVoice = require("../../src/businessflow/channels/static/naturalvoice.js");

const { isNatural, isOnline, pickNatural, splitForSpeech, createSpeaker, VOICES_WAIT_MS, START_TIMEOUT_MS, MS_PER_CHAR } = NaturalVoice;

const voice = (name, lang, extra = {}) => ({ name, lang, ...extra });
const NEERJA = voice("Microsoft Neerja Online (Natural) - English (India)", "en-IN", { localService: false });
const ARIA = voice("Microsoft Aria Online (Natural) - English (United States)", "en-US", { localService: false });
const SONIA = voice("Microsoft Sonia Online (Natural) - English (United Kingdom)", "en-GB", { localService: false });
const SWARA = voice("Microsoft Swara Online (Natural) - Hindi (India)", "hi-IN", { localService: false });
const HEERA = voice("Microsoft Heera - English (India)", "en-IN", { localService: true });
const KALPANA = voice("Microsoft Kalpana - Hindi (India)", "hi-IN", { localService: true });

// ---- a clock that only moves when told to ----------------------------------------------------------------------------
function makeClock() {
  let now = 0;
  let seq = 0;
  const tasks = new Map();
  return {
    setTimeout(fn, ms) {
      seq += 1;
      tasks.set(seq, { at: now + ms, fn });
      return seq;
    },
    clearTimeout(id) {
      tasks.delete(id);
    },
    advance(ms) {
      const target = now + ms;
      for (;;) {
        const due = [...tasks.entries()].filter(([, t]) => t.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        now = due[1].at;
        tasks.delete(due[0]);
        due[1].fn();
      }
      now = target;
    },
    pending: () => tasks.size,
  };
}

class FakeUtterance {
  constructor(text) {
    this.text = text;
    this.voice = null;
    this.lang = "";
    this.onstart = null;
    this.onend = null;
    this.onerror = null;
  }
}

class FakeSynth {
  constructor(voices = []) {
    this.voices = voices;
    this.spoken = [];
    this.cancels = 0;
    this.listeners = new Set();
    this.speakThrows = false;
  }
  getVoices() {
    return this.voices;
  }
  addEventListener(type, fn) {
    if (type === "voiceschanged") this.listeners.add(fn);
  }
  removeEventListener(type, fn) {
    this.listeners.delete(fn);
  }
  loadLater(voices) {
    this.voices = voices;
    [...this.listeners].forEach((fn) => fn());
  }
  speak(u) {
    if (this.speakThrows) throw new Error("synth broken");
    this.spoken.push(u);
  }
  cancel() {
    this.cancels += 1;
  }
  resume() {}
  get last() {
    return this.spoken[this.spoken.length - 1];
  }
}

function setup(voices = [NEERJA]) {
  const synth = new FakeSynth(voices);
  const clock = makeClock();
  const speaker = createSpeaker({ synth, utteranceCtor: FakeUtterance, timers: clock });
  return { synth, clock, speaker };
}

function recorder() {
  const log = [];
  return { log, handlers: { onEnd: () => log.push("end"), onFail: (f) => log.push(`fail:${f.code}:${f.started}`) } };
}

const sentence = (n) => `This is sentence number ${n} and it has enough words to need its own piece.`;
const LONG = [1, 2, 3, 4].map(sentence).join(" ");

// ---- finding a natural voice -----------------------------------------------------------------------------------------
test("a voice is natural when its name says so", () => {
  for (const name of ["Microsoft Neerja Online (Natural) - English (India)", "Samantha (Enhanced)", "Zoe (Premium)", "Some Neural Voice", "Microsoft Swara Online (Natural) - Hindi (India)"]) {
    assert.equal(isNatural(voice(name, "en-US")), true, name);
  }
  // built-in voices, and Google-named voices that are not reported as online
  for (const name of ["Microsoft Heera - English (India)", "Microsoft David - English (United States)", "Google US English", "Google UK English Female", "Microsoft Kalpana - Hindi (India)"]) {
    assert.equal(isNatural(voice(name, "en-US")), false, name);
  }
  assert.equal(isNatural({ lang: "en-US" }), false);
  assert.equal(isNatural(null), false);
});

// Regression: "Naturalise tone" said "not available" in Google Chrome, whose good voices are online "Google ..." ones.
const GOOGLE_US = voice("Google US English", "en-US", { localService: false });
const GOOGLE_UK = voice("Google UK English Female", "en-GB", { localService: false });
const GOOGLE_HI = voice("Google हिन्दी", "hi-IN", { localService: false });
const DAVID = voice("Microsoft David - English (United States)", "en-US", { localService: true });

test("Chrome's online Google voices count as natural; a local voice with the same kind of name does not", () => {
  assert.equal(isNatural(GOOGLE_US), true);
  assert.equal(isNatural(GOOGLE_HI), true);
  assert.equal(isNatural(voice("Google US English", "en-US", { localService: true })), false);
  assert.equal(isNatural(DAVID), false);
  assert.equal(isOnline(GOOGLE_US), true); // so the page says the reply text goes to the browser's provider
});

test("in Chrome: a Google voice is picked for English and for Hindi instead of 'not available'", () => {
  const chrome = [DAVID, HEERA, GOOGLE_US, GOOGLE_UK, GOOGLE_HI];
  assert.equal(pickNatural(chrome, "en"), GOOGLE_UK); // British before American, as for named-natural voices
  assert.equal(pickNatural(chrome, "hi"), GOOGLE_HI);
});

test("a voice whose name says natural is still preferred over a Google one", () => {
  assert.equal(pickNatural([GOOGLE_UK, ARIA, GOOGLE_US], "en"), ARIA);
  assert.equal(pickNatural([GOOGLE_HI, SWARA], "hi"), SWARA);
});

test("online voices are recognised, so the page can say the reply text is sent to the provider", () => {
  assert.equal(isOnline(NEERJA), true);
  assert.equal(isOnline(voice("Some Online (Natural) voice", "en-US")), true); // by name when localService is not reported
  assert.equal(isOnline(voice("Zoe (Premium)", "en-US", { localService: true })), false);
  assert.equal(isOnline(null), false);
});

test("English: the natural English voice, Indian English first, then British, American", () => {
  assert.equal(pickNatural([HEERA, ARIA, SWARA, NEERJA, SONIA], "en"), NEERJA);
  assert.equal(pickNatural([ARIA, SONIA], "en"), SONIA);
  assert.equal(pickNatural([ARIA], "en"), ARIA);
  assert.equal(pickNatural([voice("Some Natural Voice", "en-AU")], "en").lang, "en-AU");
});

test("Hindi: the natural Hindi voice, and never an English one", () => {
  assert.equal(pickNatural([HEERA, NEERJA, KALPANA, SWARA], "hi"), SWARA);
  assert.equal(pickNatural([NEERJA, ARIA, KALPANA], "hi"), null);
});

test("null when there is no natural voice for the language, or no voices at all", () => {
  assert.equal(pickNatural([HEERA, KALPANA, voice("Google US English", "en-US")], "en"), null);
  assert.equal(pickNatural([SWARA], "en"), null);
  assert.equal(pickNatural([], "en"), null);
  assert.equal(pickNatural(null, "hi"), null);
  assert.equal(pickNatural(undefined, "en"), null);
  assert.equal(pickNatural([voice("Some Natural Voice", "en_IN")], "en").lang, "en_IN"); // underscore tags (some Android voices)
});

// ---- splitting for speech --------------------------------------------------------------------------------------------
test("splitting is lossless, keeps decimals together and ends a Hindi sentence at the danda", () => {
  const text = "The total was 1,234.56 rupees. The rate is 3.5 percent! Is that right? Yes.";
  const pieces = splitForSpeech(text, 40);
  assert.equal(pieces.join(" "), text);
  assert.ok(pieces.some((p) => p.includes("1,234.56")));
  assert.ok(pieces.some((p) => p.includes("3.5")));

  const hindi = "आपकी किस्त बाकी है। कृपया भुगतान करें। धन्यवाद।";
  const hindiPieces = splitForSpeech(hindi, 25);
  assert.equal(hindiPieces.join(" "), hindi);
  assert.ok(hindiPieces.length > 1);
});

test("splitting keeps every piece within the limit and breaks a very long sentence at word boundaries", () => {
  const text = `${"word ".repeat(120).trim()}.`;
  const pieces = splitForSpeech(text, 100);
  assert.ok(pieces.length > 5);
  for (const p of pieces) assert.ok(p.length <= 100);
  assert.equal(pieces.join(" "), text);
});

test("splitting returns nothing for empty text", () => {
  assert.deepEqual(splitForSpeech(""), []);
  assert.deepEqual(splitForSpeech("   "), []);
  assert.deepEqual(splitForSpeech("Hello there."), ["Hello there."]);
});

// ---- waiting for the voice list --------------------------------------------------------------------------------------
test("loadVoices returns the list at once when the browser already has it", async () => {
  const { synth, speaker } = setup([NEERJA]);
  assert.deepEqual(await speaker.loadVoices(), [NEERJA]);
  assert.equal(synth.listeners.size, 0);
});

test("loadVoices waits for the list Chrome and Edge fill in after load, then stops listening", async () => {
  const { synth, speaker } = setup([]);
  const pending = speaker.loadVoices();
  assert.equal(synth.listeners.size, 1);
  synth.loadLater([NEERJA]);
  assert.deepEqual(await pending, [NEERJA]);
  assert.equal(synth.listeners.size, 0);
});

test("loadVoices gives up after the wait with an empty list when the event never comes", async () => {
  const { synth, clock, speaker } = setup([]);
  const pending = speaker.loadVoices();
  clock.advance(VOICES_WAIT_MS - 1);
  synth.loadLater([]); // fired, but the list is still empty: keep waiting
  assert.equal(synth.listeners.size, 1);
  clock.advance(1);
  assert.deepEqual(await pending, []);
  assert.equal(synth.listeners.size, 0);
  assert.equal(clock.pending(), 0);
});

test("loadVoices with no voice events, or no synthesiser at all, answers with what there is", async () => {
  const synth = new FakeSynth([]);
  synth.addEventListener = undefined;
  assert.deepEqual(await createSpeaker({ synth, utteranceCtor: FakeUtterance }).loadVoices(), []);
  const none = createSpeaker({});
  assert.equal(none.supported, false);
  assert.deepEqual(await none.loadVoices(), []);
});

// ---- speaking --------------------------------------------------------------------------------------------------------
test("speaks one piece at a time, in order, with the chosen voice, and ends once", () => {
  const { synth, speaker } = setup();
  const { log, handlers } = recorder();
  assert.equal(speaker.speak(LONG, NEERJA, handlers), true);
  assert.equal(speaker.isSpeaking(), true);
  assert.equal(synth.spoken.length, 1); // the next piece waits for the first to finish
  const pieces = splitForSpeech(LONG);
  assert.ok(pieces.length > 1);
  for (let i = 0; i < pieces.length; i += 1) {
    assert.equal(synth.spoken.length, i + 1);
    assert.equal(synth.last.text, pieces[i]);
    assert.equal(synth.last.voice, NEERJA);
    assert.equal(synth.last.lang, "en-IN");
    synth.last.onstart();
    synth.last.onend();
  }
  assert.deepEqual(log, ["end"]);
  assert.equal(speaker.isSpeaking(), false);
  assert.equal(synth.spoken.map((u) => u.text).join(" "), LONG);
});

test("stop ends it once, even if the browser fires no event, and the stopped piece cannot start the next", () => {
  const { synth, speaker } = setup();
  const { log, handlers } = recorder();
  speaker.speak(LONG, NEERJA, handlers);
  const first = synth.last;
  first.onstart();
  const cancels = synth.cancels;
  speaker.stop();
  assert.equal(synth.cancels, cancels + 1);
  assert.deepEqual(log, ["end"]);
  first.onerror({ error: "interrupted" }); // what Chrome does after cancel()
  first.onend(); // and sometimes this too
  assert.equal(synth.spoken.length, 1);
  assert.deepEqual(log, ["end"]);
  assert.equal(speaker.isSpeaking(), false);
});

test("our own cancel is never reported as a failure", () => {
  const { synth, speaker } = setup();
  const { log, handlers } = recorder();
  speaker.speak("Hello there.", NEERJA, handlers);
  synth.last.onstart();
  synth.last.onerror({ error: "interrupted" });
  assert.deepEqual(log, ["end"]);
  speaker.speak("Hello again.", NEERJA, handlers);
  synth.last.onerror({ error: "canceled" });
  assert.deepEqual(log, ["end", "end"]);
});

test("a new reply replaces the old one: the old ends once, and its late events change nothing", () => {
  const { synth, speaker } = setup();
  const a = recorder();
  const b = recorder();
  speaker.speak(LONG, NEERJA, a.handlers);
  const old = synth.last;
  speaker.speak("A different reply.", NEERJA, b.handlers);
  assert.deepEqual(a.log, ["end"]);
  old.onstart();
  old.onend();
  assert.equal(synth.last.text, "A different reply.");
  assert.equal(synth.spoken.filter((u) => u.text.startsWith("This is sentence")).length, 1);
  synth.last.onstart();
  synth.last.onend();
  assert.deepEqual(b.log, ["end"]);
  assert.deepEqual(a.log, ["end"]);
});

test("a voice that never starts is reported as a failure with nothing heard, so the caller can play the server audio", () => {
  const { synth, clock, speaker } = setup();
  const { log, handlers } = recorder();
  speaker.speak("Hello there.", NEERJA, handlers);
  clock.advance(START_TIMEOUT_MS - 1);
  assert.deepEqual(log, []);
  clock.advance(1);
  assert.deepEqual(log, ["fail:stalled:false"]);
  assert.equal(speaker.isSpeaking(), false);
  assert.ok(synth.cancels >= 2); // cancelled before it started, and again to clear the stalled piece
  const stalled = synth.last;
  stalled.onstart(); // the dead attempt waking up late changes nothing
  stalled.onend();
  assert.deepEqual(log, ["fail:stalled:false"]);
  assert.equal(synth.spoken.length, 1);
});

test("an error before any sound says nothing was heard; an error after some sound says it was", () => {
  const before = setup();
  const b = recorder();
  before.speaker.speak(LONG, NEERJA, b.handlers);
  before.synth.last.onerror({ error: "voice-unavailable" });
  assert.deepEqual(b.log, ["fail:voice-unavailable:false"]);
  assert.equal(before.synth.spoken.length, 1);

  const after = setup();
  const a = recorder();
  after.speaker.speak(LONG, NEERJA, a.handlers);
  after.synth.last.onstart();
  after.synth.last.onend(); // first piece done, second queued
  after.synth.last.onerror({ error: "synthesis-failed" });
  assert.deepEqual(a.log, ["fail:synthesis-failed:true"]);
  assert.equal(after.synth.spoken.length, 2); // it does not read on after a failure
});

test("a voice that starts and then goes quiet is a failure with sound already heard", () => {
  const { synth, clock, speaker } = setup();
  const { log, handlers } = recorder();
  speaker.speak("Hello there.", NEERJA, handlers);
  synth.last.onstart();
  clock.advance(START_TIMEOUT_MS + 1); // the start deadline no longer applies once it has started
  assert.deepEqual(log, []);
  clock.advance(4000 + "Hello there.".length * MS_PER_CHAR);
  assert.deepEqual(log, ["fail:stalled:true"]);
});

test("nothing is left scheduled once speech finishes normally", () => {
  const { synth, clock, speaker } = setup();
  const { log, handlers } = recorder();
  speaker.speak("Hello there.", NEERJA, handlers);
  synth.last.onstart();
  synth.last.onend();
  assert.equal(clock.pending(), 0);
  clock.advance(120000);
  assert.deepEqual(log, ["end"]);
});

test("without a voice, an unsupported browser or empty text it fails at once with nothing heard", () => {
  const { speaker } = setup();
  const noVoice = recorder();
  assert.equal(speaker.speak("Hello.", null, noVoice.handlers), false);
  assert.deepEqual(noVoice.log, ["fail:unavailable:false"]);

  const empty = recorder();
  assert.equal(speaker.speak("   ", NEERJA, empty.handlers), false);
  assert.deepEqual(empty.log, ["fail:unavailable:false"]);

  const unsupported = recorder();
  assert.equal(createSpeaker({}).speak("Hello.", NEERJA, unsupported.handlers), false);
  assert.deepEqual(unsupported.log, ["fail:unavailable:false"]);
});

test("recovers if speak() itself throws: one failure, nothing heard, and it can speak again", () => {
  const { synth, speaker } = setup();
  const { log, handlers } = recorder();
  synth.speakThrows = true;
  speaker.speak("Hello there.", NEERJA, handlers);
  assert.deepEqual(log, ["fail:synthesis-failed:false"]);
  assert.equal(speaker.isSpeaking(), false);
  synth.speakThrows = false;
  assert.equal(speaker.speak("Hello again.", NEERJA, handlers), true);
});
