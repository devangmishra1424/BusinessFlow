/* ============================================================
   Natural-sounding browser voices for the chat's "Naturalise tone" option.

   The speaker icon on a reply normally plays audio made by the server (Piper). With the option on, the page instead asks the
   server for the words it would have spoken (POST .../speech-text) and has a natural-sounding voice from the borrower's own
   browser say them. This file is only the browser-voice part: finding such a voice, and speaking with it in a way that
   cannot get stuck. It touches no page elements and takes the browser's objects as arguments, so node can test it
   (tests/js/naturalvoice.test.cjs) with fakes.

   A "natural" voice is one whose NAME says so (Microsoft Edge's "... Online (Natural)" voices, Windows "Natural" voices,
   macOS "Premium" and "Enhanced" voices), or one of Google Chrome's online "Google ..." voices ("Google US English",
   "Google हिन्दी"). Chrome's are streamed from Google's servers and sound far more natural than the robotic built-in
   ones, but their names don't say so; they used to be left out, so the option never worked in Chrome (found live). Only
   the ONLINE Google voices count (localService false): the older built-in voices, like Windows' "Microsoft David", still
   don't. When a browser has both kinds, a voice whose name says natural is preferred.

   Not a promise about how it sounds -- only about what the browser says it is. Online voices (Edge's Natural ones) send
   the text to the browser's provider to be spoken; isOnline() lets the page say so.
   ============================================================ */
const NaturalVoice = (() => {
  const NATURAL_NAME = /natural|neural|premium|enhanced/i;
  // Within a language, the first of these that exists wins; any other voice of that language comes after.
  const LANG_ORDER = { en: ["en-in", "en-gb", "en-us"], hi: ["hi-in"] };

  const VOICES_WAIT_MS = 2000; // Chrome and Edge fill their voice list a moment after the page loads
  const START_TIMEOUT_MS = 5000; // a voice that has not made a sound by now is not going to
  const MS_PER_CHAR = 140; // slow-voice upper bound for one piece; longer than this and it has stalled
  const MAX_PIECE_CHARS = 160; // Chrome stalls on long utterances, so speech goes a sentence or two at a time

  const langOf = (voice) => String(voice.lang || "").replace("_", "-").toLowerCase();

  const namedNatural = (voice) => NATURAL_NAME.test((voice && voice.name) || "");
  // Chrome's online voices: "Google US English", "Google UK English Female", "Google हिन्दी"...
  const chromeOnline = (voice) => Boolean(voice) && voice.localService === false && /^Google\b/.test(voice.name || "");

  function isNatural(voice) {
    return namedNatural(voice) || chromeOnline(voice);
  }

  /** Online voices stream from the browser's provider; the page tells the borrower their reply text is sent there. */
  function isOnline(voice) {
    return Boolean(voice) && (voice.localService === false || /online/i.test(voice.name || ""));
  }

  /** The best natural voice for a conversation language ("en" or "hi"); null when this browser has none. */
  function pickNatural(voices, language) {
    const ofLanguage = (voices || []).filter((v) => isNatural(v) && langOf(v).startsWith(language));
    // Voices whose name says natural first, Chrome's Google voices after; within each, the preferred accent first.
    for (const group of [ofLanguage.filter(namedNatural), ofLanguage.filter((v) => !namedNatural(v))]) {
      if (group.length === 0) continue;
      for (const wanted of LANG_ORDER[language] || []) {
        const hit = group.find((v) => langOf(v) === wanted);
        if (hit) return hit;
      }
      return group[0];
    }
    return null;
  }

  /** Pieces for the synthesiser: a sentence or two each, never splitting "1,200.50" or "3.5". */
  function splitForSpeech(text, maxLen = MAX_PIECE_CHARS) {
    // Hindi sentences end with the danda; "." only ends one when whitespace follows it.
    const sentences = String(text).split(/(?<=[.!?।])\s+/).map((s) => s.trim()).filter(Boolean);
    const pieces = [];
    let current = "";
    for (const sentence of sentences) {
      if (current && `${current} ${sentence}`.length <= maxLen) {
        current = `${current} ${sentence}`;
        continue;
      }
      if (current) pieces.push(current);
      current = "";
      let rest = sentence;
      while (rest.length > maxLen) {
        let at = rest.lastIndexOf(" ", maxLen);
        if (at < maxLen * 0.4) at = maxLen;
        pieces.push(rest.slice(0, at).trim());
        rest = rest.slice(at).trim();
      }
      current = rest;
    }
    if (current) pieces.push(current);
    return pieces;
  }

  /**
   * One speaker. synth / utteranceCtor are the browser's speechSynthesis and SpeechSynthesisUtterance (missing means
   * unsupported: `supported` is false and the page hides the option).
   *
   *   loadVoices()  -> Promise of the voice list, waiting for the browser to fill it
   *   speak(text, voice, handlers) -> boolean; handlers are
   *       onEnd()           once, when it finished or was stopped
   *       onFail({ code, started })   once, instead of onEnd, when it could not speak. `started` says whether any sound was
   *                         made first: false means nothing was heard, so the caller can play the server audio instead
   *                         without saying anything twice.
   *   stop()        -> stops at once; onEnd is still called
   *
   * Handled: a voice that never starts or goes quiet (a watchdog per piece: Chrome sometimes never reports the end), our own
   * cancel (never reported as a failure), a late event from a stopped or failed attempt (ignored), and the utterance being
   * held in a variable because Chrome can garbage-collect it mid-speech and never fire `end`.
   */
  function createSpeaker({ synth = null, utteranceCtor = null, timers = null } = {}) {
    const clock = timers || { setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: (h) => clearTimeout(h) };
    const supported = Boolean(synth) && typeof utteranceCtor === "function";
    let token = 0;
    let active = null; // { token, end }
    const inFlight = new Set();

    function readVoices() {
      try {
        return (synth && synth.getVoices && synth.getVoices()) || [];
      } catch (err) {
        return []; // voices are optional here: no list just means "no natural voice found"
      }
    }
    readVoices(); // Chrome loads the list lazily; asking early means it is ready by the first click

    function loadVoices(timeoutMs = VOICES_WAIT_MS) {
      const now = readVoices();
      if (now.length > 0 || !synth || typeof synth.addEventListener !== "function") return Promise.resolve(now);
      return new Promise((resolve) => {
        let settled = false;
        let timer = null;
        const finish = () => {
          if (settled) return;
          settled = true;
          if (typeof synth.removeEventListener === "function") synth.removeEventListener("voiceschanged", onChange);
          if (timer) clock.clearTimeout(timer);
          resolve(readVoices());
        };
        const onChange = () => {
          if (readVoices().length > 0) finish();
        };
        synth.addEventListener("voiceschanged", onChange);
        timer = clock.setTimeout(finish, timeoutMs); // some browsers never fire the event; an empty list is then the answer
      });
    }

    function stop() {
      token += 1; // a piece still queued belongs to an older request and must not start
      const current = active;
      active = null;
      inFlight.clear();
      if (synth) {
        try {
          synth.cancel();
        } catch (err) {
          /* a synthesiser that cannot cancel has nothing playing that we could stop */
        }
      }
      if (current) current.end(); // the caller is reset even if the browser fires no event for the cancel
    }

    function speak(text, voice, { onEnd, onFail } = {}) {
      stop();
      const pieces = splitForSpeech(text);
      if (!supported || !voice || pieces.length === 0) {
        if (onFail) onFail({ code: "unavailable", started: false });
        return false;
      }
      token += 1;
      const mine = token;
      let index = 0;
      let done = false;
      let started = false;
      let watchdog = null;

      const clearWatchdog = () => {
        if (watchdog) clock.clearTimeout(watchdog);
        watchdog = null;
      };
      const finish = (failure) => {
        if (done) return;
        done = true;
        clearWatchdog();
        inFlight.clear();
        if (active && active.token === mine) active = null;
        if (failure) {
          if (onFail) onFail({ code: failure, started });
        } else if (onEnd) {
          onEnd();
        }
      };
      const fail = (code) => {
        if (done) return;
        token += 1; // whatever the browser reports from here on belongs to a dead attempt
        try {
          synth.cancel();
        } catch (err) {
          /* nothing playing */
        }
        finish(code);
      };
      const next = () => {
        clearWatchdog();
        if (done || mine !== token) return;
        if (index >= pieces.length) return finish(null);
        const piece = pieces[index];
        index += 1;
        const utterance = new utteranceCtor(piece);
        inFlight.clear();
        inFlight.add(utterance);
        utterance.voice = voice;
        utterance.lang = voice.lang || "";
        utterance.onstart = () => {
          if (mine !== token) return;
          started = true;
          clearWatchdog();
          watchdog = clock.setTimeout(() => fail("stalled"), 4000 + piece.length * MS_PER_CHAR);
        };
        utterance.onend = () => {
          if (mine === token) next();
        };
        utterance.onerror = (event) => {
          if (mine !== token) return; // ours: stop() already reset the caller
          const code = event && event.error;
          if (code === "interrupted" || code === "canceled") return finish(null);
          fail(code || "synthesis-failed");
        };
        watchdog = clock.setTimeout(() => fail("stalled"), START_TIMEOUT_MS);
        try {
          synth.speak(utterance);
        } catch (err) {
          fail("synthesis-failed");
        }
      };

      active = { token: mine, end: () => finish(null) };
      try {
        if (typeof synth.resume === "function") synth.resume(); // Chrome can be left paused by an earlier page
      } catch (err) {
        /* nothing to resume */
      }
      next();
      return true;
    }

    return { supported, loadVoices, speak, stop, isSpeaking: () => active !== null };
  }

  return { isNatural, isOnline, pickNatural, splitForSpeech, createSpeaker, VOICES_WAIT_MS, START_TIMEOUT_MS, MS_PER_CHAR };
})();

// Node (the tests) loads this file as a module; in the browser it is a plain script and `NaturalVoice` is a global.
if (typeof module !== "undefined" && module.exports) module.exports = NaturalVoice;
