import { useCallback, useEffect, useRef, useState } from 'react';

interface SpeechRecognitionEventLike extends Event {
  results: {
    length: number;
    [index: number]: {
      isFinal?: boolean;
      [alternative: number]: {
        transcript: string;
      };
    };
  };
}

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onstart: (() => void) | null;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechRecognitionErrorEventLike extends Event {
  error?: string;
  message?: string;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}

interface UseWakeWordOptions {
  wakeWord: string;
  onWake: (utterance: string) => void;
}

const SHIKIGAMI_ALIASES = ['しきがみ', '式神', 'シキガミ', 'しき神'];

function toHiragana(input: string): string {
  return input.replace(/[\u30A1-\u30F6]/g, (match) =>
    String.fromCharCode(match.charCodeAt(0) - 0x60)
  );
}

function normalizeSpeechText(input: string): string {
  return toHiragana(
    input
      .normalize('NFKC')
      .toLowerCase()
  ).replace(/[\s、。！？!?,，．・ー〜～\-_]/g, '');
}

function buildWakeWordCandidates(wakeWord: string): string[] {
  const base = wakeWord.trim();
  if (!base) return [];

  const normalizedBase = normalizeSpeechText(base);
  const shikigamiNormalized = SHIKIGAMI_ALIASES.map((alias) => normalizeSpeechText(alias));
  const candidates = [normalizedBase];

  if (shikigamiNormalized.includes(normalizedBase)) {
    candidates.push(...shikigamiNormalized);
  }

  return Array.from(new Set(candidates.filter(Boolean)));
}

function collectTranscript(event: SpeechRecognitionEventLike): string {
  const segments: string[] = [];
  for (let i = 0; i < event.results.length; i += 1) {
    const segment = event.results[i]?.[0]?.transcript?.trim();
    if (segment) segments.push(segment);
  }
  return segments.join(' ').trim();
}

export function useWakeWord({ wakeWord, onWake }: UseWakeWordOptions) {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [lastTranscript, setLastTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const wakeWordRef = useRef(wakeWord);
  const wakeWordCandidatesRef = useRef<string[]>(buildWakeWordCandidates(wakeWord));
  const onWakeRef = useRef(onWake);
  const keepListeningRef = useRef(false);
  const restartTimerRef = useRef<number | null>(null);
  const lastWakeAtRef = useRef(0);
  const lastWakeKeyRef = useRef('');

  const toErrorMessage = useCallback((event: Event | unknown) => {
    const speechEvent = event as SpeechRecognitionErrorEventLike | undefined;
    if (speechEvent?.error && speechEvent.message) return `${speechEvent.error}: ${speechEvent.message}`;
    if (speechEvent?.error) return speechEvent.error;
    if (event instanceof Error && event.message) return event.message;
    return 'error';
  }, []);

  useEffect(() => {
    wakeWordRef.current = wakeWord;
    wakeWordCandidatesRef.current = buildWakeWordCandidates(wakeWord);
  }, [wakeWord]);

  useEffect(() => {
    onWakeRef.current = onWake;
  }, [onWake]);

  const stop = useCallback(() => {
    keepListeningRef.current = false;
    if (restartTimerRef.current !== null) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    recognitionRef.current?.stop();
    setListening(false);
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current || listening) return;
    try {
      keepListeningRef.current = true;
      recognitionRef.current.start();
      setError(null);
    } catch (event) {
      setError(toErrorMessage(event));
      setListening(false);
    }
  }, [listening, toErrorMessage]);

  useEffect(() => {
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Ctor) {
      setSupported(false);
      return;
    }

    const recognition = new Ctor();
    recognition.lang = 'ja-JP';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setListening(true);
      setError(null);
    };

    recognition.onresult = (event) => {
      const transcript = collectTranscript(event);
      if (!transcript) return;

      setLastTranscript(transcript);

      const candidates = wakeWordCandidatesRef.current;
      if (candidates.length === 0) return;

      const normalizedTranscript = normalizeSpeechText(transcript);
      const matched = candidates.some((candidate) => normalizedTranscript.includes(candidate));
      if (!matched) return;

      const now = Date.now();
      const wakeKey = `${normalizedTranscript}::${wakeWordRef.current.trim()}`;
      if (lastWakeKeyRef.current === wakeKey && now - lastWakeAtRef.current < 1500) {
        return;
      }

      lastWakeKeyRef.current = wakeKey;
      lastWakeAtRef.current = now;
      if (matched) {
        onWakeRef.current(transcript);
      }
    };

    recognition.onerror = (event) => {
      const speechEvent = event as SpeechRecognitionErrorEventLike;
      if (speechEvent.error === 'not-allowed' || speechEvent.error === 'service-not-allowed') {
        keepListeningRef.current = false;
      }
      setError(toErrorMessage(event));
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
      if (!keepListeningRef.current) return;
      if (restartTimerRef.current !== null) {
        window.clearTimeout(restartTimerRef.current);
      }
      restartTimerRef.current = window.setTimeout(() => {
        restartTimerRef.current = null;
        try {
          recognition.start();
        } catch {
          // Keep waiting for the next manual start when browser blocks restart.
          setListening(false);
        }
      }, 150);
    };

    recognitionRef.current = recognition;
    setSupported(true);

    return () => {
      keepListeningRef.current = false;
      if (restartTimerRef.current !== null) {
        window.clearTimeout(restartTimerRef.current);
        restartTimerRef.current = null;
      }
      recognition.stop();
      recognitionRef.current = null;
      setListening(false);
    };
  }, [toErrorMessage]);

  return {
    supported,
    listening,
    lastTranscript,
    error,
    start,
    stop,
  };
}
