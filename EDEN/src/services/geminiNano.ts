export interface ShikigamiIntent {
  mood: 'calm' | 'focused' | 'playful';
  action: 'follow' | 'stay' | 'play' | 'unknown';
  response: string;
}

export interface WorldBuildIntent {
  name: string;
  color: string;
  shape: 'box' | 'sphere' | 'cylinder' | 'cone' | 'torus';
  size: [number, number, number];
  isNpc: boolean;
}

interface NanoSession {
  prompt: (input: string) => Promise<string> | string;
  destroy?: () => void;
}

interface NanoLanguageModel {
  create: () => Promise<NanoSession>;
}

interface NanoAI {
  languageModel?: NanoLanguageModel;
}

declare global {
  interface Window {
    ai?: NanoAI;
    LanguageModel?: NanoLanguageModel;
  }
}

let nanoSessionPromise: Promise<NanoSession | null> | null = null;

async function createSession(): Promise<NanoSession | null> {
  if (nanoSessionPromise) return nanoSessionPromise;

  nanoSessionPromise = (async () => {
    try {
      if (window.ai?.languageModel?.create) {
        return await window.ai.languageModel.create();
      }
      if (window.LanguageModel?.create) {
        return await window.LanguageModel.create();
      }
    } catch (error) {
      console.warn('[GeminiNano] Session init failed:', error);
    }
    return null;
  })();

  return nanoSessionPromise;
}

export async function isGeminiNanoAvailable(): Promise<boolean> {
  const session = await createSession();
  return Boolean(session);
}

function parseJsonSafely<T>(raw: string): T | null {
  const cleaned = raw
    .replace(/```json/gi, '')
    .replace(/```/g, '')
    .trim();

  if (!cleaned) return null;

  try {
    return JSON.parse(cleaned) as T;
  } catch {
    return null;
  }
}

function heuristicIntent(utterance: string): ShikigamiIntent {
  const text = utterance.toLowerCase();
  if (text.includes('待て') || text.includes('stop') || text.includes('stay')) {
    return { mood: 'focused', action: 'stay', response: '了解、ここで待機する。' };
  }
  if (text.includes('遊') || text.includes('play')) {
    return { mood: 'playful', action: 'play', response: 'いいね、いっしょに遊ぼう。' };
  }
  return { mood: 'calm', action: 'follow', response: '任せて。そばで護るよ。' };
}

export async function inferShikigamiIntent(utterance: string): Promise<ShikigamiIntent> {
  const session = await createSession();
  if (!session) {
    return heuristicIntent(utterance);
  }

  const prompt = [
    'あなたはゲーム内AIペットの意図分類器です。',
    'ユーザー発話から pet intent を JSON で返してください。',
    '出力は JSON のみ。',
    'スキーマ:',
    '{"mood":"calm|focused|playful","action":"follow|stay|play|unknown","response":"short string"}',
    `発話: ${utterance}`,
  ].join('\n');

  try {
    const raw = await session.prompt(prompt);
    const parsed = parseJsonSafely<ShikigamiIntent>(String(raw));
    if (parsed && parsed.mood && parsed.action && parsed.response) {
      return parsed;
    }
  } catch (error) {
    console.warn('[GeminiNano] intent fallback:', error);
  }

  return heuristicIntent(utterance);
}

function heuristicBuildIntent(text: string): WorldBuildIntent {
  const lower = text.toLowerCase();
  const shape: WorldBuildIntent['shape'] =
    lower.includes('sphere') || text.includes('球') ? 'sphere' :
      lower.includes('cone') || text.includes('円錐') ? 'cone' :
        lower.includes('cylinder') || text.includes('柱') ? 'cylinder' :
          lower.includes('torus') || text.includes('リング') ? 'torus' :
            'box';

  const color =
    text.includes('赤') || lower.includes('red') ? '#ff4d4d' :
      text.includes('青') || lower.includes('blue') ? '#4da3ff' :
        text.includes('緑') || lower.includes('green') ? '#4dff88' :
          '#ffffff';

  return {
    name: text.slice(0, 20) || 'Debug Object',
    color,
    shape,
    size: [1, 1, 1],
    isNpc: lower.includes('npc') || text.includes('動く') || text.includes('自律'),
  };
}

export async function inferWorldBuildIntent(text: string): Promise<WorldBuildIntent> {
  const session = await createSession();
  if (!session) {
    return heuristicBuildIntent(text);
  }

  const prompt = [
    'あなたは3Dワールド用の構造化コマンド変換器です。',
    '自然言語を object create JSON に変換してください。',
    '出力は JSON のみ。',
    'スキーマ:',
    '{"name":"string","color":"#rrggbb","shape":"box|sphere|cylinder|cone|torus","size":[number,number,number],"isNpc":boolean}',
    `入力: ${text}`,
  ].join('\n');

  try {
    const raw = await session.prompt(prompt);
    const parsed = parseJsonSafely<WorldBuildIntent>(String(raw));
    if (parsed && parsed.name && parsed.color && parsed.shape && Array.isArray(parsed.size)) {
      return {
        ...parsed,
        size: [Number(parsed.size[0]) || 1, Number(parsed.size[1]) || 1, Number(parsed.size[2]) || 1],
      };
    }
  } catch (error) {
    console.warn('[GeminiNano] build intent fallback:', error);
  }

  return heuristicBuildIntent(text);
}
