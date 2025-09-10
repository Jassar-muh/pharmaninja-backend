// server.js — ESM (Render-friendly) with graceful Pinecone fallback + tiny chat memory
// Keep this FIRST so .env is loaded before anything uses it
import 'dotenv/config';

import express from 'express';
import cors from 'cors';
import { Agent as HttpsAgent } from 'node:https';

// -------------------------
// Config & constants
// -------------------------
const OPENAI_API_KEY  = process.env.OPENAI_API_KEY || '';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';
const PINECONE_INDEX   = process.env.PINECONE_INDEX || 'pharmaninja-bot-1536';
// Strongly recommended: set PINECONE_HOST explicitly in Render env
const PINECONE_HOST    = process.env.PINECONE_HOST || '';

const PORT = process.env.PORT || 3000;

// Embedding model must match your Pinecone index dimension (1536)
const EMBEDDING_MODEL = 'text-embedding-3-small';

// Toggle: if Pinecone is flaky, you can temporarily disable vector lookup
const USE_EMBEDDINGS = process.env.USE_EMBEDDINGS !== 'false';

// Small in-memory session store (per-process; swap for Redis if you need persistence)
const SESSIONS = new Map(); // sessionId -> { lang, stage, subject, pinnedTopic, history:[{role,content}], ts }

// IPv4-only agent (for Node core HTTPS). undici/fetch may not use it; kept for completeness.
const agentV4 = new HttpsAgent({ keepAlive: true, maxSockets: 50, timeout: 30000, family: 4 });

// -------------------------
// App
// -------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// -------------------------
// Helpers
// -------------------------
function getOrInitSession(sessionId) {
  let s = SESSIONS.get(sessionId);
  if (!s) {
    s = { lang: 'EN', stage: null, subject: null, pinnedTopic: null, history: [], ts: Date.now() };
    SESSIONS.set(sessionId, s);
  }
  return s;
}

function pushHistory(session, role, content) {
  session.history.push({ role, content: String(content || '').slice(0, 2000) });
  if (session.history.length > 10) session.history = session.history.slice(-10);
  session.ts = Date.now();
}

function detectMCQ(q, lang) {
  const t = (q || '').toLowerCase();
  if (/mcq/.test(t) || /multiple[- ]?choice/.test(t)) return true;
  if (lang === 'AR' && /(اختيار|متعدد|أسئلة|mcq)/i.test(q)) return true;
  return false;
}

function cleanSources(matches) {
  return (matches || [])
    .slice(0, 5)
    .map(m => {
      const id = m?.id || '';
      const file = m?.metadata?.file || m?.metadata?.source || id;
      return { id, file, score: m?.score ?? undefined };
    });
}

// Simple topic extraction: first header-ish line of assistant’s answer
function extractTopic(text) {
  if (!text) return null;
  const firstLine = text.split('\n').find(l => l.trim().length > 0) || '';
  const t = firstLine.replace(/^#+\s*/, '').replace(/\*\*|__/g, '').trim();
  if (t.length >= 6 && t.length <= 120) return t;
  return text.split(/\s+/).slice(0, 10).join(' ');
}

// -------------------------
// OpenAI calls
// -------------------------
async function embedTexts(texts) {
  const url = 'https://api.openai.com/v1/embeddings';
  const body = { model: EMBEDDING_MODEL, input: texts };
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings ${r.status}: ${txt}`);
  }
  const json = await r.json();
  const vectors = (json?.data || []).map(d => d.embedding);
  return vectors;
}

async function chatComplete(systemPrompt, messages) {
  const url = 'https://api.openai.com/v1/chat/completions';
  const body = {
    model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
    temperature: 0.2,
    messages: [{ role: 'system', content: systemPrompt }, ...messages]
  };
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI chat ${r.status}: ${txt}`);
  }
  const json = await r.json();
  return json?.choices?.[0]?.message?.content || '';
}

// -------------------------
// Pinecone
// -------------------------
async function pineconeQuerySafe({ indexHost, namespace, vector, topK = 8, filter = {} }, tries = 3) {
  if (!indexHost) throw new Error('Missing Pinecone host');
  const url = `https://${indexHost}/query`;

  // Pinecone rejects empty filter object — use undefined if empty
  const finalFilter = (filter && typeof filter === 'object' && Object.keys(filter).length)
    ? filter
    : undefined;

  const body = {
    vector,
    topK,
    includeMetadata: true,
    includeValues: false,
    namespace: namespace || undefined,
    filter: finalFilter
  };

  let lastErr;
  for (let i = 1; i <= tries; i++) {
    try {
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Api-Key': PINECONE_API_KEY },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => '');
        if ([408, 425, 429, 500, 502, 503, 504].includes(r.status) && i < tries) {
          await new Promise(res => setTimeout(res, 500 * i));
          continue;
        }
        throw new Error(`Pinecone ${r.status}: ${txt}`);
      }
      return await r.json();
    } catch (e) {
      lastErr = e;
      if (i < tries) {
        await new Promise(res => setTimeout(res, 600 * i));
        continue;
      }
    }
  }
  throw lastErr || new Error('Pinecone query failed');
}

// -------------------------
// Health endpoints
// -------------------------
app.get('/ping', (_req, res) => res.send('pong'));

app.get('/health', (_req, res) => {
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() });
});

app.get('/selftest', async (_req, res) => {
  const out = {
    env: {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX,
      PINECONE_HOST: !!PINECONE_HOST
    },
    openai: {},
    pinecone: {}
  };

  try {
    const [vec] = await embedTexts(['hello world']);
    out.openai.ok = Array.isArray(vec) && vec.length === 1536;
    out.openai.dim = Array.isArray(vec) ? vec.length : 0;
  } catch (e) {
    out.openai.ok = false;
    out.openai.err = String(e.message || e);
  }

  try {
    if (USE_EMBEDDINGS && PINECONE_HOST) {
      const probe = await pineconeQuerySafe({
        indexHost: PINECONE_HOST,
        namespace: undefined,
        vector: new Array(1536).fill(0),
        topK: 1,
        filter: { probe: true }
      });
      out.pinecone.ok = true;
      out.pinecone.note = `topK=${(probe?.matches || []).length}`;
    } else {
      out.pinecone.ok = false;
      out.pinecone.note = 'skipped (USE_EMBEDDINGS=false or no PINECONE_HOST)';
    }
  } catch (e) {
    out.pinecone.ok = false;
    out.pinecone.err = String(e.message || e);
  }

  res.json(out);
});

// -------------------------
// Main query endpoint
// -------------------------
app.post('/query', async (req, res) => {
  try {
    const {
      sessionId: rawSessionId,
      lang: langIn,
      stage: stageIn,
      subject: subjectIn,
      question: rawQuestion
    } = req.body || {};

    const sessionId = (rawSessionId || '').trim() || `${req.ip}:${Date.now()}`;
    const session = getOrInitSession(sessionId);

    // Merge preferences if provided
    if (langIn) session.lang = (String(langIn).toUpperCase() === 'AR') ? 'AR' : 'EN';
    if (stageIn) session.stage = String(stageIn || '');
    if (subjectIn) session.subject = String(subjectIn || '');

    // Safe question text
    const question = String(rawQuestion || '').trim();

    // Language toggle by keyword
    if (/^\s*arabic\s*$/i.test(question)) {
      session.lang = 'AR';
      pushHistory(session, 'user', question);
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [], debug: { lang: 'AR', switched: true } });
    }
    if (/^\s*english\s*$/i.test(question)) {
      session.lang = 'EN';
      pushHistory(session, 'user', question);
      return res.json({ answer: 'Done! Ask your question in English.', sources: [], debug: { lang: 'EN', switched: true } });
    }

    const lang = session.lang;
    const stage = session.stage;
    const subject = session.subject;

    // If no question, nudge
    if (!question) {
      return res.json({ answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.', sources: [] });
    }

    // Remember user question
    pushHistory(session, 'user', question);

    // --- Retrieval (optional & resilient) ---
    const needMCQs = detectMCQ(question, lang);
    let matches = [];
    let pineconeNote = null;

    if (USE_EMBEDDINGS && PINECONE_HOST) {
      try {
        const [embed] = await embedTexts([question]);

        const namespace = (lang || stage || subject)
          ? `${(lang || '').trim()}::${(stage || '').trim()}::${(subject || '').trim()}`
          : undefined;

        const filter = {};
        if (lang) filter.lang = lang;
        if (stage) filter.stage = stage;
        if (subject) filter.subject = subject;

        const pcResp = await pineconeQuerySafe({
          indexHost: PINECONE_HOST,
          namespace,
          vector: embed,
          topK: needMCQs ? 12 : 8,
          filter
        });

        matches = (pcResp?.matches || []).filter(m => m?.metadata);
        pineconeNote = `ok (matches=${matches.length})`;
      } catch (err) {
        pineconeNote = `skipped (fallback) — ${String(err && err.message || err)}`;
        matches = []; // fall back to zero-context
      }
    } else {
      pineconeNote = 'disabled or no host';
    }

    // Build context text from matches (cap ~2.8k chars)
    let contextText = '';
    for (const m of matches) {
      const chunk = m?.metadata?.text || m?.metadata?.chunk || '';
      if (!chunk) continue;
      if ((contextText + '\n\n' + chunk).length > 2800) break;
      contextText += (contextText ? '\n\n' : '') + chunk;
    }

    // Small “no context” hint
    const noContextHint = matches.length === 0
      ? (lang === 'AR'
          ? '⚠️ لا يوجد سياق مسترجع الآن. أجب بإيجاز واذكر أن المصادر غير متاحة.'
          : '⚠️ No retrieved context right now. Answer briefly and mention that sources are unavailable.')
      : '';

    // Personality / guardrails
    const sys = (lang === 'AR')
      ? [
          'أنت مساعد دراسي لطلاب الصيدلة. أجب بإيجاز وبأسلوب امتحاني.',
          'استخدم عناوين فرعية وخطوات واضحة عند الحاجة.',
          'إن طُلِب MCQs، قدم أسئلة بخيارات وإجابة نموذجية مع تبرير مختصر.',
          'اربط الإجابة بالسياق السابق إن وُجد.',
          noContextHint
        ].filter(Boolean).join('\n')
      : [
          'You are a study assistant for pharmacy students. Be concise and exam-oriented.',
          'Use short headers and clear steps when helpful.',
          'If asked for MCQs, include options + key + brief rationales.',
          'Tie your answer to the ongoing topic if present.',
          noContextHint
        ].filter(Boolean).join('\n');

    // Build messages (pinned topic + context + short memory + current question)
    const pinned = session.pinnedTopic
      ? (lang === 'AR'
          ? `الموضوع المثبت في هذه المحادثة: ${session.pinnedTopic}`
          : `Pinned topic in this chat: ${session.pinnedTopic}`)
      : null;

    const ctxLabel = lang === 'AR' ? 'سياق من المراجع:' : 'Context from references:';
    const userMsg = lang === 'AR' ? `سؤال الطالب: ${question}` : `Student question: ${question}`;

    const msgArr = [];
    if (pinned) msgArr.push({ role: 'system', content: pinned });
    if (contextText) msgArr.push({ role: 'system', content: `${ctxLabel}\n${contextText}` });

    // include recent short history
    for (const m of session.history.slice(-8)) {
      msgArr.push({ role: m.role, content: m.content });
    }
    msgArr.push({ role: 'user', content: userMsg });

    // --- OpenAI completion ---
    let finalAnswer = '';
    try {
      finalAnswer = await chatComplete(sys, msgArr);
    } catch (e) {
      // Last-resort friendly message
      return res.json({
        answer: lang === 'AR'
          ? 'عذرًا، حدث خطأ أثناء توليد الإجابة. حاول مرة أخرى.'
          : 'Sorry, something went wrong while generating the answer. Please try again.',
        sources: [],
        debug: { fatal: String(e.message || e) }
      });
    }

    // Save assistant reply
    pushHistory(session, 'assistant', finalAnswer);

    // Pin a topic after first good answer (helps follow-ups like “compare …”, “MCQs please”)
    if (!session.pinnedTopic) {
      const t = extractTopic(finalAnswer);
      if (t) session.pinnedTopic = t;
    }

    // Pack sources (cleaned)
    const sources = cleanSources(matches);

    res.json({
      answer: finalAnswer,
      sources,
      debug: {
        lang, stage, subject,
        pinnedTopic: session.pinnedTopic || null,
        pinecone: { ok: matches.length > 0, note: pineconeNote }
      }
    });
  } catch (e) {
    // Never crash the request
    res.status(200).json({
      answer: 'Sorry, an unexpected error occurred. Please try again.',
      sources: [],
      debug: { catch: String(e.message || e) }
    });
  }
});

// -------------------------
// Start server
// -------------------------
app.listen(PORT, () => {
  console.log(`PharmaNinja backend listening on :${PORT}`);
});
