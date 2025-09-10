// server.js — ESM version (Render-friendly) with graceful Pinecone fallback
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { Agent as HttpsAgent } from 'node:https';

// -------------------------
// Config & constants
// -------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'pharmaninja-bot-1536';
const PINECONE_HOST = process.env.PINECONE_HOST || ''; // <— set this in Render

const PORT = process.env.PORT || 3000;

// Embedding model must match your Pinecone index dim (1536)
const EMBEDDING_MODEL = 'text-embedding-3-small';

// Small in-memory session store
const SESSIONS = new Map(); // sessionId -> { lang, stage, subject, pinnedTopic, history: [{role, content}], ts }

// Optional IPv4-only agent (used only for Node core HTTPS; fetch(undici) doesn’t use this)
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

function extractTopic(text) {
  if (!text) return null;
  const firstLine = text.split('\n').find(l => l.trim()) || '';
  const t = firstLine.replace(/^#+\s*/, '').replace(/\*\*|__/g, '').trim();
  if (t.length >= 6 && t.length <= 120) return t;
  return text.split(/\s+/).slice(0, 10).join(' ');
}

// -------------------------
// OpenAI
// -------------------------
async function embedTexts(texts) {
  const r = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: texts })
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings ${r.status}: ${txt}`);
  }
  const json = await r.json();
  return (json?.data || []).map(d => d.embedding);
}

async function chatComplete(systemPrompt, messages) {
  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      temperature: 0.2,
      messages: [{ role: 'system', content: systemPrompt }, ...messages]
    })
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI chat ${r.status}: ${txt}`);
  }
  const json = await r.json();
  return json?.choices?.[0]?.message?.content || '';
}

// -------------------------
// Pinecone (graceful)
// -------------------------
async function pineconeQuerySafe({ host, namespace, vector, topK = 8, filter = {} }, tries = 2) {
  if (!PINECONE_API_KEY || !host) {
    // Explicitly signal that retrieval is disabled/missing config
    const why = !PINECONE_API_KEY ? 'no_api_key' : 'no_host';
    const err = new Error(`PINECONE_DISABLED:${why}`);
    err.code = 'PINECONE_DISABLED';
    throw err;
  }

  const url = `https://${host}/query`;
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
        // Retry only on transient statuses
        if ([408, 425, 429, 500, 502, 503, 504].includes(r.status) && i < tries) {
          await new Promise(res => setTimeout(res, 500 * i));
          continue;
        }
        throw new Error(`Pinecone ${r.status}: ${txt}`);
      }
      const json = await r.json();
      return json;
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
    const probe = await pineconeQuerySafe({
      host: PINECONE_HOST,
      namespace: undefined,
      vector: new Array(1536).fill(0),
      topK: 1,
      filter: { probe: true }
    });
    out.pinecone.ok = true;
    out.pinecone.matches = (probe?.matches || []).length;
  } catch (e) {
    out.pinecone.ok = false;
    out.pinecone.err = String(e.message || e);
  }

  res.json(out);
});

// -------------------------
// Main query
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

    if (langIn) session.lang = (String(langIn).toUpperCase() === 'AR') ? 'AR' : 'EN';
    if (stageIn) session.stage = String(stageIn);
    if (subjectIn) session.subject = String(subjectIn);

    const question = String(rawQuestion || '').trim();

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

    if (!question) {
      return res.json({ answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.', sources: [] });
    }

    pushHistory(session, 'user', question);

    // Embedding (for retrieval if available)
    let embed = null;
    try {
      [embed] = await embedTexts([question]);
    } catch (e) {
      // If embeddings fail, continue without retrieval
      embed = null;
    }

    // Retrieval (best-effort only)
    const needMCQs = detectMCQ(question, lang);
    let matches = [];
    let pineconeInfo = { ok: false, note: 'skipped' };

    if (embed) {
      try {
        const namespace = (lang || stage || subject)
          ? `${(lang || '').trim()}::${(stage || '').trim()}::${(subject || '').trim()}`
          : undefined;

        const filter = {};
        if (lang) filter.lang = lang;
        if (stage) filter.stage = stage;
        if (subject) filter.subject = subject;

        const pc = await pineconeQuerySafe({
          host: PINECONE_HOST,
          namespace,
          vector: embed,
          topK: needMCQs ? 12 : 8,
          filter
        });
        matches = (pc?.matches || []).filter(m => m?.metadata);
        pineconeInfo = { ok: true, note: `matches=${matches.length}` };
      } catch (e) {
        // swallow Pinecone errors; proceed with empty context
        pineconeInfo = { ok: false, note: String(e.message || e) };
      }
    }

    // Build short context text
    let contextText = '';
    for (const m of matches) {
      const chunk = m?.metadata?.text || m?.metadata?.chunk || '';
      if (!chunk) continue;
      if ((contextText + '\n\n' + chunk).length > 2800) break;
      contextText += (contextText ? '\n\n' : '') + chunk;
    }

    const noContextHint = matches.length === 0
      ? (lang === 'AR'
          ? '⚠️ لا يوجد سياق مسترجع الآن. أجب بإيجاز واذكر أن المصادر غير متاحة.'
          : '⚠️ No retrieved context right now. Answer briefly and note that sources are unavailable.')
      : '';

    const sys = (lang === 'AR')
      ? [
          'أنت مساعد دراسي لطلاب الصيدلة. أجب بإيجاز وبأسلوب امتحاني.',
          'استخدم عناوين وخطوات واضحة عند الحاجة.',
          'إن طُلب MCQs، قدّم أسئلة بخيارات وإجابات مختصرة.',
          'اربط الردود بسياق المحادثة إن وُجد.',
          noContextHint
        ].filter(Boolean).join('\n')
      : [
          'You are a study assistant for pharmacy students. Be concise and exam-oriented.',
          'Use clear headers and steps when helpful.',
          'If asked for MCQs, provide choices with a brief key.',
          'Tie responses to the ongoing chat topic.',
          noContextHint
        ].filter(Boolean).join('\n');

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

    for (const m of session.history.slice(-8)) {
      msgArr.push({ role: m.role, content: m.content });
    }
    msgArr.push({ role: 'user', content: userMsg });

    const finalAnswer = await chatComplete(sys, msgArr);
    pushHistory(session, 'assistant', finalAnswer);

    if (!session.pinnedTopic) {
      const t = extractTopic(finalAnswer);
      if (t) session.pinnedTopic = t;
    }

    return res.json({
      answer: finalAnswer,
      sources: cleanSources(matches),
      debug: {
        lang, stage, subject,
        pinnedTopic: session.pinnedTopic || null,
        pinecone: pineconeInfo
      }
    });
  } catch (err) {
    // Last-resort safety: never crash the request
    const msg = String(err?.message || err || 'Unknown error');
    return res.status(200).json({
      answer: 'Sorry, something went wrong while generating the answer. Please try again.',
      sources: [],
      debug: { fatal: msg }
    });
  }
});

// -------------------------
// Start server
// -------------------------
app.listen(PORT, () => {
  console.log(`PharmaNinja backend listening on :${PORT}`);
});
