// server.js (PharmaNinja Backend) — 2025-09-09

// -------------------------
// Imports & basic setup
// -------------------------
const express = require('express');
const cors = require('cors');
const https = require('https');

let fetchFn = global.fetch;
try {
  if (!fetchFn) fetchFn = require('node-fetch');
} catch (_) {
  // Node 18+ has global.fetch; ignore if node-fetch not present
}
const fetch = (...args) => fetchFn(...args);

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// -------------------------
// Config & constants
// -------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'pharmaninja-bot-1536';
// If you know your fixed host, you can set PINECONE_HOST in Render env.
// Else we infer from index name on GCP starter tier:
const PINECONE_HOST = process.env.PINECONE_HOST || `${PINECONE_INDEX}.svc.gcp-starter.pinecone.io`;

const PORT = process.env.PORT || 3000;

// Use the 1536-dimensional embedding model to match your index
const EMBEDDING_MODEL = 'text-embedding-3-small'; // 1536 dims

// A very small rolling memory per session to help follow-ups.
// If you want persistence across restarts, swap this for Redis.
const SESSIONS = new Map(); // sessionId -> { lang, stage, subject, pinnedTopic, history: [{role, content}], ts }

// Keep sockets alive & force IPv4 to avoid TLS/IPv6 edge cases
const agentV4 = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  timeout: 30000,
  family: 4
});

// -------------------------
// Helpers
// -------------------------

function getOrInitSession(sessionId) {
  let s = SESSIONS.get(sessionId);
  if (!s) {
    s = {
      lang: 'EN',
      stage: null,
      subject: null,
      pinnedTopic: null,
      history: [], // last 10 user/assistant messages
      ts: Date.now()
    };
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

// Simple topic extraction (cheap heuristic): first title-ish sentence of the first assistant answer
function extractTopic(text) {
  if (!text) return null;
  const firstLine = text.split('\n').find(l => l.trim().length > 0) || '';
  // strip markdown headers and bold
  const t = firstLine.replace(/^#+\s*/, '').replace(/\*\*|__/g, '').trim();
  if (t.length >= 6 && t.length <= 120) return t;
  // fallback: first 10 words
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
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body),
    agent: agentV4
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings ${r.status}: ${txt}`);
  }
  const json = await r.json();
  const vectors = (json?.data || []).map(d => d.embedding);
  return vectors;
}

async function chatComplete(systemPrompt, messages, lang) {
  // Use a modern gpt model name if you have access; fallback to gpt-4o-mini/4.1-mini style
  const url = 'https://api.openai.com/v1/chat/completions';
  const body = {
    model: 'gpt-4o-mini',
    temperature: 0.2,
    messages: [
      { role: 'system', content: systemPrompt },
      ...messages
    ]
  };
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body),
    agent: agentV4
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI chat ${r.status}: ${txt}`);
  }
  const json = await r.json();
  const text = json?.choices?.[0]?.message?.content || '';
  return text;
}

// -------------------------
// Pinecone
// -------------------------

async function pineconeQuerySafe({ indexHost, namespace, vector, topK = 8, filter = {} }, tries = 3) {
  if (!indexHost) throw new Error('Missing Pinecone host');
  const url = `https://${indexHost}/query`;

  const finalFilter = (filter && typeof filter === 'object' && Object.keys(filter).length)
    ? filter
    : undefined; // IMPORTANT: undefined if empty so Pinecone doesn't error

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
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': PINECONE_API_KEY
        },
        body: JSON.stringify(body),
        agent: agentV4
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => '');
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
app.get('/ping', (req, res) => res.send('pong'));

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    uptime: process.uptime(),
    ts: Date.now()
  });
});

app.get('/selftest', async (req, res) => {
  const out = {
    env: {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX
    },
    openai: {},
    pinecone: {}
  };
  try {
    const [vec] = await embedTexts(['hello world']);
    out.openai.ok = Array.isArray(vec) && vec.length > 0;
    out.openai.dim = Array.isArray(vec) ? vec.length : 0;
  } catch (e) {
    out.openai.ok = false;
    out.openai.note = String(e.message || e);
  }

  try {
    const probe = await pineconeQuerySafe({
      indexHost: PINECONE_HOST,
      namespace: undefined,
      vector: new Array(1536).fill(0),
      topK: 1,
      filter: { probe: true } // likely no results; just ensures call works
    });
    out.pinecone.ok = true;
    out.pinecone.note = `topK=${(probe?.matches || []).length}`;
  } catch (e) {
    out.pinecone.ok = false;
    out.pinecone.note = String(e.message || e);
  }

  res.json(out);
});

// -------------------------
// Main query endpoint
// -------------------------
app.post('/query', async (req, res) => {
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
  if (langIn) session.lang = (langIn || '').toUpperCase() === 'AR' ? 'AR' : 'EN';
  if (stageIn) session.stage = String(stageIn || '');
  if (subjectIn) session.subject = String(subjectIn || '');

  // Safe message text
  const question = String(rawQuestion || '').trim();

  // Language toggle by keyword
  if (/^\s*arabic\s*$/i.test(question)) {
    session.lang = 'AR';
    pushHistory(session, 'user', question);
    return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
  }
  if (/^\s*english\s*$/i.test(question)) {
    session.lang = 'EN';
    pushHistory(session, 'user', question);
    return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
  }

  const lang = session.lang;
  const stage = session.stage;
  const subject = session.subject;

  // If no question, nudge
  if (!question) {
    return res.json({
      answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.',
      sources: []
    });
  }

  // Remember user question
  pushHistory(session, 'user', question);

  // Create embedding for retrieval
  let embed;
  try {
    [embed] = await embedTexts([question]);
  } catch (e) {
    return res.status(500).json({ error: 'Embedding failed', detail: String(e.message || e) });
  }

  // Pinecone query with safe retries + graceful fallback
  const needMCQs = detectMCQ(question, lang);
  let matches = [];
  let pineconeError = null;

  try {
    const namespace =
      (lang || stage || subject) ? `${(lang || '').trim()}::${(stage || '').trim()}::${(subject || '').trim()}` : undefined;

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
  } catch (err) {
    pineconeError = String(err && err.message || err);
    console.warn('Pinecone fallback -> using empty context:', pineconeError);
    matches = [];
  }

  // Build context text from matches (caps at ~2-3k chars)
  let contextText = '';
  for (const m of matches) {
    const chunk = m?.metadata?.text || m?.metadata?.chunk || '';
    if (!chunk) continue;
    if ((contextText + '\n\n' + chunk).length > 2800) break;
    contextText += (contextText ? '\n\n' : '') + chunk;
  }

  // Small “no context” hint to steer the model
  const noContextHint = matches.length === 0
    ? (lang === 'AR'
      ? '⚠️ لا يوجد سياق مسترجع الآن. أجب بإيجاز واذكر أن المصادر غير متاحة.'
      : '⚠️ No retrieved context right now. Answer briefly and note that sources are unavailable.')
    : '';

  // System prompt (personality + guardrails)
  const sys = (lang === 'AR')
    ? [
        'أنت مساعد دراسي لطلاب الصيدلة. أجب بإيجاز وبأسلوب امتحاني.',
        'استخدم خطوات منظمة وعناوين فرعية عند الحاجة.',
        'إن طُلب MCQs، قدم أسئلة بخيارات وإجابة نموذجية، مع تبرير مختصر.',
        'اربط الردود بالسياق السابق للطالب إن وُجد.',
        noContextHint
      ].filter(Boolean).join('\n')
    : [
        'You are a study assistant for pharmacy students. Be concise and exam-oriented.',
        'Use clear steps and small headers when helpful.',
        'If asked for MCQs, provide multiple-choice items with the answer key and brief rationales.',
        'Tie responses to the ongoing topic if present.',
        noContextHint
      ].filter(Boolean).join('\n');

  // Build chat messages: brief memory + (optional) pinned topic + retrieved context
  const pinned = session.pinnedTopic ? (lang === 'AR'
    ? `الموضوع المثبت في هذه المحادثة: ${session.pinnedTopic}`
    : `Pinned topic in this chat: ${session.pinnedTopic}`) : null;

  const ctxLabel = lang === 'AR' ? 'سياق من المراجع:' : 'Context from references:';
  const userMsg = lang === 'AR'
    ? `سؤال الطالب: ${question}`
    : `Student question: ${question}`;

  const msgArr = [];
  if (pinned) msgArr.push({ role: 'system', content: pinned });
  if (contextText) msgArr.push({ role: 'system', content: `${ctxLabel}\n${contextText}` });

  // include short rolling history (user/assistant only)
  for (const m of session.history.slice(-8)) {
    // Avoid duplicating this turn's user message here; we add fresh one next.
    msgArr.push({ role: m.role, content: m.content });
  }
  msgArr.push({ role: 'user', content: userMsg });

  let finalAnswer = '';
  try {
    finalAnswer = await chatComplete(sys, msgArr, lang);
  } catch (e) {
    return res.status(500).json({ error: 'Chat failed', detail: String(e.message || e) });
  }

  // Save assistant reply to memory
  pushHistory(session, 'assistant', finalAnswer);

  // Pin a topic if not pinned yet (help “compare …” & “MCQs please”)
  if (!session.pinnedTopic) {
    const t = extractTopic(finalAnswer);
    if (t) session.pinnedTopic = t;
  }

  // Prepare sources (top 5 cleaned)
  const sources = cleanSources(matches);

  // Answer out
  res.json({
    answer: finalAnswer,
    sources,
    debug: {
      usedMatches: matches.length,
      pineconeError: pineconeError || null,
      lang,
      stage,
      subject,
      pinnedTopic: session.pinnedTopic || null
    }
  });
});

// -------------------------
// Start server
// -------------------------
app.listen(PORT, () => {
  console.log(`PharmaNinja backend listening on :${PORT}`);
});
