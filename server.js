// server.js
// Pharmaninja backend — ready to paste
// Requires: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX
// Optional: PORT (default 3000)

import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

// --- Guard env ---
if (!OPENAI_API_KEY) console.warn('WARN: OPENAI_API_KEY missing');
if (!PINECONE_API_KEY) console.warn('WARN: PINECONE_API_KEY missing');
if (!PINECONE_INDEX) console.warn('WARN: PINECONE_INDEX missing');

// --- OpenAI endpoints ---
const OA_CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const OA_EMB_URL  = 'https://api.openai.com/v1/embeddings';
// Models (embedding dim 1536)
const CHAT_MODEL  = 'gpt-4o-mini';
const EMB_MODEL   = 'text-embedding-3-small';

// --- Pinecone endpoints ---
const PC_HOST = `https://${PINECONE_INDEX}.svc.${inferPineconeEnv()}.pinecone.io`;
// If you know your environment (e.g., "gcp-starter"), hardcode it above for speed

// --- Tiny in-memory session store ---
const SESS = new Map();
/*
  Session shape:
  {
    lang: 'EN'|'AR',
    stage: '3rd' | ...,
    subject: 'Pharmacology' | ...,
    topic: 'string | null',
    history: [ {role:'user'|'assistant', content:'...'} ] (short)
  }
*/

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---------- Utilities ----------
function normalizeLang(x) {
  if (!x) return null;
  const t = String(x).trim().toUpperCase();
  if (['EN','ENGLISH'].includes(t)) return 'EN';
  if (['AR','ARABIC','ع','عربي','العربية'].includes(t)) return 'AR';
  return null;
}

function ensureSession(id) {
  if (!SESS.has(id)) SESS.set(id, { lang: 'EN', stage: null, subject: null, topic: null, history: [] });
  return SESS.get(id);
}

function shortPushHistory(s, turn) {
  s.history.push(turn);
  // keep last ~6 turns (user+assistant pairs)
  if (s.history.length > 12) s.history.splice(0, s.history.length - 12);
}

function buildFilter(s) {
  const f = {};
  if (s.stage) f.stage = s.stage;
  if (s.subject) f.subject = s.subject;
  // NOTE: we do NOT force lang in the filter — mixed language PDFs can still be useful
  return Object.keys(f).length ? f : null;
}

function languageGuardText(lang) {
  return (lang === 'AR')
    ? 'أجب باللغة العربية فقط. لا تستخدم أي لغة أخرى على الإطلاق.'
    : 'Answer ONLY in English. Do not use any other language.';
}

function prefsLine(s, lang) {
  if (lang === 'AR') {
    const parts = [];
    if (s.stage) parts.push(`المرحلة: ${s.stage}`);
    if (s.subject) parts.push(`المادة: ${s.subject}`);
    return parts.length ? `تفضيلات المستخدم: ${parts.join(' | ')}` : 'تفضيلات المستخدم: (غير محددة)';
  }
  const parts = [];
  if (s.stage) parts.push(`Stage: ${s.stage}`);
  if (s.subject) parts.push(`Subject: ${s.subject}`);
  return parts.length ? `User prefs: ${parts.join(' | ')}` : 'User prefs: (not set)';
}

function topicLine(s, lang) {
  if (!s.topic) return (lang === 'AR') ? 'الموضوع المثبّت: (لا يوجد)' : 'Pinned topic: (none)';
  return (lang === 'AR') ? `الموضوع المثبّت: ${s.topic}` : `Pinned topic: ${s.topic}`;
}

function histBlockText(s, lang) {
  if (!s.history?.length) return (lang === 'AR') ? 'تاريخ المحادثة: (لا يوجد)' : 'Conversation history: (none)';
  const lastFew = s.history.slice(-6);
  const asText = lastFew.map(h => `${h.role === 'user' ? (lang==='AR'?'المستخدم':'User') : (lang==='AR'?'المساعد':'Assistant')}: ${h.content}`).join('\n');
  return (lang === 'AR') ? `تاريخ المحادثة (الأحدث أولاً):\n${asText}` : `Conversation history (newest last):\n${asText}`;
}

function detectArabicSwitch(q) {
  // if user just says "arabic" (case-insensitive) switch to AR
  return /^\s*arabic\s*$/i.test(q);
}
function detectEnglishSwitch(q) {
  // if user just says "english"
  return /^\s*english\s*$/i.test(q);
}

function detectMCQ(q, lang) {
  if (lang === 'AR') {
    return /(\bMCQ\b|اختيار|اختر|أسئلة اختيار|اختيار من متعدد)/i.test(q);
  }
  return /\b(MCQ|MCQs|multiple[-\s]?choice)\b/i.test(q);
}

function detectCompare(q, lang) {
  if (lang === 'AR') return /(قارن|مقارنة)/i.test(q);
  return /\b(compare|vs\.?)\b/i.test(q);
}

function rewriteCompare(q, lang, s) {
  // Try to capture the other target (e.g., "compare with glycopeptides")
  let other = null;
  if (lang === 'AR') {
    const m = q.match(/(?:قارن(?:\s+بين)?|مقارنة(?:\s+بين)?)\s+(.*)$/i);
    if (m && m[1]) other = m[1].trim();
  } else {
    const m = q.match(/compare(?:\s+with|\s+to)?\s+(.*)$/i);
    if (m && m[1]) other = m[1].trim();
  }
  const pinned = s.topic || null;
  if (pinned && other) {
    return (lang === 'AR')
      ? `قدّم مقارنة موجزة ومنظمة بين "${pinned}" و "${other}". ركّز على آلية العمل، الطيف، الاستطبابات، الآثار الجانبية، والمقاومة.`
      : `Provide a concise, structured comparison between "${pinned}" and "${other}". Focus on MoA, spectrum, indications, adverse effects, and resistance.`;
  }
  // fall back: ask for both sides clearly
  return (lang === 'AR')
    ? `أعد صياغة السؤال: مقارنة، ولكن حدّد الطرفين بوضوح.`
    : `Rewrite as a comparison, but clearly specify both sides.`;
}

function mcqHintText(lang) {
  return (lang === 'AR')
    ? 'إذا كان السؤال يطلب MCQs، اصنع 5 أسئلة MCQ مع الإجابات النموذجية والنقطة التعليمية بعد كل سؤال.'
    : 'If the user asks for MCQs, produce 5 exam-style MCQs with correct answers and a one-line learning point after each.';
}

// ---------- RAG (Pinecone) ----------
async function embedQuery(text) {
  const r = await fetch(OA_EMB_URL, {
    method: 'POST',
    headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMB_MODEL, input: text })
  });
  if (!r.ok) throw new Error(`OpenAI Embeddings ${r.status}: ${await r.text()}`);
  const j = await r.json();
  const v = j?.data?.[0]?.embedding;
  if (!v) throw new Error('No embedding returned');
  return v;
}

async function pineconeQuery({ vector, topK = 6, filter = null, includeMetadata = true }) {
  const url = `${PC_HOST}/query`;
  const body = {
    vector,
    topK,
    includeMetadata,
  };
  if (filter && Object.keys(filter).length) body.filter = filter;

  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type':'application/json',
      'Api-Key': PINECONE_API_KEY
    },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`Pinecone ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j?.matches || [];
}

function renderContextFromMatches(matches, lang) {
  if (!matches?.length) return '';
  const lines = matches.map((m, i)=> {
    const meta = m.metadata || {};
    const src = meta.file || meta.id || `doc-${i+1}`;
    const chunk = (meta.text || meta.content || '').slice(0, 1200);
    return `[${i+1}] Source: ${src}\n${chunk}`;
  });
  return lines.join('\n\n');
}

function collectSources(matches) {
  return (matches || []).map((m, i)=> {
    const meta = m.metadata || {};
    return {
      id: meta.id || meta.file || `doc-${i+1}`,
      file: meta.file || undefined,
      score: typeof m.score === 'number' ? m.score : undefined
    };
  });
}

// ---------- OpenAI Chat ----------
async function chatComplete(messages, lang) {
  const r = await fetch(OA_CHAT_URL, {
    method: 'POST',
    headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: CHAT_MODEL,
      temperature: 0.3,
      messages
    })
  });
  if (!r.ok) throw new Error(`OpenAI Chat ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j?.choices?.[0]?.message?.content || (lang==='AR' ? 'عذرًا، حدث خطأ.' : 'Sorry, something went wrong.');
}

// ---------- Routes ----------
app.get('/ping', (_req, res)=> res.type('text').send('pong'));

app.get('/health', async (_req, res) => {
  res.json({ ok: true, uptime: process.uptime()*1000, ts: Date.now() });
});

app.get('/selftest', async (_req, res) => {
  const env = {
    OPENAI_API_KEY: !!OPENAI_API_KEY,
    PINECONE_API_KEY: !!PINECONE_API_KEY,
    PINECONE_INDEX: PINECONE_INDEX || null
  };
  // very light smoke checks
  let openai = { ok: false, dim: null };
  try {
    const v = await embedQuery('ping');
    openai = { ok: true, dim: v.length };
  } catch(e) { openai = { ok: false, err: String(e.message).slice(0,180) }; }

  let pinecone = { ok: false, matches: null };
  try {
    const v = await embedQuery('ping pinecone');
    const f = null; // no filter for smoke
    const matches = await pineconeQuery({ vector: v, topK: 1, filter: f });
    pinecone = { ok: true, matches: matches.length };
  } catch(e) { pinecone = { ok: false, err: String(e.message).slice(0,180) }; }

  res.json({ env, openai, pinecone });
});

app.post('/query', async (req, res) => {
  try {
    const { sessionId, question: rawQ, lang: bodyLang, stage, subject } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: 'sessionId required' });

    const q = (String(rawQ || '')).trim();
    const s = ensureSession(sessionId);

    // Persist prefs if provided this turn
    const langFromBody = normalizeLang(bodyLang);
    if (langFromBody) s.lang = langFromBody;
    if (stage) s.stage = stage;
    if (subject) s.subject = subject;

    // Language switch shortcuts
    if (detectArabicSwitch(q)) {
      s.lang = 'AR';
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [], debug: { lang: s.lang, switched: true }});
    }
    if (detectEnglishSwitch(q)) {
      s.lang = 'EN';
      return res.json({ answer: 'Done! Ask your question in English.', sources: [], debug: { lang: s.lang, switched: true }});
    }

    // Language (locked)
    const lang = s.lang || 'EN';

    // Topic pinning: if first turn or explicit "explain ..." treat as topic setter
    // (only pin if user asks an informative question)
    const isFirstTopicLike = !s.topic && q.length >= 8;
    if (isFirstTopicLike) {
      // Pin a short topic label (first 8 words)
      s.topic = (q.split(/\s+/).slice(0, 8).join(' ')).trim();
    }

    // If compare asked, rewrite to include pinned topic (if any)
    let rewrittenQuestion = q;
    const isCompare = detectCompare(q, lang);
    if (isCompare) {
      rewrittenQuestion = rewriteCompare(q, lang, s);
    }

    // Build Pinecone filter safely (optional)
    const filter = buildFilter(s);

    // Embed and retrieve
    const emb = await embedQuery(rewrittenQuestion);
    const matches = await pineconeQuery({ vector: emb, topK: 6, filter });

    const context = renderContextFromMatches(matches, lang);
    const sources = collectSources(matches);

    // MCQ hint if requested
    const wantsMCQ = detectMCQ(q, lang);
    const mcqHint = wantsMCQ ? (lang==='AR'
        ? 'اصنع 5 أسئلة اختيار من متعدد (MCQ) مع الإجابة الصحيحة وتفسير سطر واحد لكل سؤال.'
        : 'Create 5 multiple-choice questions (MCQs) with the correct answer and a one-line explanation for each.'
      ) : '';

    // Language guard first, then system behavior
    const languageGuard = languageGuardText(lang);

    const sys = (lang === 'AR')
      ? 'أنت مساعد دراسي مختصر وعملي لطلاب الصيدلة. قدّم إجابات موجزة، منظمة، ومناسبة للامتحانات.'
      : 'You are a concise, exam-oriented pharmacy tutor. Give succinct, structured answers.';

    const prefs = prefsLine(s, lang);
    const topicLn = topicLine(s, lang);
    const histBlock = histBlockText(s, lang);
    const ctxLabel = (lang === 'AR') ? 'السياق' : 'Context';
    const qLabel   = (lang === 'AR') ? 'السؤال' : 'Question';

    const userContent =
`Answer language: ${lang}
${prefs}
${topicLn}
${histBlock}

${ctxLabel}:
${context || (lang === 'AR' ? '(لا سياق متاح)' : '(no context available)')}

${qLabel}: ${rewrittenQuestion}

${mcqHint}
`;

    const messages = [
      { role: 'system', content: languageGuard },
      { role: 'system', content: sys },
      { role: 'user', content: userContent }
    ];

    const answer = await chatComplete(messages, lang);

    // Update history + refine topic if needed
    shortPushHistory(s, { role: 'user', content: q });
    shortPushHistory(s, { role: 'assistant', content: answer });
    // Topic refinement: if we didn't have one but the model clearly summarized a subject line, keep the initial pin

    return res.json({
      answer,
      sources: sources.slice(0, 5),
      debug: {
        lang,
        stage: s.stage || null,
        subject: s.subject || null,
        topic: s.topic || null,
        rewrittenQuestion,
        filterUsed: filter || '(none)',
        hits: matches.length
      }
    });
  } catch (e) {
    console.error('ERROR /query:', e);
    return res.status(500).json({ error: 'Query failed', detail: String(e.message).slice(0, 300) });
  }
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`Pharmaninja backend listening on :${PORT}`);
});

// ---------- Helpers ----------
function inferPineconeEnv() {
  // If you know your env (e.g., 'gcp-starter'), hardcode it:
  // return 'gcp-starter';
  // Else default to 'gcp-starter' (common in tutorials). Adjust if needed.
  return 'gcp-starter';
}
