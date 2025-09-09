// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { Pinecone } from '@pinecone-database/pinecone';

// ----------------------------
// Basic setup
// ----------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX);

// In-memory sessions (simple)
const sessions = Object.create(null);

// ----------------------------
// Helpers
// ----------------------------
function nonEmpty(obj) {
  return obj && typeof obj === 'object' && Object.keys(obj).length > 0;
}
function getSession(id) {
  if (!sessions[id]) sessions[id] = { history: [] };
  return sessions[id];
}
function trimStr(s, n = 1000) {
  if (!s) return '';
  return s.length <= n ? s : s.slice(0, n) + '…';
}
function detectLangSwitch(q) {
  if (!q) return null;
  const t = q.trim().toLowerCase();
  if (t === 'arabic' || t === 'ar') return 'AR';
  if (t === 'english' || t === 'en') return 'EN';
  return null;
}
function isLikelyMCQRequest(q) {
  if (!q) return false;
  const t = q.toLowerCase();
  return /\bmcq/.test(t) || /multiple[- ]choice/.test(t) || /اختيار/.test(t);
}

// ----------------------------
// OpenAI helpers
// ----------------------------
async function embed(text) {
  const r = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: 'text-embedding-3-small' }, // 1536-dim
    { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding;
}

async function chat(messages, model = 'gpt-4o-mini') {
  const r = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    { model, messages, temperature: 0.2 },
    { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } }
  );
  return r.data.choices?.[0]?.message?.content?.trim() || '';
}

// ----------------------------
// Health endpoints
// ----------------------------
app.get('/ping', (req, res) => res.type('text/plain').send('pong'));

app.get('/health', (req, res) => {
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() });
});

app.get('/selftest', async (req, res) => {
  try {
    // Check env
    const env = {
      OPENAI_API_KEY: !!process.env.OPENAI_API_KEY,
      PINECONE_API_KEY: !!process.env.PINECONE_API_KEY,
      PINECONE_INDEX: process.env.PINECONE_INDEX || null
    };

    // Check embeddings
    let dim = 0;
    try {
      const v = await embed('hello');
      dim = v.length;
    } catch (e) {
      return res.json({ env, openai: { ok: false, note: `OpenAI embeddings error ${e.response?.status || ''}` } });
    }

    // Light Pinecone check (no filter)
    let pineOk = false;
    try {
      const pine = await index.query({ vector: Array(dim).fill(0), topK: 1, includeMetadata: true });
      pineOk = true;
    } catch {
      pineOk = false;
    }

    res.json({
      env,
      openai: { ok: dim > 0, dim },
      pinecone: { ok: pineOk, matches: pineOk ? 1 : 0 }
    });
  } catch (err) {
    res.status(500).json({ error: 'Selftest failed', detail: String(err?.message || err) });
  }
});

// ----------------------------
// Main query endpoint
// ----------------------------
app.post('/query', async (req, res) => {
  const body = req.body || {};
  const sessionId = body.sessionId || `${req.ip}`;
  const s = getSession(sessionId);

  try {
    let { lang, stage, subject, question } = body;

    // Mid-chat language switch
    const langSwitch = detectLangSwitch(question);
    if (langSwitch) {
      s.lang = langSwitch;
      return res.json({
        answer: langSwitch === 'AR' ? 'تم! اكتب سؤالك الآن بالعربية.' : 'Done! Ask your question in English.',
        sources: []
      });
    }

    // Keep / backfill session prefs
    lang = lang || s.lang || 'EN';
    stage = stage || s.stage;
    subject = subject || s.subject;

    if (!s.lang) s.lang = lang;
    if (!s.stage && stage) s.stage = stage;
    if (!s.subject && subject) s.subject = subject;

    // Pin a topic from the first substantive question
    if (!s.topic && question && question.length > 8) {
      s.topic = trimStr(question, 120);
    }

    // Empty question? Prompt to ask
    if (!question || !question.trim()) {
      return res.json({
        answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.',
        sources: []
      });
    }

    // Embed the user question
    const qVec = await embed(question);

    // ---------- Pinecone filter (SAFE) ----------
    const filter = {};
    if (stage) filter.stage = stage;
    if (subject) filter.subject = subject;
    if (s.topic) filter.topic = s.topic; // optional if you stored topic with each vector

    const pineParams = {
      vector: qVec,
      topK: 5,
      includeMetadata: true
    };
    if (nonEmpty(filter)) pineParams.filter = filter;

    const pine = await index.query(pineParams);
    const matches = pine.matches || [];

    // Build context from top matches
    const context = matches
      .map((m, i) => `[#${i + 1}] ${trimStr(m.metadata?.text || '', 700)}`)
      .join('\n\n');

    // ---- Build the chat messages ----
    const sys = (lang === 'AR')
      ? 'أنت مُعلِّم صيدلة مساعد وودود. اشرح بإيجاز وبأسلوب منظم للامتحان. إن لم تكن المعلومة في السياق، أذكر ذلك بصراحة.'
      : 'You are a helpful pharmacy study tutor. Be concise, exam-oriented, and structured. If info is not in context, say so.';

    const topicLine = s.topic
      ? (lang === 'AR' ? `الموضوع الحالي: ${s.topic}` : `Current topic: ${s.topic}`)
      : (lang === 'AR' ? 'لا يوجد موضوع مثبت بعد.' : 'No pinned topic yet.');

    const prefsLine = [
      stage ? (lang === 'AR' ? `المرحلة: ${stage}` : `Stage: ${stage}`) : null,
      subject ? (lang === 'AR' ? `المادة: ${subject}` : `Subject: ${subject}`) : null
    ].filter(Boolean).join(' | ');

    const mcqHint = isLikelyMCQRequest(question)
      ? (lang === 'AR'
          ? 'أنتج 5 أسئلة اختيار من متعدد (أ، ب، ج، د) حول الموضوع الحالي. ضع الإجابات بعد ذلك.'
          : 'Produce 5 multiple-choice questions (A–D) about the current topic, then give an answer key.')
      : (lang === 'AR'
          ? 'أجب بشكل واضح وبنقاط موجزة.'
          : 'Answer clearly in concise bullet points.');

    const messages = [
      { role: 'system', content: sys },
      {
        role: 'user',
        content:
`${lang === 'AR' ? 'لغة الإجابة' : 'Answer language'}: ${lang}
${prefsLine}
${topicLine}

${lang === 'AR' ? 'السياق' : 'Context'}:
${context || (lang === 'AR' ? '(لا سياق متاح)' : '(no context available)')}

${lang === 'AR' ? 'السؤال' : 'Question'}: ${question}

${mcqHint}
`
      }
    ];

    const answer = await chat(messages);

    // Save a small rolling history (optional)
    s.history.push({ q: question, a: answer });
    if (s.history.length > 8) s.history.shift();

    // Return sources
    const sources = matches.map(m => ({
      id: m.id,
      score: m.score,
      file: m.metadata?.file || m.id,
      lang: m.metadata?.lang,
      stage: m.metadata?.stage,
      subject: m.metadata?.subject
    }));

    res.json({ answer, sources });
  } catch (err) {
    const detail = err?.response?.data ? JSON.stringify(err.response.data, null, 2) : String(err?.message || err);
    console.error('Query failed:', detail);
    res.status(500).json({ error: 'Query failed', detail });
  }
});

// ----------------------------
// Start
// ----------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`API listening on :${PORT}`);
});
