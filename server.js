// server.js  — PharmaNinja backend (clean build)
// Requirements in package.json: express, axios, cors, dotenv, @pinecone-database/pinecone

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// ---------- ENV & CLIENTS ----------
const {
  OPENAI_API_KEY,
  PINECONE_API_KEY,
  PINECONE_INDEX,
  PORT = 3000,
} = process.env;

if (!OPENAI_API_KEY) console.warn('⚠️ Missing OPENAI_API_KEY');
if (!PINECONE_API_KEY) console.warn('⚠️ Missing PINECONE_API_KEY');
if (!PINECONE_INDEX) console.warn('⚠️ Missing PINECONE_INDEX');

const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX);

// ---------- SIMPLE SESSION MEMORY ----------
/** sessions: Map<sessionId, { lang, stage, subject, lastTopic, history: Array<{role,content}> }> */
const sessions = globalThis._pharmaSessions || (globalThis._pharmaSessions = new Map());

// ---------- HELPERS ----------
const EN_EMBED_MODEL = 'text-embedding-3-small'; // 1536-dim

function norm(s = '') { return s.trim().toLowerCase(); }

function isMcqAsk(q = '') {
  const t = norm(q);
  return (
    /mcq/.test(t) ||
    /multiple\s*choice/.test(t) ||
    /اختيار/.test(t) || /متعدد/.test(t) || /اسئلة|أسئلة/.test(t)
  );
}

function detectTopicFromQuestion(q = '', fallback = '') {
  // crude heuristic: if question contains a noun phrase after verbs like explain/define
  const m =
    q.match(/(?:about|on|regarding|explain|define)\s+(.+?)[.?!]*$/i) ||
    q.match(/اشرح(?:\s*بإيجاز)?\s+(.+?)$/i);
  const candidate = (m && m[1]) ? m[1] : q;
  return (candidate || fallback || '').slice(0, 120);
}

async function embed(text) {
  const r = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: EN_EMBED_MODEL },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding; // 1536-length
}

async function chat({ lang, system, user, context }) {
  const messages = [
    { role: 'system', content: system },
  ];

  // add small, trimmed context as a tool-like note
  if (context && context.trim()) {
    messages.push({
      role: 'system',
      content:
        (lang === 'AR')
          ? `مقتطفات مرجعية للاستئناس فقط (قد تتضمن نصًا مقتبسًا):\n${context.slice(0, 8000)}`
          : `Reference snippets for grounding (use only if helpful):\n${context.slice(0, 8000)}`
    });
  }

  messages.push({ role: 'user', content: user });

  const r = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    {
      model: 'gpt-4o-mini',
      temperature: 0.2,
      messages,
    },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );

  return r.data.choices?.[0]?.message?.content?.trim() || '';
}

function formatSources(matches = [], max = 3) {
  return (matches || [])
    .slice(0, max)
    .map((m, i) => {
      const id = m.id || '';
      const file = m?.metadata?.file || '';
      return `[#${i + 1}] ${file || id}`;
    })
    .join('\n');
}

// ---------- ROUTES (health) ----------
app.get('/ping', (_req, res) => res.send('pong'));

app.get('/health', async (_req, res) => {
  try {
    const vec = await embed('health probe');
    const pine = await index.query({ vector: vec, topK: 1, includeMetadata: true });
    res.json({
      ok: true,
      env: {
        OPENAI_API_KEY: !!OPENAI_API_KEY,
        PINECONE_API_KEY: !!PINECONE_API_KEY,
        PINECONE_INDEX,
      },
      openai: { dim: vec?.length || 0 },
      pinecone: { ok: true, matches: pine?.matches?.length || 0 }
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.response?.data || e?.message) });
  }
});

// ---------- MAIN: /query ----------
app.post('/query', async (req, res) => {
  try {
    let { sessionId, lang, stage, subject, question } = req.body || {};

    if (!sessionId) sessionId = `anon:${req.ip}:${Date.now()}`;
    const s = sessions.get(sessionId) || { history: [] };

    // language instant switch
    const qRaw = (question || '').trim();
    const qNorm = norm(qRaw);
    if (/^(arabic|عربي|العربية)$/.test(qNorm)) {
      s.lang = 'AR';
      sessions.set(sessionId, s);
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
    }
    if (/^(english|انجليزي|الانجليزية)$/.test(qNorm)) {
      s.lang = 'EN';
      sessions.set(sessionId, s);
      return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
    }

    // carry forward prefs if not provided
    s.lang = lang || s.lang || 'EN';
    s.stage = stage || s.stage;
    s.subject = subject || s.subject;

    // “MCQs please” with no topic → use lastTopic/subject
    if (isMcqAsk(qRaw) && qRaw.length < 24) {
      const topic = s.lastTopic || s.subject || 'the last discussed topic';
      question = (s.lang === 'AR')
        ? `اعطني 5 أسئلة اختيار من متعدد (مع الإجابات والتفسير المختصر) عن: ${topic}`
        : `Give me 5 multiple-choice questions (with answers & brief rationales) about: ${topic}`;
    } else if (!qRaw) {
      return res.json({
        answer: (s.lang === 'AR') ? 'اكتب سؤالك.' : 'Ask your question.',
        sources: []
      });
    } else {
      // update lastTopic from question text
      s.lastTopic = detectTopicFromQuestion(qRaw, s.subject);
    }

    // embed & retrieve
    const qVec = await embed(question);
    const filter = {
      ...(s.stage ? { stage: s.stage } : {}),
      ...(s.subject ? { subject: s.subject } : {}),
    };
    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      filter
    });

    const matches = pine?.matches || [];
    const context = matches.map((m, i) => `[#${i + 1}] ${m?.metadata?.text || ''}`).join('\n');

    // system prompt (tutor personality)
    const system =
      (s.lang === 'AR')
        ? 'أنت مُدرّس صيدلة ذكي ومختصر. جاوب بإسلوب امتحاني منظم، واستخدم النقاط والجداول المختصرة عندما يناسب.'
        : 'You are a sharp, exam-focused pharmacy tutor. Be clear, concise, and structured. Use lists/tables when helpful.';

    // build user message
    const userMsg =
      (s.lang === 'AR')
        ? `المرحلة: ${s.stage || '-'}\nالمادة: ${s.subject || '-'}\nالسؤال: ${qRaw}\n`
        : `Stage: ${s.stage || '-'}\nSubject: ${s.subject || '-'}\nQuestion: ${qRaw}\n`;

    // call LLM
    const answer = await chat({ lang: s.lang, system, user: userMsg, context });

    // store minimal history (last 6 turns) — keeps it light
    s.history.push({ role: 'user', content: qRaw });
    s.history.push({ role: 'assistant', content: answer });
    if (s.history.length > 12) s.history.splice(0, s.history.length - 12);

    sessions.set(sessionId, s);

    res.json({
      answer,
      sources: (matches || []).map(m => ({
        id: m.id,
        file: m?.metadata?.file,
        lang: m?.metadata?.lang,
        stage: m?.metadata?.stage,
        subject: m?.metadata?.subject,
        score: m?.score
      }))
    });
  } catch (e) {
    console.error('❌ /query error:', e?.response?.data || e);
    res.status(500).json({ error: 'Query failed' });
  }
});

// ---------- START ----------
app.listen(PORT, () => {
  console.log(`✅ API running on http://localhost:${PORT}`);
});
