// server.js — clean build for Render + Botpress + Pinecone + OpenAI
import 'dotenv/config';
import express from 'express';
import { Pinecone } from '@pinecone-database/pinecone';

// -------- CONFIG --------
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY   = process.env.OPENAI_API_KEY || '';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';
const PINECONE_INDEX   = process.env.PINECONE_INDEX || ''; // e.g. pharmaninja-bot-1536
const EMBED_MODEL = 'text-embedding-3-small';               // 1536-dim

// -------- APP --------
const app = express();
app.use(express.json({ limit: '5mb' }));

// CORS (so Botpress/Telegram can call it)
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// -------- PINECONE --------
let index = null;
if (PINECONE_API_KEY && PINECONE_INDEX) {
  const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
  index = pc.index(PINECONE_INDEX);
}

// -------- HELPERS --------
async function openaiEmb(text) {
  const r = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMBED_MODEL, input: text })
  });
  if (!r.ok) {
    const t = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings ${r.status}: ${t}`);
  }
  const j = await r.json();
  return j?.data?.[0]?.embedding || [];
}

async function openaiAnswer({ lang, stage, subject, question, context }) {
  try {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${OPENAI_API_KEY}` },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        temperature: 0.2,
        messages: [
          { role: 'system', content: 'You are a concise pharmacy tutor. Use retrieved context if relevant.' },
          { role: 'user', content:
`Language: ${lang || 'EN'}
Stage: ${stage || '-'}
Subject: ${subject || '-'}
Question: ${question}

Context:
${context || '(none)'} 

Answer in ${lang || 'EN'}. If unknown, say so.` }
        ]
      })
    });
    if (!r.ok) throw new Error(`OpenAI chat ${r.status}: ${await r.text()}`);
    const j = await r.json();
    return j?.choices?.[0]?.message?.content?.trim() || '';
  } catch {
    // Fallback if chat fails (billing, etc.)
    return context
      ? `(${lang || 'EN'}) Based on retrieved context:\n\n${context}`
      : `(${lang || 'EN'}) No relevant context found.`;
  }
}

function mapSources(matches = []) {
  return matches.map(m => ({
    id: m?.id,
    score: m?.score,
    file: m?.metadata?.file,
    lang: m?.metadata?.lang,
    stage: m?.metadata?.stage,
    subject: m?.metadata?.subject
  }));
}

function buildFilter({ stage, subject }) {
  const f = {};
  if (stage) f.stage = stage;
  if (subject) f.subject = subject;
  return Object.keys(f).length ? f : undefined;
}

// -------- HEALTH --------
app.get('/', (_req, res) => res.json({ ok: true, service: 'pharmaninja-backend' }));
app.get('/ping', (_req, res) => res.send('pong'));
app.get('/health', (_req, res) => res.json({ ok: true, uptime: process.uptime(), ts: Date.now() }));

app.get('/selftest', async (_req, res) => {
  try {
    const env = {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX: PINECONE_INDEX || '(missing)'
    };

    // embeddings probe
    let embOK = false, embDim = 0, embNote = '';
    try {
      const v = await openaiEmb('hello');
      embDim = v.length;
      embOK = embDim > 0;
    } catch (e) { embNote = e?.message || String(e); }

    // pinecone probe
    let pcOK = false, pcNote = '';
    try {
      if (!index) throw new Error('Index not configured');
      const q = await index.query({ vector: new Array(1536).fill(0), topK: 1 });
      pcOK = true; pcNote = `topK=${q?.matches?.length ?? 0}`;
    } catch (e) { pcNote = e?.message || String(e); }

    res.json({ env, openai: { ok: embOK, dim: embDim, note: embNote }, pinecone: { ok: pcOK, note: pcNote } });
  } catch (e) {
    res.status(500).json({ error: e?.message || String(e) });
  }
});

// -------- MAIN: /query --------
app.post('/query', async (req, res) => {
  const t0 = Date.now();
  try {
    const { lang = 'EN', stage, subject, question } = req.body || {};
    if (!question || typeof question !== 'string' || !question.trim()) {
      return res.status(400).json({ error: 'Missing question' });
    }
    if (!OPENAI_API_KEY)   return res.status(500).json({ error: 'Server missing OPENAI_API_KEY' });
    if (!PINECONE_API_KEY) return res.status(500).json({ error: 'Server missing PINECONE_API_KEY' });
    if (!PINECONE_INDEX)   return res.status(500).json({ error: 'Server missing PINECONE_INDEX' });
    if (!index)            return res.status(500).json({ error: 'Pinecone index not initialized' });

    // 1) embed
    const qVec = await openaiEmb(question);

    // 2) query pinecone
    const filter = buildFilter({ stage, subject });
    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      ...(filter ? { filter } : {})
    });

    const matches = pine?.matches || [];
    const context = matches.map((m, i) => `[#${i + 1}] ${m?.metadata?.text || ''}`).join('\n').trim();

    // 3) answer (chat with fallback)
    const answer = await openaiAnswer({ lang, stage, subject, question, context });

    res.json({
      answer,
      sources: mapSources(matches),
      took_ms: Date.now() - t0
    });
  } catch (e) {
    const detail = e?.response?.data || e?.message || String(e);
    console.error('QUERY ERROR:', detail);
    res.status(500).json({ error: 'Query failed', detail });
  }
});

// -------- START --------
app.listen(PORT, () => {
  console.log(`✅ API running on port ${PORT}`);
});
