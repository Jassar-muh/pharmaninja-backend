// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';
import { Pinecone } from '@pinecone-database/pinecone';

// ---------- ENV ----------
const { OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PORT } = process.env;
if (!OPENAI_API_KEY) console.warn('⚠️ Missing OPENAI_API_KEY');
if (!PINECONE_API_KEY) console.warn('⚠️ Missing PINECONE_API_KEY');
if (!PINECONE_INDEX) console.warn('⚠️ Missing PINECONE_INDEX');

// ---------- APP ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---------- PINECONE ----------
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX);

// ---------- HELPERS ----------
function buildFilter({ stage, subject }) {
  const f = {};
  if (stage) f.stage = stage;
  if (subject) f.subject = subject;
  return Object.keys(f).length ? f : null;
}

async function openaiEmb(text) {
  const r = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small', // 1536-dim
      input: text,
    }),
  });
  if (!r.ok) {
    throw new Error(`OpenAI embeddings ${r.status}: ${await r.text()}`);
  }
  const j = await r.json();
  return j.data[0].embedding; // Float32Array-like (length 1536)
}

async function openaiChat({ lang, question, context }) {
  // Keep very close to your original “Answer in ${lang}. Use this context…” style
  const userPrompt =
`Answer in ${lang || 'EN'}. Use this context if relevant:

${context || '(no matches)'}

Q: ${question}
A:`;

  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            'You are a helpful pharmacy study tutor. Keep answers exam-oriented, concise, and do not hallucinate. If unsure, say so.',
        },
        { role: 'user', content: userPrompt },
      ],
    }),
  });
  if (!r.ok) {
    throw new Error(`OpenAI completion ${r.status}: ${await r.text()}`);
  }
  const j = await r.json();
  return j.choices?.[0]?.message?.content || '(no answer)';
}

// ---------- HEALTH ----------
app.get('/ping', (_req, res) => res.send('pong'));

app.get('/health', async (_req, res) => {
  try {
    // quick light checks
    res.json({ ok: true, uptime: process.uptime(), ts: Date.now() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/selftest', async (_req, res) => {
  try {
    // embed + pinecone round-trip to verify config
    const emb = await openaiEmb('selftest');
    let pineOk = false;
    let matches = 0;
    try {
      const pineRes = await index.query({ vector: emb, topK: 1 });
      pineOk = true;
      matches = pineRes?.matches?.length || 0;
    } catch (e) {
      pineOk = false;
    }
    res.json({
      env: {
        OPENAI_API_KEY: !!OPENAI_API_KEY,
        PINECONE_API_KEY: !!PINECONE_API_KEY,
        PINECONE_INDEX,
      },
      openai: { ok: emb?.length === 1536, dim: emb?.length || 0 },
      pinecone: { ok: pineOk, matches },
    });
  } catch (e) {
    res.status(500).json({ error: 'Embedding failed', detail: e.message });
  }
});

// ---------- MAIN QUERY ----------
app.post('/query', async (req, res) => {
  try {
    const { lang = 'EN', stage, subject, question } = req.body || {};
    if (!question || typeof question !== 'string' || !question.trim()) {
      return res.status(400).json({ error: 'Missing question' });
    }
    if (!OPENAI_API_KEY || !PINECONE_API_KEY || !PINECONE_INDEX) {
      return res.status(500).json({ error: 'Missing env vars' });
    }

    // 1) Embed question
    const qVec = await openaiEmb(question);

    // 2) Pinecone search with optional filter
    const filter = buildFilter({ stage, subject });
    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      ...(filter ? { filter } : {}),
    });

    const matches = pine?.matches || [];
    const context = matches
      .map((m, i) => `[#${i + 1}] ${m?.metadata?.text || ''}`)
      .join('\n');

    // 3) Generate answer
    const answer = await openaiChat({ lang, question, context });

    // 4) Return with compact source list
    return res.json({
      answer,
      sources: matches.map(m => ({
        id: m.id,
        score: m.score,
        file: m?.metadata?.file,
      })),
    });
  } catch (e) {
    console.error('QUERY ERROR:', e.message);
    return res.status(500).json({ error: 'Query failed', detail: e.message });
  }
});

// ---------- START ----------
const port = Number(PORT) || 3000;
app.listen(port, () => console.log(`✅ API running on :${port}`));
