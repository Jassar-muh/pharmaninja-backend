// server.js
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const app = express();
app.use(express.json({ limit: "2mb" }))
// --- diagnostics ---
app.get("/ping", (req, res) => res.send("pong"));

app.get("/health", (req, res) =>
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() })
);

// Runs minimal checks for env, OpenAI embeddings, and Pinecone index query
app.get("/selftest", async (req, res) => {
  try {
    const haveOpenAI = !!process.env.OPENAI_API_KEY;
    const havePCKey  = !!process.env.PINECONE_API_KEY;
    const idxName    = process.env.PINECONE_INDEX || "(missing)";

    // 1) tiny embedding
    let embOK = false, embLen = 0;
    try {
      const r = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({ model: "text-embedding-3-small", input: "hello" })
      });
      if (r.ok) {
        const j = await r.json();
        embLen = j?.data?.[0]?.embedding?.length || 0;
        embOK = embLen > 0;
      }
// server.js
import 'dotenv/config';
import express from 'express';
import { Pinecone } from '@pinecone-database/pinecone';

// ====== CONFIG ======
const PORT = process.env.PORT || 3000;
const OPENAI_KEY = process.env.OPENAI_API_KEY || '';
const PC_KEY = process.env.PINECONE_API_KEY || '';
const PC_INDEX = process.env.PINECONE_INDEX || '';   // e.g. "pharmaninja-bot-1536"
const EMBEDDING_MODEL = 'text-embedding-3-small';     // 1536-dim

// ====== APP SETUP ======
const app = express();
app.use(express.json({ limit: '2mb' }));

// simple CORS (Botpress/Telegram safe)
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Origin, X-Requested-With, Content-Type, Accept, Authorization'
  );
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// ====== PINECONE CLIENT ======
const pc = new Pinecone({ apiKey: PC_KEY });
const index = PC_INDEX ? pc.index(PC_INDEX) : null;

// ====== HELPERS ======
async function embed(text) {
  // Uses native fetch; Node 18+ is fine
  const r = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENAI_KEY}`
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text })
  });

  if (!r.ok) {
    const errTxt = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings error ${r.status}: ${errTxt}`);
  }
  const j = await r.json();
  return j?.data?.[0]?.embedding || [];
}

async function genAnswer({ lang, stage, subject, question, context }) {
  // Try to get a nice answer; if it fails (e.g., billing), fall back.
  try {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${OPENAI_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a helpful pharmacy study tutor. Cite which snippets you used if possible.' },
          {
            role: 'user',
            content:
`Language: ${lang || 'EN'}
Stage: ${stage || '-'}
Subject: ${subject || '-'}
Question: ${question}

Context (top matches):
${context}

Answer in ${lang || 'EN'}. If unknown, say so briefly.`
          }
        ],
        temperature: 0.2
      })
    });

    if (!r.ok) {
      const txt = await r.text().catch(() => '');
      throw new Error(`OpenAI chat error ${r.status}: ${txt}`);
    }
    const j = await r.json();
    return j?.choices?.[0]?.message?.content?.trim() || '';
  } catch (e) {
    // Fallback to context only
    const fallback = context
      ? `(${lang || 'EN'}) Answer based on retrieved context:\n\n${context}`
      : `(${lang || 'EN'}) No relevant context found.`;
    return fallback;
  }
}

function cleanFilter({ stage, subject }) {
  const f = {};
  if (stage) f.stage = stage;
  if (subject) f.subject = subject;
  return Object.keys(f).length ? f : undefined;
}

function mapMatches(matches = []) {
  return matches.map(m => ({
    id: m?.id,
    score: m?.score,
    file: m?.metadata?.file,
    lang: m?.metadata?.lang,
    stage: m?.metadata?.stage,
    subject: m?.metadata?.subject
  }));
}

// ====== HEALTH ENDPOINTS ======
app.get('/', (req, res) => res.json({ ok: true, service: 'pharmaninja-backend' }));

app.get('/ping', (req, res) => res.send('pong'));

app.get('/health', (req, res) =>
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() })
);

// quick check: env + tiny embedding + pinecone touch
app.get('/selftest', async (req, res) => {
  try {
    const env = {
      OPENAI_API_KEY: !!OPENAI_KEY,
      PINECONE_API_KEY: !!PC_KEY,
      PINECONE_INDEX: PC_INDEX || '(missing)'
    };

    // embeddings probe
    let embOK = false, embLen = 0, embMsg = '';
    try {
      const v = await embed('hello');
      embLen = Array.isArray(v) ? v.length : 0;
      embOK = embLen > 0;
    } catch (e) {
      embMsg = e?.message || String(e);
    }

    // pinecone probe
    let pcOK = false, pcMsg = '';
    try {
      if (!index) throw new Error('Index not configured (PINECONE_INDEX missing)');
      const q = await index.query({ vector: new Array(1536).fill(0), topK: 1 });
      pcOK = true;
      pcMsg = `topK=${q?.matches?.length ?? 0}`;
    } catch (e) {
      pcMsg = e?.message || String(e);
    }

    res.json({ env, openai: { ok: embOK, dim: embLen, note: embMsg }, pinecone: { ok: pcOK, note: pcMsg } });
  } catch (e) {
    res.status(500).json({ error: e?.message || String(e) });
  }
});

// ====== MAIN QUERY ======
app.post('/query', async (req, res) => {
  const started = Date.now();
  const { lang = 'EN', stage, subject, question } = req.body || {};
  console.log('POST /query', { lang, stage, subject, qLen: (question || '').length });

  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ error: 'Missing question' });
  }
  if (!OPENAI_KEY) return res.status(500).json({ error: 'Server missing OPENAI_API_KEY' });
  if (!PC_KEY)     return res.status(500).json({ error: 'Server missing PINECONE_API_KEY' });
  if (!PC_INDEX)   return res.status(500).json({ error: 'Server missing PINECONE_INDEX' });

  try {
    // 1) Embed the question
    const qVec = await embed(question);

    // 2) Query Pinecone
    const filter = cleanFilter({ stage, subject });
    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      ...(filter ? { filter } : {})
    });

    const matches = pine?.matches || [];
    const context = matches
      .map((m, i) => `[#${i + 1}] ${m?.metadata?.text || ''}`)
      .join('\n')
      .trim();

    // 3) Try to generate an answer (fallback to context if chat fails)
    const answer = await genAnswer({ lang, stage, subject, question, context });

    // 4) Respond
    const payload = {
      answer,
      sources: mapMatches(matches),
      took_ms: Date.now() - started
    };
    console.log('OK /query took', payload.took_ms, 'ms, matches:', matches.length);
    return res.json(payload);
  } catch (e) {
    const detail = e?.response?.data || e?.message || String(e);
    console.error('QUERY ERROR:', detail);
    return res.status(500).json({ error: 'Query failed', detail });
  }
});

// ====== START SERVER ======
app.listen(PORT, () => {
  console.log(`✅ API running on port ${PORT}`);
});
