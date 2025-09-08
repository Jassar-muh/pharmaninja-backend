// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { Pinecone } from '@pinecone-database/pinecone';

// ---------------------------
// App & Middleware
// ---------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ---------------------------
// Pinecone setup
// ---------------------------
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX);

// ---------------------------
// Helper: system prompt
// ---------------------------
function systemPrompt(lang = 'EN') {
  const L = lang === 'AR' ? 'Arabic' : 'English';
  return `You are PharmaNinja, a concise exam-focused pharmacy tutor.
- Always answer in ${L}.
- Be structured: headings, bullets, **bold key terms**.
- Ground answers ONLY in the provided context; if missing, ask ONE short clarifying question.
- If the user says "more", "continue", or "MCQs", stay on the SAME topic.
- End with a 1–3 line takeaway.`;
}

// ---------------------------
// Simple in-memory session store
// (resets when server restarts)
// ---------------------------
const SESSIONS = new Map();
function getSession(id, lang) {
  if (!SESSIONS.has(id)) {
    SESSIONS.set(id, {
      lang: lang || 'EN',
      history: [{ role: 'system', content: systemPrompt(lang || 'EN') }],
      examplesAdded: false
    });
  }
  return SESSIONS.get(id);
}

// ---------------------------
// OpenAI Embeddings
// ---------------------------
async function embed(text) {
  const r = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: 'text-embedding-3-small' }, // 1536-dim
    { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding;
}

// ---------------------------
// Health Endpoints
// ---------------------------
app.get('/ping', (_req, res) => res.type('text/plain').send('pong'));

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    uptime: process.uptime(),
    ts: Date.now()
  });
});

app.get('/selftest', async (_req, res) => {
  const env = {
    OPENAI_API_KEY: !!process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: !!process.env.PINECONE_API_KEY,
    PINECONE_INDEX: process.env.PINECONE_INDEX || null
  };
  try {
    const vec = await embed('selftest');
    const dim = vec.length || 0;
    // quick pinecone ping (topK=1)
    const pine = await index.query({ vector: vec, topK: 1, includeMetadata: false });
    return res.json({ env, openai: { ok: true, dim }, pinecone: { ok: true, matches: (pine.matches || []).length } });
  } catch (e) {
    return res.status(500).json({ env, error: 'Embedding or Pinecone failed', detail: e?.response?.data || e?.message });
  }
});

// ---------------------------
// Main Query Endpoint
// ---------------------------
app.post('/query', async (req, res) => {
  try {
    const { sessionId, lang, stage, subject, question } = req.body || {};

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId' });
    }
    if (!question || !question.trim()) {
      return res.json({ answer: (lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.'), sources: [] });
    }

    // --- session + language ---
    const sess = getSession(sessionId, lang);
    if (lang && lang !== sess.lang) {
      sess.lang = lang;
      if (sess.history.length) sess.history[0].content = systemPrompt(lang);
    }

    // --- simple language switch via message ---
    const qRaw = question.trim();
    const qNorm = qRaw.toLowerCase();
    if (/^\s*arabic\s*$/.test(qNorm)) {
      sess.lang = 'AR';
      sess.history[0].content = systemPrompt('AR');
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
    }
    if (/^\s*english\s*$/.test(qNorm)) {
      sess.lang = 'EN';
      sess.history[0].content = systemPrompt('EN');
      return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
    }

    // --- embed the question ---
    const qVec = await embed(qRaw);

    // --- Pinecone retrieval with filters & gating ---
    const pine = await index.query({
      vector: qVec,
      topK: 12,
      includeMetadata: true,
      filter: { ...(stage && { stage }), ...(subject && { subject }) }
    });

    const MATCH_THRESHOLD = 0.75;
    let matches = (pine.matches || []).filter(m => (m.score ?? 0) >= MATCH_THRESHOLD);
    if (matches.length < 3) matches = (pine.matches || []).slice(0, 5);

    // diversity by file/page
    const seen = new Set();
    const picked = [];
    for (const m of matches) {
      const key = `${m.metadata?.file || ''}:${m.metadata?.page || ''}`;
      if (!seen.has(key)) { seen.add(key); picked.push(m); }
      if (picked.length >= 5) break;
    }
    const context = picked.map((m, i) => `[#${i + 1}] ${m.metadata?.text || ''}`).join('\n');

    // --- intent: mcq / continue / answer ---
    let mode = 'answer';
    if (/^(mcq|mcqs|more mcq|mcqs please|اسئلة|أسئلة)$/i.test(qNorm)) mode = 'mcq';
    else if (/^(more|continue|تابع|كمل)$/i.test(qNorm)) mode = 'continue';

    const userMsg =
      mode === 'mcq'
        ? (sess.lang === 'AR'
            ? 'أنشئ 6 أسئلة MCQ عن نفس الموضوع، كل سؤال بخيارات A–E ومفتاح الإجابة في النهاية.'
            : 'Generate 6 MCQs about the SAME topic, each with choices A–E and an answer key at the end.')
        : mode === 'continue'
          ? (sess.lang === 'AR'
              ? 'تابع في نفس الموضوع بنقاط جديدة مختصرة.'
              : 'Continue the SAME topic with new concise key points.')
          : qRaw;

    // --- add one tiny example (first time only) ---
    if (!sess.examplesAdded) {
      sess.history.push(
        { role: 'user', content: 'Explain beta-lactams' },
        { role: 'assistant', content:
`**Mechanism (fast):**
- Inhibit PBPs → ↓ transpeptidation → weak peptidoglycan → lysis
**Resistance:** beta-lactamases, altered PBPs
**Takeaway:** Cell wall synthesis blockers; watch for MRSA/ESBL.` }
      );
      sess.examplesAdded = true;
    }

    // --- style nudges & instructions ---
    const styleNudge = (sess.lang === 'AR')
      ? { role: 'system', content: 'اكتب بعناوين قصيرة ونقاط واضحة وخلاصة في النهاية.' }
      : { role: 'system', content: 'Use short headings, crisp bullets, and a one-line takeaway.' };

    const instructions = (sess.lang === 'AR')
      ? 'استخدم السياق للإجابة. لا تخترع. إن كان السياق ضعيفًا، اسأل سؤالًا توضيحيًا واحدًا.'
      : 'Use the context to answer. Do not invent. If context is weak, ask ONE clarifying question.';

    const finalUser = `Context:\n${context}\n\nQuestion: ${userMsg}`;

    // --- build messages & call OpenAI ---
    sess.history[0].content = systemPrompt(sess.lang);
    sess.history.push(styleNudge);
    sess.history.push({ role: 'system', content: instructions });
    sess.history.push({ role: 'user', content: finalUser });

    const comp = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-4o-mini',
        temperature: 0.3,
        top_p: 0.9,
        messages: sess.history
      },
      { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } }
    );

    const answer = comp.data.choices?.[0]?.message?.content?.trim()
      || (sess.lang === 'AR' ? 'لم أجد إجابة.' : 'No answer.');
    sess.history.push({ role: 'assistant', content: answer });

    // --- trim history to keep memory small ---
    while (sess.history.length > 1 + 2 * 12) { // system + 12 exchanges
      sess.history.splice(1, 1);
    }

    // --- neat sources back to Botpress ---
    const neatSources = picked.map((m, i) => ({
      id: m.id,
      file: m.metadata?.file,
      page: m.metadata?.page,
      score: m.score
    }));

    return res.json({ answer, sources: neatSources });
  } catch (e) {
    console.error('Query error:', e?.response?.data || e?.message);
    return res.status(500).json({ error: 'Query failed' });
  }
});

// ---------------------------
// Start Server
// ---------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ API running on :${PORT}`);
});
