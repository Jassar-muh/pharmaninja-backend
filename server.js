// server.js  — conversational memory + AR/EN + Pinecone RAG
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ---- CONFIG ----
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'pharmaninja-bot-1536';
const EMBEDDING_MODEL = 'text-embedding-3-small'; // 1536-dim
const CHAT_MODEL = 'gpt-4o-mini';

// ---- CLIENTS ----
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX);

// ---- SIMPLE IN-MEMORY STORE (per sessionId) ----
// structure: sessions.set(sessionId, {lang, stage, subject, topic, history:[{role,content}]})
const sessions = new Map();
function getSession(sessionId) {
  if (!sessions.has(sessionId)) sessions.set(sessionId, { lang: 'EN', history: [] });
  return sessions.get(sessionId);
}
function pushHistory(session, role, content) {
  session.history.push({ role, content });
  // keep last ~10 messages max (5 turns)
  if (session.history.length > 10) session.history.splice(0, session.history.length - 10);
}

// ---- HELPERS ----
async function embed(text) {
  const r = await axios.post('https://api.openai.com/v1/embeddings',
    { input: text, model: EMBEDDING_MODEL },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding;
}

async function retrieveContext(question, stage, subject) {
  const qVec = await embed(question);
  const filter = {
    ...(stage ? { stage } : {}),
    ...(subject ? { subject } : {})
  };
  const res = await index.query({
    vector: qVec,
    topK: 5,
    filter,
    includeMetadata: true
  });
  const matches = res.matches || [];
  const contextText = matches
    .map((m, i) => `[#${i + 1}] ${m.metadata?.text || ''}`)
    .join('\n');
  return { contextText, sources: matches.map(m => ({
    id: m.id, score: m.score, file: m.metadata?.file, lang: m.metadata?.lang,
    stage: m.metadata?.stage, subject: m.metadata?.subject
  })) };
}

function buildSystem(lang) {
  const base = `You are PharmaNinja, a concise, friendly pharmacy tutor. 
- Prefer bullet points. Be exam-oriented. 
- Cite up to 3 short sources as [#1], [#2]... if useful. 
- If user asks for MCQs, generate 5 high-quality MCQs with answers at the end.`;
  return lang === 'AR'
    ? base + ` Answer fully in Arabic.`
    : base + ` Answer fully in English.`;
}

// Very small heuristic to set/keep a "topic" for follow-ups like "more", "MCQs", "compare", etc.
function updateTopic(session, question, topSourceSubject) {
  const q = (question || '').trim().toLowerCase();
  const isFollowup = /^(more|mcq|mcqs|another|continue|compare|examples?|next|تابع|المزيد|اسئلة|أسئلة)/i.test(q);
  if (!isFollowup && q.length > 6) {
    // make this new question the topic (shorten to 10 words)
    session.topic = question.split(/\s+/).slice(0, 10).join(' ');
  } else if (!session.topic && topSourceSubject) {
    session.topic = topSourceSubject;
  }
  return session.topic;
}

// ---- ROUTES ----
app.get('/ping', (req, res) => res.send('pong'));
app.get('/health', (req, res) => res.json({ ok: true, uptime: process.uptime(), ts: Date.now() }));
app.get('/selftest', async (req, res) => {
  const checks = {
    env: {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX: PINECONE_INDEX
    }
  };
  try {
    const v = await embed('hello'); // test key + model dim
    checks.openai = { ok: true, dim: v.length };
  } catch (e) {
    checks.openai = { ok: false, note: (e.response?.data && JSON.stringify(e.response.data, null, 2)) || e.message };
  }
  try {
    const resQ = await index.query({ vector: Array(1536).fill(0), topK: 1, includeMetadata: false });
    checks.pinecone = { ok: true, note: `topK=${resQ.topK || 1}` };
  } catch (e) {
    checks.pinecone = { ok: false, note: e.message };
  }
  res.json(checks);
});

// ---- MAIN QA ENDPOINT with memory ----
app.post('/query', async (req, res) => {
  try {
    const { sessionId, lang, stage, subject, question } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });
    if (!question || !question.trim()) return res.json({ answer: (lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.'), sources: [] });

    // get session and update prefs
    const session = getSession(sessionId);
    if (lang) session.lang = lang;
    if (stage) session.stage = stage;
    if (subject) session.subject = subject;

    // special keywords to flip language
    const qLower = question.trim().toLowerCase();
    if (qLower === 'arabic') {
      session.lang = 'AR';
      pushHistory(session, 'user', question);
      pushHistory(session, 'assistant', 'تم! اكتب سؤالك الآن بالعربية.');
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
    }
    if (qLower === 'english') {
      session.lang = 'EN';
      pushHistory(session, 'user', question);
      pushHistory(session, 'assistant', 'Done! Ask your question in English.');
      return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
    }

    // retrieve RAG context (filtered by stage/subject when provided)
    const { contextText, sources } = await retrieveContext(question, session.stage, session.subject);

    // set/keep topic to help follow-ups (“MCQs please” etc.)
    const topSubj = sources?.[0]?.subject;
    const topic = updateTopic(session, question, topSubj);

    // build chat history: system + last 5 turns + current user + context
    const systemMsg = { role: 'system', content: buildSystem(session.lang) };
    const historyMsgs = (session.history || []).slice(-8); // last 4 turns
    const contextMsg = {
      role: 'system',
      content:
        (session.lang === 'AR'
          ? `سياق تم استرجاعه من ملفاتك (استخدمه إن كان مناسبًا):\n${contextText || '(لا يوجد)'}`
          : `Retrieved study context (use if relevant):\n${contextText || '(none)'}`) +
        (topic ? (session.lang === 'AR' ? `\n\nالموضوع الجاري: ${topic}` : `\n\nCurrent topic: ${topic}`) : '')
    };
    const userMsg = { role: 'user', content: question };

    const messages = [systemMsg, contextMsg, ...historyMsgs, userMsg];

    // call OpenAI chat
    const chat = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: CHAT_MODEL,
        messages,
        temperature: 0.3
      },
      { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
    );

    const answer = chat.data.choices?.[0]?.message?.content?.trim() || (session.lang === 'AR' ? 'عذرًا، لم أجد إجابة.' : 'Sorry, I could not find an answer.');

    // update memory
    pushHistory(session, 'user', question);
    pushHistory(session, 'assistant', answer);

    // return
    res.json({ answer, sources: (sources || []).slice(0, 3) });
  } catch (e) {
    console.error(e?.response?.data || e.message);
    res.status(500).json({ error: 'Query failed' });
  }
});

// ---- START ----
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ API running on :${PORT}`));
