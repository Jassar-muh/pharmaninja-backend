// server.js
// Pharmaninja backend — Express + Pinecone RAG + OpenAI chat
// Features:
// - AR/EN tutor personality
// - Session memory (short rolling history)
// - Topic pinning + guided follow-ups (MCQs / compare ...)
// - Mid-chat language switch ("arabic", "english")
// - Health routes for Render

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---- Config ----
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;           // e.g. pharmaninja-bot-1536
const EMBED_MODEL     = 'text-embedding-3-small';             // 1536-dim
const CHAT_MODEL      = 'gpt-4o-mini';                        // fast+cheap tutor
const MAX_HISTORY     = 8;                                    // ~4 turns
const TOPK            = 5;

// ---- Pinecone ----
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX);

// ---- In-memory session store ----
/**
 * sessions = {
 *   [sessionId]: {
 *     lang: 'EN'|'AR',
 *     stage: '3rd', subject: 'Pharmacology',
 *     topic: 'beta-lactam antibiotics',
 *     history: [{role:'user'|'assistant', content: '...'}, ...]
 *   }
 * }
 */
const sessions = new Map();

// ---------- Helpers ----------
function getSession(id) {
  let s = sessions.get(id);
  if (!s) {
    s = { lang: 'EN', stage: '', subject: '', topic: null, history: [] };
    sessions.set(id, s);
  }
  return s;
}

function pushHistory(s, role, content) {
  if (!content) return;
  s.history.push({ role, content });
  if (s.history.length > MAX_HISTORY) {
    s.history = s.history.slice(-MAX_HISTORY);
  }
}

function detectIntent(q) {
  const s = (q || '').toLowerCase().trim();
  if (!s) return 'ask';
  if (/^(arabic|english)\s*$/.test(s)) return 'lang_switch';
  if (/(^|\b)(mcq|mcqs|questions|quiz|more|continue|next|practice|اسئلة|أسئلة|تابع)(\b|$)/.test(s)) return 'mcq';
  if (/(^|\b)(summary|summarize|short notes|ملخص|خلاصة)(\b|$)/.test(s)) return 'summary';
  if (/^compare\b| vs | versus |قارن/.test(s)) return 'compare';
  return 'ask';
}

function extractTopicFromAnswer(answer) {
  if (!answer) return null;
  const firstLine = (answer.split('\n').find(l => l.trim()) || '').replace(/^#+\s*/, '').trim();
  return firstLine.slice(0, 140) || null;
}

async function embed(text) {
  const r = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: EMBED_MODEL },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding;
}

async function pineconeSearch(queryText, filters = {}) {
  const vec = await embed(queryText);
  const r = await index.query({
    vector: vec,
    topK: TOPK,
    filter: filters,
    includeMetadata: true
  });
  return r.matches || [];
}

function buildSystemPrompt(lang, stage, subject, intent) {
  if (lang === 'AR') {
    if (intent === 'mcq') {
      return `أنت مُدرّس صيدلة. اكتب 5 أسئلة MCQ قصيرة مع 4 اختيارات وإجابة مشروحة بإيجاز. اجعل المستوى مناسباً للمرحلة ${stage} وبموضوع السؤال الحالي. لا تُخرج أسئلة عامة خارج الموضوع.`;
    }
    if (intent === 'compare') {
      return `أنت مُدرّس صيدلة. قدّم مقارنة منظّمة ومختصرة بين المفاهيم المطلوبة، مع جداول عند الحاجة. حافظ على الدقة والاختصار المناسب لمرحلة ${stage}.`;
    }
    if (intent === 'summary') {
      return `أنت مُدرّس صيدلة. قدّم ملخصاً نقطياً قصيراً ومباشراً للموضوع المطروح، مناسباً للمراجعة قبل الامتحان لمرحلة ${stage}.`;
    }
    return `أنت مُدرّس صيدلة عربي يساعد الطلاب بإجابات مركزة ومناسبة للامتحان دون تطويل، مع ذكر النقاط الأساسية فقط.`;
  } else {
    if (intent === 'mcq') {
      return `You are a pharmacy tutor. Write 5 short MCQs with 4 options and a brief keyed explanation. Keep it on-topic and appropriate for ${stage} stage.`;
    }
    if (intent === 'compare') {
      return `You are a pharmacy tutor. Provide a concise, structured comparison (use a table if helpful). Keep it accurate and focused for ${stage} stage.`;
    }
    if (intent === 'summary') {
      return `You are a pharmacy tutor. Provide a concise bullet summary suitable for last-minute exam review for a ${stage} student.`;
    }
    return `You are a helpful pharmacy study tutor. Give exam-focused, concise answers; list key mechanisms, indications, adverse effects, and contraindications when relevant.`;
  }
}

function buildUserPrompt(lang, stage, subject, question, contextText) {
  const header = (lang === 'AR')
    ? `اللغة: ${lang}\nالمرحلة: ${stage || '-'}\nالمادة: ${subject || '-'}\nالسؤال: ${question}\n\nالسياق من مراجع الطالب:\n${contextText}\n\nأجب باللغة ${lang}. إن كان الجواب غير موجود، قل ذلك بوضوح.`
    : `Language: ${lang}\nStage: ${stage || '-'}\nSubject: ${subject || '-'}\nQuestion: ${question}\n\nContext from the student's materials:\n${contextText}\n\nAnswer in ${lang}. If unknown or insufficient, say so clearly.`;
  return header;
}

async function chat(systemPrompt, history, userPrompt) {
  const messages = [
    { role: 'system', content: systemPrompt },
    ...history.map(h => ({ role: h.role, content: h.content })),
    { role: 'user', content: userPrompt }
  ];
  const r = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    { model: CHAT_MODEL, messages },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );
  return r.data.choices[0].message.content;
}

// ---------- Routes ----------
app.get('/ping', (_req, res) => res.type('text').send('pong'));

app.get('/health', async (_req, res) => {
  try {
    const ok = {
      ok: true,
      uptime: process.uptime(),
      ts: Date.now()
    };
    res.json(ok);
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

app.get('/selftest', async (_req, res) => {
  const out = {
    env: {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX: PINECONE_INDEX || null
    },
    openai: {},
    pinecone: {}
  };
  try {
    const v = await embed('hello');
    out.openai = { ok: true, dim: v.length };
  } catch (e) {
    out.openai = { ok: false, note: `OpenAI embeddings ${e.response?.status || ''}: ${JSON.stringify(e.response?.data || e.message)}` };
  }
  try {
    const m = await pineconeSearch('test', { stage: '3rd' });
    out.pinecone = { ok: true, matches: m.length };
  } catch (e) {
    out.pinecone = { ok: false, note: String(e) };
  }
  res.json(out);
});

app.post('/query', async (req, res) => {
  try {
    const body = req.body || {};
    const sessionId = body.sessionId || `${req.ip}:${(req.headers['user-agent'] || '').slice(0, 40)}`;

    const s = getSession(sessionId);

    // keep latest prefs if provided
    if (body.lang)    s.lang = body.lang;
    if (body.stage)   s.stage = body.stage;
    if (body.subject) s.subject = body.subject;

    const lang   = s.lang || 'EN';
    const stage  = s.stage;
    const subject= s.subject;
    const questionRaw = (body.question || '').trim();

    // intent & language quick switch
    const intent = detectIntent(questionRaw);
    if (intent === 'lang_switch') {
      if (/arabic/i.test(questionRaw))  s.lang = 'AR';
      if (/english/i.test(questionRaw)) s.lang = 'EN';
      return res.json({
        answer: s.lang === 'AR' ? 'تم! اكتب سؤالك الآن بالعربية.' : 'Done! Ask your question in English.',
        sources: []
      });
    }

    // Build retrieval query (pin topic for vague follow-ups)
    let retrievalQuery = questionRaw;
    if (intent === 'mcq' || intent === 'summary' || intent === 'compare' || retrievalQuery.length < 6) {
      if (s.topic) retrievalQuery = `${s.topic} — ${retrievalQuery || 'follow-up'}`;
    }

    // Pinecone retrieval (filter by stage/subject if available)
    const filter = {};
    if (stage)   filter.stage   = stage;
    if (subject) filter.subject = subject;

    const matches = retrievalQuery ? await pineconeSearch(retrievalQuery, filter) : [];
    const contextText = matches
      .map(m => (m.metadata?.text || '').toString().slice(0, 1000))
      .filter(Boolean)
      .join('\n---\n')
      .slice(0, 4000);

    // Build prompts
    const systemPrompt = buildSystemPrompt(lang, stage, subject, intent);
    const userPrompt   = buildUserPrompt(lang, stage, subject, questionRaw || '(no question)', contextText);

    // Chat
    const answer = await chat(systemPrompt, s.history, userPrompt);

    // Save memory
    pushHistory(s, 'user', questionRaw || '(no question)');
    pushHistory(s, 'assistant', answer);

    // Try to set a topic if we don't have one yet and the question looks substantial
    if (!s.topic && questionRaw && questionRaw.length > 8) {
      const guess = extractTopicFromAnswer(answer) || questionRaw;
      if (guess) s.topic = guess;
    }

    // sources
    const sources = (matches || []).slice(0, TOPK).map(m => ({
      id: m.id,
      score: m.score,
      file: m.metadata?.file,
      lang: m.metadata?.lang,
      stage: m.metadata?.stage,
      subject: m.metadata?.subject
    }));

    return res.json({ answer, sources });
  } catch (e) {
    const status = e.response?.status || 500;
    const detail = e.response?.data ? JSON.stringify(e.response.data) : e.message;
    return res.status(200).json({ error: 'Query failed', detail });
  }
});

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ API running on :${PORT}`);
});
