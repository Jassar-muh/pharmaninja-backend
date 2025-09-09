// server.js
// Pharmaninja backend — Express + Pinecone RAG + OpenAI chat
// Features:
// - AR/EN tutor personality
// - Session memory (short rolling history)
// - Topic pinning + guided follow-ups (MCQs + compare with ...)
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
const sessions = Object.create(null);

// ---- Helpers ----
function getSession(id) {
  if (!sessions[id]) sessions[id] = { lang: 'EN', stage: null, subject: null, topic: null, history: [] };
  return sessions[id];
}

function pushHistory(session, role, content) {
  session.history = session.history || [];
  session.history.push({ role, content });
  if (session.history.length > MAX_HISTORY) {
    session.history = session.history.slice(-MAX_HISTORY);
  }
}

function buildSystem(lang = 'EN') {
  if (lang === 'AR') {
    return `أنت مساعد دراسة صيدلي ذكي ومتعاون. استخدم أسلوباً مُنظماً وقصيراً ومباشراً، وركّز على ما يهم للامتحان (آلية العمل، الاستطبابات، الآثار الجانبية، موانع الاستعمال، نقاط تمييز سريعة). 
إن طلب المستخدم "MCQs"، أنشئ أسئلة اختيار من متعدد مع الإجابات المبررة باختصار. 
إن كان الطلب غامضاً، فتابع على **الموضوع الجاري** في الجلسة. 
استخدم السياق المسترجع من الملفات فقط عند الفائدة واذكر المراجع باختصار إذا طُلب ذلك.`;
  }
  return `You are a smart, exam-focused pharmacy tutor. Be concise, structured, and practical (MoA, indications, adverse effects, contraindications, quick pearls).
If the user asks for "MCQs", generate targeted MCQs with brief answer key.
If the request is ambiguous, continue on the **current topic** in this session.
Use retrieved study context when helpful; keep sources brief if asked.`;
}

async function embed(text) {
  const r = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: EMBED_MODEL },
    { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
  );
  return r.data.data[0].embedding;
}

function cleanSources(matches = []) {
  return (matches || []).map(m => ({
    id: m.id,
    score: m.score,
    file: m?.metadata?.file,
    stage: m?.metadata?.stage,
    subject: m?.metadata?.subject,
    lang: m?.metadata?.lang
  }));
}

// ---- Health routes ----
app.get('/ping', (req, res) => res.type('text').send('pong'));
app.get('/health', (req, res) => {
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() });
});
app.get('/selftest', async (req, res) => {
  try {
    const v = await embed('hello');
    const q = await index.query({ vector: v, topK: 1, includeMetadata: true });
    res.json({
      env: {
        OPENAI_API_KEY: !!OPENAI_API_KEY,
        PINECONE_API_KEY: !!PINECONE_API_KEY,
        PINECONE_INDEX: PINECONE_INDEX
      },
      openai: { ok: Array.isArray(v), dim: Array.isArray(v) ? v.length : 0 },
      pinecone: { ok: true, matches: q?.matches?.length ?? 0 }
    });
  } catch (e) {
    res.status(500).json({ error: 'Embedding or Pinecone failed', detail: String(e?.response?.data || e?.message || e) });
  }
});

// ---- Main query ----
app.post('/query', async (req, res) => {
  try {
    const {
      sessionId,
      lang: langIn,
      stage,
      subject,
      question: questionIn
    } = req.body || {};

    if (!sessionId) return res.status(400).json({ error: 'Missing sessionId' });

    // session + prefs
    const session = getSession(sessionId);
    if (langIn)   session.lang = langIn;
    if (stage)    session.stage = stage;
    if (subject)  session.subject = subject;

    let question = (questionIn || '').trim();

    // quick language switch keywords
    if (/^\s*arabic\s*$/i.test(question)) {
      session.lang = 'AR';
      return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
    }
    if (/^\s*english\s*$/i.test(question)) {
      session.lang = 'EN';
      return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
    }

    const lang = session.lang || 'EN';

    // If no real question, nudge
    if (!question) {
      return res.json({
        answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.',
        sources: []
      });
    }

    // ---- RAG: embed + Pinecone search
    let sources = [];
    let contextText = '';

    try {
      const qVec = await embed(question);
      const q = await index.query({
        vector: qVec,
        topK: TOPK,
        filter: {
          ...(session.stage   ? { stage: session.stage }   : {}),
          ...(session.subject ? { subject: session.subject } : {})
        },
        includeMetadata: true
      });

      sources = cleanSources(q.matches || []);
      contextText = (q.matches || [])
        .map((m, i) => `[#${i + 1}] ${(m?.metadata?.text || '').slice(0, 1200)}`)
        .join('\n\n');
    } catch (e) {
      // RAG failed → continue chat without external context
      sources = [];
      contextText = '';
    }

    // ---- FOLLOW-UP PATCH (topic pinning + guided prompts) ----
    const qRaw   = question;
    const qLower = qRaw.toLowerCase();

    // detect short follow-ups
    const isFollowup =
      /^(more|mcq|mcqs|quiz|continue|another|examples?|next|summar(y|ise)|clarify|explain more|تابع|المزيد|اسئلة|أسئلة|اختبار|امثلة)\b/.test(qLower) ||
      qLower.length < 14;

    // If this is a substantive new question, refresh the session topic
    if (!isFollowup && qLower.length > 14) {
      session.topic = qRaw.split(/\s+/).slice(0, 10).join(' ');
    }

    // Bootstrap topic from top match metadata if missing
    const topMeta = sources?.[0] || {};
    if (!session.topic && topMeta?.subject) session.topic = topMeta.subject;

    // Shape follow-ups
    let guidedUserPrompt = qRaw;

    // MCQs follow-up → force to current topic
    if (/(^|\b)(mcq|mcqs|quiz|questions?|اسئلة|أسئلة)(\b|$)/i.test(qLower) && session.topic) {
      guidedUserPrompt =
        (lang === 'AR')
          ? `أنشئ 5 أسئلة اختيار من متعدد مع الإجابات المختصرة في النهاية حول الموضوع التالي فقط: "${session.topic}".`
          : `Create 5 multiple-choice questions (with short answers at the end) strictly about: "${session.topic}".`;
    }

    // "compare with X" → compare WITH that target vs current topic
    if (/compare/.test(qLower) && /with/i.test(qLower) && session.topic) {
      const m = qLower.match(/compare(?:\s+\w+)?\s+with\s+(.+)$/i);
      const target = m ? m[1].trim() : null;
      if (target) {
        guidedUserPrompt =
          (lang === 'AR')
            ? `قارن بشكل مركز بين "${session.topic}" و "${target}" في جدول: آلية العمل، الطيف، الاستطبابات، السمية، المقاومة، ملاحظات سريرية.`
            : `Compare "${session.topic}" WITH "${target}" in a compact table: MoA, spectrum, indications, toxicity, resistance, clinical notes. Keep it exam-oriented.`;
      }
    }

    // Very short/ambiguous follow-up → remind of topic
    if (isFollowup && session.topic && guidedUserPrompt === qRaw) {
      guidedUserPrompt =
        (lang === 'AR')
          ? `تابع على نفس الموضوع: "${session.topic}". المطلوب: ${qRaw}`
          : `Follow up on the SAME topic: "${session.topic}". Request: ${qRaw}`;
    }

    // ---- Build chat messages ----
    const systemMsg = { role: 'system', content: buildSystem(lang) };
    const contextMsg = {
      role: 'system',
      content:
        (lang === 'AR'
          ? `سياق مسترجع من ملفاتك (استخدمه عند الحاجة):\n${contextText || '(لا يوجد)'}`
          : `Retrieved study context (use if relevant):\n${contextText || '(none)'}`) +
        (session.topic
          ? (lang === 'AR'
              ? `\n\nالموضوع الجاري: ${session.topic}`
              : `\n\nCurrent topic: ${session.topic}`)
          : '')
    };

    const historyMsgs = (session.history || []).slice(-MAX_HISTORY);
    const userMsg = { role: 'user', content: guidedUserPrompt };

    const messages = [systemMsg, contextMsg, ...historyMsgs, userMsg];

    // ---- OpenAI chat ----
    const chat = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      { model: CHAT_MODEL, messages, temperature: 0.3 },
      { headers: { Authorization: `Bearer ${OPENAI_API_KEY}` } }
    );

    const answer = chat.data.choices?.[0]?.message?.content?.trim()
      || (lang === 'AR' ? 'عذرًا، لم أجد إجابة.' : 'Sorry, I could not find an answer.');

    // update memory
    pushHistory(session, 'user', qRaw);
    pushHistory(session, 'assistant', answer);

    // respond
    res.json({
      answer,
      sources: (sources || []).slice(0, 5)
    });

  } catch (err) {
    const detail = err?.response?.data || err?.message || String(err);
    res.status(500).json({ error: 'Query failed', detail });
  }
});

// ---- Start server (Render uses PORT) ----
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ API running on :${PORT}`);
});
