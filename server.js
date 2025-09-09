// server.js
// Pharmaninja backend — conversational memory + AR/EN + topic pin + guided followups
// PASTE THIS WHOLE FILE to replace your current server.js

import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import fetch from 'node-fetch';
import { Pinecone } from '@pinecone-database/pinecone';

/* ---------- ENV ---------- */
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'pharmaninja-bot-1536';
if (!OPENAI_API_KEY) throw new Error('Missing OPENAI_API_KEY');
if (!PINECONE_API_KEY) throw new Error('Missing PINECONE_API_KEY');

/* ---------- OPENAI (HTTP) ---------- */
const OPENAI_URL = 'https://api.openai.com/v1';

/* choose your models */
const EMBEDDING_MODEL = 'text-embedding-3-small'; // 1536 dim
const CHAT_MODEL = 'gpt-4o-mini';

/* ---------- PINECONE ---------- */
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX);

/* ---------- EXPRESS ---------- */
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '2mb' }));

/* ---------- IN-MEMORY SESSIONS ----------
   s = {
     lang: 'EN'|'AR',
     stage: '3rd'|...,
     subject: 'Pharmacology'|...,
     topic: 'pinned topic for followups',
     history: [{q,a,timestamp}]
   }
*/
const sessions = new Map();

/* ---------- UTILS ---------- */
const sleep = ms => new Promise(r => setTimeout(r, ms));

function getSession(id) {
  if (!sessions.has(id)) sessions.set(id, { lang: 'EN', history: [] });
  return sessions.get(id);
}
function trimStr(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}
function extractCompareTarget(q, lang) {
  if (!q) return null;
  const t = q.trim().toLowerCase();
  if (lang === 'AR') {
    const m = t.match(/قارن(?:\s+)?(?:مع|بين)\s+(.+)/);
    return m ? m[1].trim() : null;
  } else {
    const m = t.match(/^compare\s+(?:with|vs\.?)\s+(.+)/);
    return m ? m[1].trim() : null;
  }
}
function normalizeLang(l) {
  if (!l) return null;
  const t = (l || '').toUpperCase();
  if (t.startsWith('AR')) return 'AR';
  if (t.startsWith('EN')) return 'EN';
  return null;
}

/* ---------- OPENAI HELPERS ---------- */
async function embed(text) {
  const r = await fetch(`${OPENAI_URL}/embeddings`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: text,
      model: EMBEDDING_MODEL
    })
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI embeddings ${r.status}: ${txt}`);
  }
  const j = await r.json();
  return j.data[0].embedding;
}

async function chat(messages) {
  const r = await fetch(`${OPENAI_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages,
      temperature: 0.2
    })
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`OpenAI chat ${r.status}: ${txt}`);
  }
  const j = await r.json();
  return j.choices?.[0]?.message?.content?.trim() || '';
}

/* ---------- HEALTH ---------- */
app.get('/ping', (_req, res) => res.type('text/plain').send('pong'));
app.get('/health', (_req, res) => {
  res.json({ ok: true, uptime: process.uptime() * 1000, ts: Date.now() });
});
app.get('/selftest', async (_req, res) => {
  try {
    // quick ping: openai embed, pinecone describe
    let openaiOk = false;
    let dim = null;
    try {
      const v = await embed('hello');
      openaiOk = Array.isArray(v);
      dim = v?.length || null;
    } catch {}

    let pineOk = false;
    let matches = 0;
    try {
      const q = await embed('test');
      const pine = await index.query({ vector: q, topK: 1, includeMetadata: false });
      pineOk = true;
      matches = pine.matches?.length || 0;
    } catch {}

    res.json({
      env: {
        OPENAI_API_KEY: !!OPENAI_API_KEY,
        PINECONE_API_KEY: !!PINECONE_API_KEY,
        PINECONE_INDEX
      },
      openai: { ok: openaiOk, dim },
      pinecone: { ok: pineOk, matches }
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

/* ---------- QUERY ---------- */
app.post('/query', async (req, res) => {
  const { sessionId, lang: bodyLang, stage, subject } = req.body || {};
  let { question } = req.body || {};

  if (!sessionId) return res.status(400).json({ error: 'Missing sessionId' });

  // session & prefs
  const s = getSession(sessionId);
  // allow explicit override from body
  if (normalizeLang(bodyLang)) s.lang = normalizeLang(bodyLang);
  if (stage) s.stage = stage;
  if (subject) s.subject = subject;

  // allow single-word language switches via question
  const qRaw = (question || '').trim();
  if (/^\s*arabic\s*$/i.test(qRaw)) {
    s.lang = 'AR';
    return res.json({ answer: 'تم! اكتب سؤالك الآن بالعربية.', sources: [] });
  }
  if (/^\s*english\s*$/i.test(qRaw)) {
    s.lang = 'EN';
    return res.json({ answer: 'Done! Ask your question in English.', sources: [] });
  }

  const lang = s.lang || 'EN';
  const prefsLine =
    (lang === 'AR')
      ? `المرحلة: ${s.stage || '-'} | المادة: ${s.subject || '-'}`
      : `Stage: ${s.stage || '-'} | Subject: ${s.subject || '-'}`;

  // pin a topic softly from the first meaningful question (no commands like "MCQs please")
  if (!s.topic) {
    const qForTopic = qRaw.toLowerCase();
    if (qForTopic && !/^mcqs?(\s|$)/i.test(qForTopic) && !/^قارن/.test(qForTopic)) {
      // simple topic pin from the user’s wording
      s.topic = trimStr(qRaw.replace(/^(explain|اشرح)\s*/i, ''), 80);
    }
  }
  const topicLine = s.topic
    ? (lang === 'AR' ? `الموضوع المثبّت: ${s.topic}` : `Pinned topic: ${s.topic}`)
    : (lang === 'AR' ? 'لا يوجد موضوع مثبّت بعد.' : 'No topic pinned yet.');

  // REWRITE: "compare with ..." or "قارن مع/بين ..."
  const cmpTarget = extractCompareTarget(qRaw, lang);
  let rewrittenQuestion = qRaw || '';
  if (cmpTarget && s.topic) {
    rewrittenQuestion = (lang === 'AR')
      ? `قارن بين "${s.topic}" و "${cmpTarget}".`
      : `Compare "${s.topic}" with "${cmpTarget}".`;
  }

  // MCQ hint for the model (kept short)
  const mcqHint =
    (/^mcqs?(\s|$)/i.test(qRaw) || /^أسئلة(?:\s|-)?اختيار(?:\s|-)?من(?:\s|-)?متعدد/i.test(qRaw))
      ? (lang === 'AR'
          ? 'إن طلبت MCQs، أعطني أسئلة قصيرة مع مفتاح الإجابة في النهاية.'
          : 'If I ask for MCQs, give short items with an answer key at the end.')
      : '';

  try {
    // No question? prompt user
    if (!rewrittenQuestion) {
      return res.json({
        answer: lang === 'AR' ? 'اكتب سؤالك.' : 'Ask your question.',
        sources: []
      });
    }

    // EMBED
    const qVec = await embed(rewrittenQuestion);

    // SAFE Pinecone filter: only include metadata we *actually stored*
    const filter = {};
    if (s.stage)   filter.stage = s.stage;
    if (s.subject) filter.subject = s.subject;
    // IMPORTANT: do NOT add s.topic here unless you actually upserted it in metadata
    // if (s.topic) filter.topic = s.topic; // <-- leave OUT

    const pineParams = {
      vector: qVec,
      topK: 5,
      includeMetadata: true
    };
    if (Object.keys(filter).length > 0) pineParams.filter = filter;

    const pine = await index.query(pineParams);

    // Build context + sources
    const hits = pine.matches || [];
    const context = hits
      .map((m, i) => {
        const meta = m.metadata || {};
        const text = meta.text || meta.chunk || '';
        const src = meta.file || meta.id || meta.source || '';
        return `[#${i + 1}] ${src}\n${text}`;
      })
      .join('\n\n---\n\n');

    const sources = hits.map((m, i) => {
      const meta = m.metadata || {};
      return {
        id: meta.file || meta.id || (m.id || `match-${i + 1}`),
        score: m.score
      };
    });

    // Short recent history (last 3 Q&A)
    const historyTail = (s.history || [])
      .slice(-3)
      .map(h => `Q: ${trimStr(h.q, 200)}\nA: ${trimStr(h.a, 300)}`)
      .join('\n---\n');

    const histBlock = historyTail
      ? (lang === 'AR'
          ? `\nآخر الحوار:\n${historyTail}\n`
          : `\nRecent context:\n${historyTail}\n`)
      : '';

    // SYSTEM PROMPT
    const sys =
      (lang === 'AR')
        ? `أنت مُعلّم صيدلة ذكي: مختصر، دقيق، موجّه للاختبارات.
- استخدم لغة بسيطة وعناوين فرعية ونِقاط.
- اربط الإجابات بسياق السؤال والمصادر المقتبسة عند وجودها.
- عندما dim أو السياق ضعيفان، كن واضحًا بما لا تعرفه واقترح سؤالًا متابعة مناسبًا.
- عند طلب MCQs أعطِ أسئلة قصيرة متدرجة الصعوبة مع مفتاح الإجابة في النهاية.`
        : `You are a smart pharmacy tutor: concise, accurate, exam-oriented.
- Use plain language, mini-headings, and bullets.
- Tie answers to the user’s question and the retrieved context when helpful.
- If signal is weak, say so and propose a reasonable follow-up.
- When asked for MCQs, provide short items with an answer key at the end.`;

    const messages = [
      { role: 'system', content: sys },
      {
        role: 'user',
        content:
`${lang === 'AR' ? 'لغة الإجابة' : 'Answer language'}: ${lang}
${prefsLine}
${topicLine}
${histBlock}

${lang === 'AR' ? 'السياق' : 'Context'}:
${context || (lang === 'AR' ? '(لا سياق متاح)' : '(no context available)')}

${lang === 'AR' ? 'السؤال' : 'Question'}: ${rewrittenQuestion}

${mcqHint}
`
      }
    ];

    // CHAT
    let answer = await chat(messages);
    if (!answer) {
      answer = lang === 'AR'
        ? 'عذرًا، لم أجد إجابة موثوقة من السياق.'
        : 'Sorry, I could not find a reliable answer from context.';
    }

    // Save to history using the rewritten question
    s.history.push({ q: rewrittenQuestion, a: answer, t: Date.now() });

    return res.json({
      answer,
      sources: sources.slice(0, 5)
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'Query failed', detail: String(e) });
  }
});

/* ---------- START ---------- */
app.listen(PORT, () => {
  console.log(`pharmaninja-backend listening on :${PORT}`);
});
