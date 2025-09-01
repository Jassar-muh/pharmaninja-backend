// server.js
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

if (!process.env.OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY in .env");
if (!process.env.PINECONE_API_KEY) throw new Error("Missing PINECONE_API_KEY in .env");
if (!process.env.PINECONE_INDEX) throw new Error("Missing PINECONE_INDEX in .env");
if (!process.env.PINECONE_HOST) throw new Error("Missing PINECONE_HOST in .env");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX, process.env.PINECONE_HOST);

function clip(str, max = 2000) {
  if (!str) return "";
  return str.length > max ? str.slice(0, max) + "..." : str;
}
function detectLang(s) { return /[\u0600-\u06FF]/.test(s) ? "AR" : "EN"; }
async function embedQuery(q) {
  const e = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: q,
  });
  return e.data[0].embedding;
}
function buildPrompt({ lang, question, context }) {
  const sys = (lang === "AR")
    ? "أنت مُدرّس صيدلة مساعد. أجب بدقة وباختصار اعتمادًا على النص المُقدّم فقط. إذا لم يكفِ، قل لا أعلم. لا تختلق مصادر."
    : "You are a helpful pharmacology tutor. Answer concisely and accurately based ONLY on the provided context. If it’s insufficient, say you don’t know. Do not fabricate sources.";
  const user = (lang === "AR")
    ? `السؤال: ${question}\n\nنص مرجعي:\n${context}\n\nأجب بالعربية.`
    : `Question: ${question}\n\nReference context:\n${context}\n\nAnswer in English.`;
  return { sys, user };
}

app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/query", async (req, res) => {
  try {
    const { lang: langIn, stage, subject, question } = req.body || {};
    if (!question || !question.trim()) {
      return res.status(400).json({ error: "Missing 'question'." });
    }

    const lang = (langIn || detectLang(question)).toUpperCase();
    const qVec = await embedQuery(question);

    const filter = {};
    if (stage) filter.stage = stage;
    if (subject) filter.subject = subject;

    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      filter: Object.keys(filter).length ? filter : undefined,
    });

    const matches = pine.matches || [];
    const context = clip(matches.map((m,i)=>`[${i+1}] ${m?.metadata?.text ?? ""}`).join("\n\n"), 2000);
    if (!context) {
      const fallback = (lang === "AR")
        ? "عذرًا، لم أعثر على نص مناسب للإجابة من المراجع المُحمّلة."
        : "Sorry, I couldn't find relevant context from the uploaded materials.";
      return res.json({ answer: fallback, sources: [] });
    }

    const { sys, user } = buildPrompt({ lang, question, context });
    const chat = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      messages: [
        { role: "system", content: sys },
        { role: "user", content: user },
      ],
    });

    const answer = chat.choices?.[0]?.message?.content?.trim() || "";
    const sources = matches.map(m => ({
      id: m.id,
      score: Number(m.score?.toFixed?.(3) ?? m.score ?? 0),
      file: m?.metadata?.source || "unknown",
      lang: m?.metadata?.lang || "unknown",
      stage: m?.metadata?.stage || undefined,
      subject: m?.metadata?.subject || undefined,
    }));

    return res.json({ answer, sources });
  } catch (e) {
    console.error(e?.response?.data || e.message);
    res.status(500).json({ error: "Query failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ API running: http://localhost:${PORT}`));
