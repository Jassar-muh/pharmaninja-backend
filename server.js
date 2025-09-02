// --------- server.js ---------
import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import cors from "cors";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

// --- ENV CHECK ---
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

// --- GLOBALS ---
let pineconeClient, index;
(async () => {
  try {
    pineconeClient = new Pinecone({ apiKey: PINECONE_API_KEY });
    index = pineconeClient.index(PINECONE_INDEX);
    console.log("✅ Pinecone index ready:", PINECONE_INDEX);
  } catch (err) {
    console.error("❌ Pinecone init failed:", err.message);
  }
})();

// --- HELPERS ---
async function openaiEmb(input) {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "text-embedding-3-small",
      input,
    }),
  });
  if (!r.ok) throw new Error("OpenAI embeddings error " + r.status);
  const j = await r.json();
  return j.data[0].embedding;
}

function buildFilter({ stage, subject }) {
  const f = {};
  if (stage) f.stage = stage;
  if (subject) f.subject = subject;
  return Object.keys(f).length ? f : null;
}

// --- ROUTES ---
app.get("/ping", (_req, res) => {
  res.send("pong");
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, uptime: process.uptime(), ts: Date.now() });
});

app.get("/selftest", async (_req, res) => {
  try {
    // env check
    const env = {
      OPENAI_API_KEY: !!OPENAI_API_KEY,
      PINECONE_API_KEY: !!PINECONE_API_KEY,
      PINECONE_INDEX: PINECONE_INDEX || "(missing)",
    };

    // test embedding
    let embDim = 0;
    try {
      const vec = await openaiEmb("hello");
      embDim = vec.length;
    } catch (e) {
      return res.status(500).json({ error: "Embedding failed", detail: e.message });
    }

    // test pinecone
    let pineRes;
    try {
      pineRes = await index.query({
        vector: new Array(1536).fill(0),
        topK: 1,
      });
    } catch (e) {
      return res.status(500).json({ error: "Pinecone failed", detail: e.message });
    }

    res.json({
      env,
      openai: { ok: embDim > 0, dim: embDim },
      pinecone: { ok: true, matches: pineRes?.matches?.length || 0 },
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// -------- MAIN QUERY --------
app.post("/query", async (req, res) => {
  try {
    const { lang = "EN", stage, subject, question } = req.body || {};
    if (!question || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Missing question" });
    }
    if (!OPENAI_API_KEY || !PINECONE_API_KEY || !PINECONE_INDEX) {
      return res.status(500).json({ error: "Missing env vars" });
    }
    if (!index) {
      return res.status(500).json({ error: "Pinecone index not ready" });
    }

    // 1. Embed question
    const qVec = await openaiEmb(question);

    // 2. Search Pinecone
    const filter = buildFilter({ stage, subject });
    const pine = await index.query({
      vector: qVec,
      topK: 5,
      includeMetadata: true,
      ...(filter ? { filter } : {}),
    });

    const matches = pine?.matches || [];
    const context = matches
      .map((m, i) => `[#${i + 1}] ${m?.metadata?.text || ""}`)
      .join("\n");

    // 3. Ask OpenAI completion
    const prompt = `Answer in ${lang}. Use this context:\n${context}\n\nQ: ${question}\nA:`;
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      throw new Error("OpenAI completion error: " + r.status + " " + txt);
    }
    const j = await r.json();
    const answer = j.choices?.[0]?.message?.content || "(no answer)";

    res.json({
      answer,
      sources: matches.map((m) => ({
        id: m.id,
        score: m.score,
        file: m?.metadata?.file,
      })),
    });
  } catch (e) {
    console.error("QUERY ERROR:", e.message);
    res.status(500).json({ error: "Query failed", detail: e.message });
  }
});

// --- START SERVER ---
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
