// ingest.js
import dotenv from 'dotenv';
dotenv.config();

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { createHash } from 'crypto';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

// ---- Clients ----
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
// Pinecone Serverless needs HOST (from index page)
const index = pc.index(process.env.PINECONE_INDEX, process.env.PINECONE_HOST);

// ---- Paths ----
const dataDir = path.join(process.cwd(), 'data');

// ---- Helpers ----
function pdftotext(filePath) {
  // requires: brew install poppler
  const out = execFileSync('pdftotext', ['-enc', 'UTF-8', '-layout', filePath, '-']);
  return out.toString('utf8');
}

function tryOCR(filePath) {
  // requires: brew install ocrmypdf
  const ocrPath = filePath.replace(/\.pdf$/i, '.ocr.pdf');
  execFileSync('ocrmypdf', ['--deskew', '--optimize', '1', '--force-ocr', filePath, ocrPath], { stdio: 'ignore' });
  return ocrPath;
}

// Remove invalid surrogate code units to avoid “Missing low surrogate”
function cleanText(s) {
  return s.replace(/[\uD800-\uDFFF]/g, '');
}

function detectLang(s) {
  return /[\u0600-\u06FF]/.test(s) ? 'AR' : 'EN';
}

function chunkText(text, size = 1200, overlap = 150) {
  const chunks = [];
  for (let i = 0; i < text.length; i += (size - overlap)) {
    const part = text.slice(i, i + size).trim();
    if (part) chunks.push(part);
  }
  return chunks;
}

function safeId(s) {
  return createHash('sha1').update(s, 'utf8').digest('hex'); // ASCII hex id
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Retry wrapper for OpenAI 429s (TPM/RPM)
async function embedBatchWithRetry(texts, { retries = 5, baseDelayMs = 1500 } = {}) {
  let attempt = 0;
  while (true) {
    try {
      const res = await openai.embeddings.create({
        model: 'text-embedding-3-small', // 1536 dims
        input: texts,
      });
      return res.data.map(d => d.embedding);
    } catch (e) {
      const msg = e?.response?.data?.error?.message || e.message || '';
      const is429 = e?.status === 429 || /rate limit/i.test(msg);
      if (!is429 || attempt >= retries) throw e;
      const m = msg.match(/try again in ([\d.]+)s/i);
      const serverWait = m ? Math.ceil(parseFloat(m[1]) * 1000) : null;
      const delay = serverWait ?? Math.min(baseDelayMs * (2 ** attempt), 15000);
      console.warn(`⚠️  429 rate limit. Waiting ${Math.ceil(delay/1000)}s… (attempt ${attempt + 1}/${retries})`);
      await sleep(delay);
      attempt++;
    }
  }
}

// ---- Main ----
async function run() {
  if (!process.env.OPENAI_API_KEY) throw new Error('Missing OPENAI_API_KEY in .env');
  if (!process.env.PINECONE_API_KEY) throw new Error('Missing PINECONE_API_KEY in .env');
  if (!process.env.PINECONE_INDEX) throw new Error('Missing PINECONE_INDEX in .env');
  if (!process.env.PINECONE_HOST) throw new Error('Missing PINECONE_HOST in .env (serverless host url)');

  if (!fs.existsSync(dataDir)) throw new Error(`data folder not found: ${dataDir}`);
  const pdfs = fs.readdirSync(dataDir).filter(f => f.toLowerCase().endsWith('.pdf'));
  if (!pdfs.length) throw new Error('No PDFs in ./data');

  // small batch to stay under TPM
  const BATCH = 8;

  for (const fname of pdfs) {
    const fpath = path.join(dataDir, fname);
    console.log('📄', fname);

    let text = '';
    try {
      text = pdftotext(fpath);
    } catch (e) {
      console.error(`⚠️  pdftotext failed for ${fname}: ${e.message}`);
    }

    if (!text.trim()) {
      try {
        const ocrPath = tryOCR(fpath);
        console.log(`🔎 OCR applied → ${path.basename(ocrPath)}`);
        text = pdftotext(ocrPath);
      } catch (e) {
        console.warn(`⚠️  Skipping ${fname}: no text extracted even after OCR`);
        continue;
      }
    }

    const chunks = chunkText(text);
    if (!chunks.length) {
      console.warn(`⚠️  Skipping ${fname}: empty after chunking`);
      continue;
    }
    console.log(`✂️  ${chunks.length} chunks`);

    const baseId = safeId(fname);

    for (let i = 0; i < chunks.length; i += BATCH) {
      const batchTexts = chunks.slice(i, i + BATCH).map(cleanText);
      let embeddings;
      try {
        embeddings = await embedBatchWithRetry(batchTexts);
      } catch (e) {
        console.error('❌ Embedding failed:', e?.response?.data || e.message);
        await sleep(3000);
        continue;
      }

      const vectors = embeddings.map((values, j) => ({
        id: `${baseId}-${i + j}`,
        values,
        metadata: {
          text: batchTexts[j],
          source: fname,
          lang: detectLang(batchTexts[j]),
          stage: '3rd',
          subject: 'Pharmacology',
        },
      }));

      await index.upsert(vectors);
      console.log(`⬆️  ${Math.min(i + BATCH, chunks.length)}/${chunks.length}`);
      await sleep(300); // gentle pacing
    }

    console.log(`✅ Finished ${fname}`);
  }

  console.log('🎉 All files uploaded');
}

run().catch(e => {
  console.error('❌', e?.response?.data || e.message);
  process.exit(1);
});
