import 'dotenv/config';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { StateGraph, START } from '@langchain/langgraph';
import { z } from 'zod';
import { traceable } from 'langsmith/traceable';
import { knownTechs } from './seed';

// Initialize LLM and embeddings client
const model = new ChatOpenAI({ temperature: 1, model: 'o4-mini' });
const embeddings = new OpenAIEmbeddings();

// Cosine similarity helper
function cosine(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot    += a[i] * b[i];
    magA   += a[i] * a[i];
    magB   += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// Pre-embed knownTechs once
let knownTechVectors;
async function initKnownTechVectors() {
  knownTechVectors = await embeddings.embedDocuments(knownTechs);
}
await initKnownTechVectors();

// Hardcoded job post for demo
const POST_TEXT = `🚀 We’re hiring! | Senior Fullstack Engineer (React / Node / GraphQL)

🌎 100% Remote | 💼 6-month contract (with extension) | 🌐 Global client

We’re looking for a strong fullstack dev experienced in:
✅ React.js + TypeScript
 ✅ Node.js
 ✅ Apollo GraphQL (Client & Server)

💡 You’ll work on system migration, federated GraphQL APIs, performance tuning, and fullstack delivery.`;

// const POST_TEXT = `We’re Hiring: Senior Machine Learning Engineer (LLMs & Infrastructure)

// Hi everyone! 👋 At WillDom, we’re looking for a Senior ML Engineer to help us build smart, scalable NLP solutions using the latest deep learning and MLOps tools.

// 💰 USD pay | 📝 Contractor role | 🌎 100% Remote (Latam) | 🤖 Cutting-edge AI

// What you’ll do:
// Build real-time NLP agents with BERT, SmallBERT, and Hugging Face TGI.
// Deploy and manage models at scale on Azure AKS (GPU support) using Kubernetes & Helm.
// Develop high-performance APIs with FastAPI.
// Automate workflows with CI/CD pipelines (Azure DevOps).

// ✅ What we’re looking for:
// React, Next.js, Typescript, Tailwind CSS,
// Nest, Node, Jest,
// 5+ years in ML or software engineering.
// Strong Python skills (3.x).
// Experience with ML infrastructure, deployment, and cloud GPUs (Azure preferred).
// Bonus: knowledge of C++, C#, or Rust.`;

// Title classification buckets
const titleBuckets = [
  { name: 'Full-Stack', keywords: ['Full-Stack','Fullstack','Full Stack'] },
  { name: 'Backend', keywords: ['Backend','Back-End','API'] },
  { name: 'Frontend', keywords: ['Frontend','Front-End','UI'] },
  { name: 'Support', keywords: ['Support','Help Desk','Technical Support'] },
  { name: 'Data', keywords: ['Data','Scientist','Engineer'] },
];

// State schema definition
const StateSchema = z.object({
  post: z.string(),
  extracted: z.record(z.any()).optional(),
  normalized: z.record(z.any()).optional(),
  matchResult: z.object({
    match: z.boolean(),
    reasons: z.array(z.string()),
    techMatchCount: z.number(),
    totalRequiredTechs: z.number(),
    matchPercentage: z.number()
  }).optional(),
});

// 1) Fetch job post
const fetchPostDescription = traceable(
  async () => ({ post: POST_TEXT }),
  { name: 'fetchPostDescription' }
);

// 2) Extract structured fields from post
const extractJobFields = traceable(
  async (state) => {
    const prompt = `Extract title, technologies, seniority, remote (true/false), and salary range from this job post as JSON.\n\nPost:\n${state.post}`;
    const resp = await model.invoke([{ role: 'user', content: prompt }]);
    let parsed;
    try { parsed = JSON.parse(resp.content); } catch { parsed = {}; }
    console.log('parsed', parsed);
    return { extracted: parsed };
  },
  { name: 'extractJobFields' }
);

// 3) Normalize extracted fields
const normalizeFields = traceable(
  (state) => {
    const { title = '', Technologies = [], Seniority, Remote, SalaryRange } = state.extracted || {};
    let category = 'Other';
    for (const bucket of titleBuckets) {
      if (bucket.keywords.some(k => title.includes(k))) { category = bucket.name; break; }
    }
    const techs = Technologies.map(t => {
      const lower = t.toLowerCase();
      if (lower.includes('node')) return 'Node.js';
      if (lower.includes('react')) return 'React';
      if (lower.includes('javascript')) return 'JavaScript';
      if (lower.includes('aws')) return 'AWS';
      return t;
    });
    return { normalized: { title, techs, Seniority, Remote, SalaryRange, category } };
  },
  { name: 'normalizeFields' }
);

// 4) Embedding-based match classification (on-the-fly)
const classifyMatch = traceable(
  async (state) => {
    const requiredTechs = state.normalized?.techs || [];
    let techMatchCount = 0;
    for (const tech of requiredTechs) {
      const [reqVec] = await embeddings.embedDocuments([tech]);
      const bestSim = Math.max(...knownTechVectors.map(kv => cosine(reqVec, kv)));
      if (bestSim >= 0.75) techMatchCount++;
    }
    const totalRequiredTechs = requiredTechs.length;
    const matchPercentage = totalRequiredTechs ? Math.round((techMatchCount / totalRequiredTechs) * 100) : 0;
    const match = matchPercentage >= 49;
    const reasons = match ? ['Sufficient similarity overlap'] : ['Insufficient similarity overlap'];
    return { matchResult: { match, reasons, techMatchCount, totalRequiredTechs, matchPercentage } };
  },
  { name: 'classifyMatch' }
);

// 5) Notify or log final result
const notify = traceable(
  (state) => {
    console.log('Extracted:', state.extracted);
    console.log('Normalized:', state.normalized);
    console.log('Match Result:', state.matchResult);
    return {};
  },
  { name: 'notify' }
);

// 6) Build and export graph
const graph = new StateGraph(StateSchema)
  .addNode('fetchPostDescription', fetchPostDescription)
  .addEdge(START, 'fetchPostDescription')
  .addNode('extractJobFields', extractJobFields)
  .addEdge('fetchPostDescription', 'extractJobFields')
  .addNode('normalizeFields', normalizeFields)
  .addEdge('extractJobFields', 'normalizeFields')
  .addNode('classifyMatch', classifyMatch)
  .addEdge('normalizeFields', 'classifyMatch')
  .addNode('notify', notify)
  .addEdge('classifyMatch', 'notify');

export const app = graph.compile({ name: 'job-matcher-embeddings', slug: 'job-matcher-embeddings' });
