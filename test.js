import 'dotenv/config';
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/embeddings";
import { FAISS } from "@langchain/vectorstores/faiss";
import { Document } from "@langchain/document";
import { RetrievalQAChain } from "@langchain/chains";
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";
import { traceable } from "langsmith/traceable";

// your resume & post
const RESUME_TEXT = `Front end React, Next.js, Redux
TypeScript & JavaScript
Figma to code & Animations
Tanstack/React Query & SWR
Tailwind CSS, Bootstrap, MaterialUI, Ant Design, PrimeReact
PostCSS, Webpack, Vite
Jest & Playwright
Progressive Web App (PWA) backend Node.js, Express, Nest.js
TypeScript
Redis
REST & GraphQL APIs
PostgreSQL, MySQL, Oracle
MongoDB, Amazon DocumentDB
MVC, DDD, Clean & Event-Driven Arch.
AWS, Vercel, Heroku
SonarQube
Docker, Rancher
GitHub Actions & Bitbucket Pipelines
Sentry
CloudWatch
Google Analytics
Error Handling & Log Monitoring
Cursor, OpenAI
LangChain & LangGraph & LangSmith`;

const POST_TEXT = `We’re Hiring: Senior Machine Learning Engineer (LLMs & Infrastructure)

Hi everyone! 👋 At WillDom, we’re looking for a Senior ML Engineer to help us build smart, scalable NLP solutions using the latest deep learning and MLOps tools.

💰 USD pay | 📝 Contractor role | 🌎 100% Remote (Latam) | 🤖 Cutting-edge AI

What you’ll do:
Build real-time NLP agents with BERT, SmallBERT, and Hugging Face TGI.
Deploy and manage models at scale on Azure AKS (GPU support) using Kubernetes & Helm.
Develop high-performance APIs with FastAPI.
Automate workflows with CI/CD pipelines (Azure DevOps).

✅ What we’re looking for:
React, Next.js, Typescript, Tailwind CSS,
Nest, Node, Jest,
5+ years in ML or software engineering.
Strong Python skills (3.x).
Experience with ML infrastructure, deployment, and cloud GPUs (Azure preferred).
Bonus: knowledge of C++, C#, or Rust.`;

// classification buckets
const titleBuckets = [
  { name: 'Full-Stack', keywords: ['Full-Stack','Fullstack','Full Stack'] },
  { name: 'Backend',    keywords: ['Backend','Back-End','API'] },
  { name: 'Frontend',   keywords: ['Frontend','Front-End','UI'] },
  { name: 'Support',    keywords: ['Support','Help Desk','Technical Support'] },
  { name: 'Data',       keywords: ['Data','Scientist','Engineer'] },
];

// schema for our graph state
const StateSchema = z.object({
  resume: z.string(),
  post: z.string(),
  extracted: z.record(z.any()).optional(),
  normalized: z.record(z.any()).optional(),
  vectorStore: z.any().optional(),
  retrievedContext: z.string().optional(),
  matchResult: z.object({
    match: z.boolean(),
    reasons: z.array(z.string()),
    techMatchCount: z.number(),
    totalRequiredTechs: z.number(),
    matchPercentage: z.number(),
  }).optional(),
});

// 1) fetch texts
const fetchTexts = traceable(
  async () => ({ resume: RESUME_TEXT, post: POST_TEXT }),
  { name: 'fetchTexts' }
);

// 2) build a FAISS vector store over your resume
const buildVectorStore = traceable(
  async (state) => {
    const embeddings = new OpenAIEmbeddings();
    // split resume into chunks (naïve: one paragraph per doc)
    const docs = state.resume.split('\n\n').map((chunk, i) =>
      new Document({ pageContent: chunk, metadata: { chunk } })
    );
    const vectorStore = await FAISS.fromDocuments(docs, embeddings);
    return { vectorStore };
  },
  { name: 'buildVectorStore' }
);

// 3) retrieve the top‑k relevant chunks given the job post
const retrieveRelevant = traceable(
  async (state) => {
    const chain = RetrievalQAChain.fromLLM(
      new ChatOpenAI({ temperature: 0, model: 'o4-mini' }),
      state.vectorStore.asRetriever()
    );
    const res = await chain.call({ query: state.post });
    return { retrievedContext: res.text };
  },
  { name: 'retrieveRelevant' }
);

// 4) extract raw fields from the post
const extractJobFields = traceable(
  async (state) => {
    const prompt = `Extract Title, Technologies, Seniority, Remote (true/false), SalaryRange from this job post as JSON.\n\nPost:\n${state.post}`;
    const resp = await new ChatOpenAI({ temperature: 0, model: 'o4-mini' })
      .invoke([{ role: 'user', content: prompt }]);
    let parsed;
    try { parsed = JSON.parse(resp.content); }
    catch { parsed = {}; }
    return { extracted: parsed };
  },
  { name: 'extractJobFields' }
);

// 5) normalize those fields into your buckets & standard names
const normalizeFields = traceable(
  (state) => {
    const { title = '', Technologies = [], Seniority, Remote, SalaryRange } = state.extracted ?? {};
    let category = 'Other';
    for (const bucket of titleBuckets) {
      if (bucket.keywords.some(k => title.includes(k))) {
        category = bucket.name; break;
      }
    }
    const techs = Technologies.map((t) => {
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

// 6) final LLM‑based match decision using both retrieved context and normalized tech list
const classifyMatch = traceable(
  async (state) => {
    const { techs = [] } = state.normalized;
    const techMatchCount = techs.filter(t => state.retrievedContext.includes(t)).length;
    const totalRequiredTechs = techs.length;
    const matchPercentage = totalRequiredTechs
      ? Math.round((techMatchCount / totalRequiredTechs) * 100)
      : 0;

    const prompt = `Based on these resume excerpts:\n${state.retrievedContext}\n\n` +
      `And this job:\n${JSON.stringify(state.normalized, null, 2)}\n\n` +
      `I match ${techMatchCount} out of ${totalRequiredTechs} required techs (${matchPercentage}%). ` +
      `Respond with valid JSON { match: boolean, reasons: string[], techMatchCount, totalRequiredTechs, matchPercentage }.\n` +
      `Threshold for match is >=50%.`;

    const resp = await new ChatOpenAI({ temperature: 0, model: 'o4-mini' })
      .invoke([{ role: 'user', content: prompt }]);

    let mr;
    try { mr = JSON.parse(resp.content); }
    catch {
      mr = {
        match: matchPercentage >= 50,
        reasons: [`Parse error, defaulting to ${matchPercentage >= 50}`],
        techMatchCount, totalRequiredTechs, matchPercentage
      };
    }
    return { matchResult: mr };
  },
  { name: 'classifyMatch' }
);

// 7) log it all out
const notify = traceable(
  (state) => {
    console.log('Extracted:', state.extracted);
    console.log('Normalized:', state.normalized);
    console.log('Match Result:', state.matchResult);
    return {};
  },
  { name: 'notify' }
);

// assemble your graph
const graph = new StateGraph(StateSchema)
  .addNode('fetchTexts', fetchTexts)
  .addEdge(START, 'fetchTexts')
  .addNode('buildVectorStore', buildVectorStore)
  .addEdge('fetchTexts', 'buildVectorStore')
  .addNode('retrieveRelevant', retrieveRelevant)
  .addEdge('buildVectorStore', 'retrieveRelevant')
  .addNode('extractJobFields', extractJobFields)
  .addEdge('retrieveRelevant', 'extractJobFields')
  .addNode('normalizeFields', normalizeFields)
  .addEdge('extractJobFields', 'normalizeFields')
  .addNode('classifyMatch', classifyMatch)
  .addEdge('normalizeFields', 'classifyMatch')
  .addNode('notify', notify)
  .addEdge('classifyMatch', 'notify');

export const app = graph.compile({ name: 'job-matcher-rag', slug: 'job-matcher-rag' });
