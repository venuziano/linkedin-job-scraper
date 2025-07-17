import 'dotenv/config';
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";
import { traceable } from "langsmith/traceable";
import { knownTechs } from './seed';

// Initialize LLM model
const model = new ChatOpenAI({
  temperature: 1,
  // topP
  model: "o4-mini"
});

// Hardcoded resume and post for demo
// const RESUME_TEXT = `Front end React, Next.js, Redux
// TypeScript & JavaScript
// Figma to code & Animations
// Tanstack/React Query & SWR
// Tailwind CSS, Bootstrap, MaterialUI, Ant Design, PrimeReact
// PostCSS, Webpack, Vite
// Jest & Playwright
// Progressive Web App (PWA) backend Node.js, Express, Nest.js
// TypeScript
// Redis
// REST & GraphQL APIs
// PostgreSQL, MySQL, Oracle
// MongoDB, Amazon DocumentDB
// MVC, DDD, Clean & Event-Driven Arch.
// AWS, Vercel, Heroku
// SonarQube
// Docker, Rancher
// GitHub Actions & Bitbucket Pipeline
// Sentry
// CloudWatch
// Google Analytics
// Error Handling & Log Monitoring
// Cursor, OpenAI
// LangChain & LangGraph & LangSmith
// `;

// const POST_TEXT = `🚀 We’re hiring! | Senior Fullstack Engineer (React / Node / GraphQL)

// 🌎 100% Remote | 💼 6-month contract (with extension) | 🌐 Global client

// We’re looking for a strong fullstack dev experienced in:
// ✅ React.js + TypeScript
//  ✅ Node.js
//  ✅ Apollo GraphQL (Client & Server)
//  ✅ AWS (EKS, RDS), CI/CD, Docker, Kubernetes
// Nice to have: Python, Web Vitals, ArgoCD, CircleCI

// 💡 You’ll work on system migration, federated GraphQL APIs, performance tuning, and fullstack delivery.`;



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

// Title classification buckets (for normalization)
const titleBuckets = [
  { name: 'Full-Stack', keywords: ['Full-Stack','Fullstack','Full Stack'] },
  { name: 'Backend', keywords: ['Backend','Back-End','API'] },
  { name: 'Frontend', keywords: ['Frontend','Front-End','UI'] },
  { name: 'Support', keywords: ['Support','Help Desk','Technical Support'] },
  { name: 'Data', keywords: ['Data','Scientist','Engineer'] },
];

const StateSchema = z.object({
  // resume: z.string(),
  post: z.string(),
  extracted: z.record(z.any()).optional(),
  normalized: z.record(z.any()).optional(),
  // techMatch: z.boolean().optional(),
  matchResult: z.object({
    match: z.boolean(),
    reasons: z.array(z.string()),
    techMatchCount: z.number(),
    totalRequiredTechs: z.number(),
    matchPercentage: z.number()
  }).optional(),
});

// 2) Traceable nodes
// Fetch resume and post
const fetchPostDescription = traceable(
  async () => ({ post: POST_TEXT }),
  // async () => ({ resume: RESUME_TEXT, post: POST_TEXT }),
  { name: 'fetchPostDescription' }
);

// Dynamic extraction: parse structured fields
const extractJobFields = traceable(
  async (state) => {
    const prompt = `Extract Title, Technologies, Seniority, Remote (true/false), SalaryRange from this job post as JSON.\n\nPost:\n${state.post}`;
    const resp = await model.invoke([{ role: 'user', content: prompt }]);
    let parsed;
    try { parsed = JSON.parse(resp.content); }
    catch { parsed = {}; }
    console.log('parsed', parsed)
    return { extracted: parsed };
  },
  { name: 'extractJobFields' }
);

// Lightweight normalization
const normalizeFields = traceable(
  (state) => {
    const { title = '', Technologies = [], Seniority, Remote, SalaryRange } = state.extracted ?? {};
    // Normalize category
    let category = 'Other';
    for (const bucket of titleBuckets) {
      if (bucket.keywords.some(k => title.includes(k))) { category = bucket.name; break; }
    }
    // Normalize techs
    const techs = (Technologies ).map(t => {
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

// Filter by known technologies
// const filterTechMatch = traceable(
//   (state) => {
//     const techs = state.normalized?.techs || [];
//     const matched = techs.filter(t => knownTechs.includes(t));
//     return { techMatch: matched.length > 0 };
//   },
//   { name: 'filterTechMatch' }
// );

// Final LLM-based match decision (with tech match counts)
// const classifyMatch = traceable(
//   (state) => {
//     const req       = state.normalized?.techs || [];
//     const matchList = req.filter(t => knownTechs.includes(t));
//     const techMatchCount     = matchList.length;
//     const totalRequiredTechs = req.length;
//     const matchPercentage    = totalRequiredTechs
//       ? Math.round((techMatchCount / totalRequiredTechs) * 100)
//       : 0;
//     const match = matchPercentage >= 50;  // adjust your threshold here
//     const reasons = match
//       ? ['Sufficient technology overlap']
//       : ['Insufficient technology overlap'];
//     return {
//       matchResult: {
//         match,
//         reasons,
//         techMatchCount,
//         totalRequiredTechs,
//         matchPercentage
//       }
//     };
//   },
//   { name: 'classifyMatch' }
// );
const classifyMatch = traceable(
  async (state) => {
    const requiredTechs = state.normalized?.techs || [];
    const matchedTechs = requiredTechs.filter(t => knownTechs.includes(t));
    const techMatchCount = matchedTechs.length;
    const totalRequiredTechs = requiredTechs.length;
    const matchPercentage = totalRequiredTechs
      ? Math.round((techMatchCount / totalRequiredTechs) * 100)
      : 0;

    // if (!state.techMatch) {
    //   return { matchResult: { match: false, reasons: ['No matching technologies'], techMatchCount, totalRequiredTechs, matchPercentage } };
    // }
    const prompt = `Given my resume:\n${knownTechs.join(',')}\nAnd this job:\n${JSON.stringify(state.normalized)}\n` +
      `I match ${techMatchCount} out of ${totalRequiredTechs} required technologies (${matchPercentage}%). ` +
      `Respond with valid JSON { match, reasons: ["techMatch","techNonMatch"], techMatchCount, totalRequiredTechs, matchPercentage } explaining the fit. Note: match threshold is 49%`;

    const resp = await model.invoke([{ role: 'user', content: prompt }]);
    let mr;
    try { mr = JSON.parse(resp.content); }
    catch { mr = { match: false, reasons: ['Parse error'], techMatchCount, totalRequiredTechs, matchPercentage }; }
    return { matchResult: mr };
  },
  { name: 'classifyMatch' }
);

// Notify / output
const notify = traceable(
  (state) => {
    console.log('Extracted Fields:', state.extracted);
    console.log('Normalized:', state.normalized);
    // console.log('Tech Match:', state.techMatch);
    console.log('Match Result:', state.matchResult);
    return {};
  },
  { name: 'notify' }
);

// 3) Assemble & compile graph
const graph = new StateGraph(StateSchema)
  .addNode('fetchPostDescription', fetchPostDescription)
  .addEdge(START, 'fetchPostDescription')
  .addNode('extractJobFields', extractJobFields)
  .addEdge('fetchPostDescription', 'extractJobFields')
  .addNode('normalizeFields', normalizeFields)
  .addEdge('extractJobFields', 'normalizeFields')
  // .addNode('filterTechMatch', filterTechMatch)
  // .addEdge('normalizeFields', 'filterTechMatch')
  .addNode('classifyMatch', classifyMatch)
  .addEdge('normalizeFields', 'classifyMatch')
  .addNode('notify', notify)
  .addEdge('classifyMatch', 'notify');

export const app = graph.compile({ name: 'job-matcher', slug: 'job-matcher' });
