import 'dotenv/config';
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";
import { traceable } from "langsmith/traceable";

// 1) Define state schema
// type MatchResult = { match: boolean; reasons?: string[] };
const StateSchema = z.object({
  resume: z.string(),
  post: z.string(),
  matchResult: z.object({ match: z.boolean(), reasons: z.array(z.string()).optional() }).optional(),
});

// 2) Traceable nodes
const fetchPostNode = traceable(
  async () => {
    const resumeText = `Passionate Software Engineer with 5+ years experience in JavaScript, React, Node.js, and cloud technologies.`;
    const postText = `🛠️Technical Support Engineer
🧩Experience in Technical Support Engineering, Customer Support, or in similar roles.
🗣Native level English proficiency.
👨‍💻Software Engineering Experience.
💰USD 15/USD 45 per hour`;
    return { resume: resumeText, post: postText };
  },
  { name: "fetchPost" }
);

const classifyMatchNode = traceable(
  async (state) => {
    const model = new ChatOpenAI({ temperature: 0 });
    const prompt = `You are an expert at matching job descriptions to candidate resumes.\n\nResume:\n${state.resume}\n\nJob Post:\n${state.post}\n\nRespond with valid JSON with keys \"match\" (boolean) and \"reasons\" (array of strings) explaining why it's a good or poor fit.`;
    const resp = await model.call([{ role: 'user', content: prompt }]);
    try {
      const parsed = JSON.parse(resp.content);
      return { matchResult: parsed };
    } catch (err) {
      console.error('GPT parse error:', resp.content);
      return { matchResult: { match: false, reasons: ['Unable to parse model response'] } };
    }
  },
  { name: "classifyMatch" }
);

const notifyNode = traceable(
  async (state) => {
    console.log('Match:', state.matchResult?.match);
    if (state.matchResult?.reasons) {
      console.log('Reasons:', state.matchResult.reasons.join('; '));
    }
    return {};
  },
  { name: "notify" }
);

// 3) Assemble & compile the graph
const graph = new StateGraph(StateSchema)
  .addNode("fetchPost", fetchPostNode)
  .addEdge(START, "fetchPost")
  .addNode("classifyMatch", classifyMatchNode)
  .addEdge("fetchPost", "classifyMatch")
  .addNode("notify", notifyNode)
  .addEdge("classifyMatch", "notify");

export const app = graph.compile({
  name: "job-matcher",
  slug: "job-matcher"
});
