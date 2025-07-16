import "dotenv/config";
import { v4 as uuidv4 } from "uuid";
import { app } from "./index.js";        // <-- note: importing the exported graph

(async () => {
  await app.invoke({}, { configurable: { thread_id: uuidv4() } });
})();
