import "dotenv/config";
import { v4 as uuidv4 } from "uuid";
import { app } from "./index.js";        // <-- note: importing the exported graph

process.on('uncaughtException', err => {
  console.error('✘ Uncaught Exception – shutting down:', err);
  // flush logs/metrics here if needed…
  process.exit(1);        // or use a graceful shutdown routine
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('✘ Unhandled Rejection at:', promise, 'reason:', reason);
  // same deal: flush & exit
  process.exit(1);
});

(async () => {
  try {
    await app.invoke(
      {},
      { configurable: { thread_id: uuidv4() } }
    );
  } catch (err) {
    console.error('✘ Graph invocation failed:', err);
    // decide if to exit(1) or keep running
  }
})();
