import natural from "natural";
import axios from "axios";
import path from "path";
import { promises as fs } from "fs";

// -------------------- Load knowledge.json --------------------
let knowledge = [];
const knowledgePath = path.join(process.cwd(), "knowledge.json"); // adjust if your file is elsewhere

async function loadKnowledge() {
  if (!knowledge.length) {
    const data = await fs.readFile(knowledgePath, "utf-8");
    knowledge = JSON.parse(data);
  }
  return knowledge;
}

// -------------------- Utility Functions --------------------
function extractKeywords(text) {
  const tokenizer = new natural.WordTokenizer();
  const words = tokenizer.tokenize(text.toLowerCase());
  const stopwords = ["i","want","do","you","is","the","a","an","und","die","das","der"];
  return words.filter(w => !stopwords.includes(w));
}

async function findRelevantKnowledge(question) {
  const knowledgeData = await loadKnowledge();
  const keywords = extractKeywords(question);
  let bestMatch = null;
  let bestScore = 0;

  for (const entry of knowledgeData) {
    if (!entry.patterns) continue;
    for (const pattern of entry.patterns) {
      for (const keyword of keywords) {
        const similarity = natural.JaroWinklerDistance(keyword, pattern.toLowerCase());
        if (similarity > bestScore) {
          bestScore = similarity;
          bestMatch = entry;
        }
      }
    }
  }
  return bestScore > 0.8 ? bestMatch : null;
}

// -------------------- API Handler --------------------
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ reply: "Only POST requests are allowed." });
  }

  const { message } = req.body;
  if (!message) return res.json({ reply: "Keine Nachricht erhalten." });

  try {
    const matched = await findRelevantKnowledge(message);

    if (!matched) {
      return res.json({
        reply:
          "Entschuldigung, das habe ich nicht verstanden. Bitte schreiben Sie uns an <a href='mailto:info@labelmonster.eu'>info@labelmonster.eu</a>.",
      });
    }

    const prompt = `
You are the official assistant of Labelmonster.

Your ONLY task:
Use **only** the provided "Knowledge Base Entry" to answer the user's question. 
Understand the context of the user's question and look for similar questions in the "Knowledge Base Entry" to answer.
You are NOT allowed to add, guess, or invent information beyond what is in the Answer field.
If the provided answer already fits, repeat it naturally — do not rephrase or expand beyond minor adjustments for fluency.

Knowledge Base Entry:
Patterns: ${matched.patterns.join(", ")}
Answer: ${matched.answer}

Instructions:
- Answer **only** from the "Answer" field above.
- NEVER add external or unrelated information.
- Always answer in **German**, even if the question is in another language.
- If the question is about you – even indirectly – assume the user means Labelmonster. Provide the answer only from the 'Answer' field.
- Write in a professional, natural, and friendly tone suitable for a company chatbot.
- Do NOT start the reply with phrases like "Antwort:", "Answer:", or quotes.
- Any user question related to address or location should be "Großenbaumer Allee 98, 47269 Duisburg".
- If you are not confident or the information is not in the "Answer", do not attempt to answer; instead, say exactly:
  "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <a href='mailto:info@labelmonster.eu'>info@labelmonster.eu</a>, damit wir Ihnen besser weiterhelfen können."

User Question: ${message}

Antwort (strictly based on knowledge base):
`;

    const HF_API_KEY = process.env.HF_API_KEY;

    const response = await axios.post(
      "https://api-inference.huggingface.co/models/google/gemma-2b-it",
      { inputs: prompt },
      {
        headers: {
          Authorization: `Bearer ${HF_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: 60000,
      }
    );

    let reply = "Entschuldigung, ich konnte keine Antwort generieren.";
    if (Array.isArray(response.data) && response.data[0]?.generated_text) {
      reply = response.data[0].generated_text.trim();
    } else if (typeof response.data === "string") {
      reply = response.data.trim();
    }

    return res.status(200).json({ reply });
  } catch (err) {
    console.error("❌ Fehler beim Abrufen der Antwort:", err.message);
    return res.status(500).json({
      reply: "Fehler beim Abrufen der Antwort von Gemma.",
    });
  }
}
