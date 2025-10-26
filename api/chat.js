import natural from "natural";
import path from "path";
import { promises as fs } from "fs";
import { OpenAI } from "openai";
import axios from "axios"; 

// -------------------- Load knowledge.json --------------------
let knowledge = [];
const knowledgePath = path.join(process.cwd(), "knowledge.json");

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

// -------------------- Hugging Face Client --------------------
const hfClient = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_API_KEY,
});

// -------------------- Google Form Config --------------------
const GOOGLE_FORM_URL = "https://docs.google.com/forms/u/0/d/e/1FAIpQLSf-mODoH3tjHMrzY8-j-choYfYHMXQUiGZCnWQ8V6cg02GXeA/formResponse";

// Replace with actual entry IDs from your Google Form fields
const GOOGLE_FORM_ENTRIES = {
  question: "entry.1378060286", // Field 1
  answer: "entry.2072247045",   // Field 2
  timestamp: "entry.455345515", // Field 3
};

// -------------------- API Handler --------------------
export default async function handler(req, res) {
  // --- Enable CORS for any origin ---
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ reply: "Only POST requests are allowed." });
  }

  const { message } = req.body;
  if (!message) return res.json({ reply: "Keine Nachricht erhalten." });

  try {
    const matched = await findRelevantKnowledge(message);

    if (!matched) {
      const fallbackReply =
        "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <a href='mailto:info@labelmonster.eu'>info@labelmonster.eu</a>, damit wir Ihnen besser weiterhelfen können.";

      // ✅ Log question + fallback reply
      await logToGoogleForm(message, fallbackReply);

      return res.json({ reply: fallbackReply });
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

    const chatCompletion = await hfClient.chat.completions.create({
      model: "google/gemma-2-2b-it:nebius",
      messages: [
        { role: "user", content: prompt },
      ],
    });

    let reply = chatCompletion.choices[0].message?.content?.trim() ||
                "Entschuldigung, ich konnte keine Antwort generieren.";

    // ✅ Log question + reply to Google Form
    await logToGoogleForm(message, reply);

    return res.status(200).json({ reply });
  } catch (err) {
    console.error("❌ Fehler beim Abrufen der Antwort:", err.message);
    return res.status(500).json({
      reply: "Fehler beim Abrufen der Antwort von Gemma.",
    });
  }
}

// -------------------- Log to Google Form --------------------
async function logToGoogleForm(question, answer) {
  try {
    const payload = new URLSearchParams();
    payload.append(GOOGLE_FORM_ENTRIES.question, question);
    payload.append(GOOGLE_FORM_ENTRIES.answer, answer);

    await axios.post(GOOGLE_FORM_URL, payload.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
  } catch (err) {
    console.error("⚠️ Fehler beim Senden an Google Form:", err.message);
  }
}
