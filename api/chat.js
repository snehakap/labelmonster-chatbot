import fs from "fs";
import path from "path";
import natural from "natural";
import OpenAI from "openai";

// -------------------- Wissensdatenbank laden --------------------
let knowledge = [];
try {
  const data = fs.readFileSync(path.join(process.cwd(), "api/knowledge.json"), "utf8");
  knowledge = JSON.parse(data);
} catch (err) {
  console.warn("⚠️ knowledge.json nicht gefunden. Verwende leere Wissensdatenbank.");
}

// -------------------- Hilfsfunktionen --------------------
function extractKeywords(text) {
  const tokenizer = new natural.WordTokenizer();
  const words = tokenizer.tokenize(text.toLowerCase());
  const stopwords = ["i","want","do","you","is","the","a","an","und","die","das","der"];
  return words.filter(w => !stopwords.includes(w));
}

function findRelevantKnowledge(question) {
  const keywords = extractKeywords(question);
  let bestMatch = null;
  let bestScore = 0;

  for (const entry of knowledge) {
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

function saveChat(userMsg, botReply) {
  const log = { timestamp: new Date().toISOString(), user: userMsg, bot: botReply };
  fs.appendFile("chatlogs.json", JSON.stringify(log) + "\n", (err) => {
    if (err) console.error("❌ Fehler beim Speichern des Chats:", err);
  });
}

// -------------------- Hugging Face Client Setup --------------------
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_API_KEY,
});

// -------------------- Vercel Handler --------------------
export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).send("Method Not Allowed");
    return;
  }

  const { message } = req.body;
  if (!message) return res.json({ reply: "Keine Nachricht erhalten." });

  try {
    const matchedEntry = findRelevantKnowledge(message);

    if (!matchedEntry) {
      const fallback =
        "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <a href='mailto:info@labelmonster.eu'>info@labelmonster.eu</a>.";
      saveChat(message, fallback);
      return res.json({ reply: fallback });
    }

    const prompt = `
You are the official assistant of Labelmonster.

Your ONLY task:
Use **only** the provided "Knowledge Base Entry" to answer the user's question. 
Understand the context of the user's question and look for similar questions in the "Knowledge Base Entry" to answer.
You are NOT allowed to add, guess, or invent information beyond what is in the Answer field.
If the provided answer already fits, repeat it naturally — do not rephrase or expand beyond minor adjustments for fluency.

Knowledge Base Entry:
Patterns: ${matchedEntry.patterns.join(", ")}
Answer: ${matchedEntry.answer}

Instructions:
- Answer **only** from the "Answer" field above.
- NEVER add external or unrelated information.
- Always answer in **German**, even if the question is in another language.
- If the question is about you – even indirectly – assume the user means Labelmonster. Provide the answer only from the 'Answer' field.
- Write in a professional, natural, and friendly tone suitable for a company chatbot.
- Do NOT start the reply with phrases like "Antwort:", "Answer:", or quotes.
- Any user question related to address or location should be "Großenbaumer Allee 98, 47269 Duisburg".
- If the you are not confident or the information is not in the "Answer", do not attempt to answer; instead, say exactly:
  "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <a href='mailto:info@labelmonster.eu'>info@labelmonster.eu</a>, damit wir Ihnen besser weiterhelfen können."

User Question: ${message}

Antwort (strictly based on knowledge base):
`;

    const chatCompletion = await client.chat.completions.create({
      model: "google/gemma-2-2b-it:nebius",
      messages: [{ role: "user", content: prompt }],
    });

    const cleanReply = chatCompletion.choices?.[0]?.message?.content?.trim() || "Keine Antwort gefunden.";
    saveChat(message, cleanReply);
    res.json({ reply: cleanReply });

  } catch (err) {
    console.error("❌ Hugging Face API Error:", err.message);
    res.json({ reply: "Fehler beim Abrufen der Antwort." });
  }
}
