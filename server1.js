import express from "express";
import dotenv from "dotenv";
import path from "path";
import axios from "axios";
import fs from "fs";

import { ChatMistralAI } from "@langchain/mistralai";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { z } from "zod";

dotenv.config();

const port = process.env.PORT || 3000;
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

const __dirname = path.resolve();

// Global Maps for state management
const chatModes = new Map(); // Tracks "AI" vs "HUMAN" mode per sender
const conversationHistories = new Map(); // Tracks chat memory per sender/session

// Helper: Get or initialize chat history
function getHistory(sessionId) {
  if (!conversationHistories.has(sessionId)) {
    conversationHistories.set(sessionId, [
      new SystemMessage(
        `You are the official digital concierge for "Salt & Sage Restaurant".
        
        Guidelines:
        - Welcome guests warmly and professionally.
        - Always answer questions regarding menu items, prices, opening hours, specials, dietary options, delivery policies, and contact info using official document information via 'getSaltAndSageInfoTool'.
        - When a customer wishes to place an order or make a reservation, collect their: Name, Contact Number, Order Details/Booking details, and Delivery Address (if applicable).
        - NEVER call 'processOrderOrBookingTool' with empty or blank values for customerName, contactNumber, or details. Collect them step-by-step first.
        - As soon as you have full Name, Contact Number, and Details/Address, immediately invoke 'processOrderOrBookingTool'.`
      ),
    ]);
  }
  return conversationHistories.get(sessionId);
}

// Helper Function: Send outbound message via Meta WhatsApp Cloud API
async function sendMetaWhatsAppMessage(toPhone, textContent) {
  try {
    const cleanPhone = String(toPhone).replace(/\D/g, "");
    const url = `https://graph.facebook.com/v20.0/${process.env.META_PHONE_NUMBER_ID}/messages`;
    
    const response = await axios.post(
      url,
      {
        messaging_product: "whatsapp",
        recipient_type: "individual",
        to: cleanPhone,
        type: "text",
        text: { body: textContent },
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.META_WHATSAPP_TOKEN}`,
          "Content-Type": "application/json",
        },
      }
    );
    console.log(`✅ WhatsApp message sent to +${cleanPhone}`);
    return response.data;
  } catch (error) {
    console.error("❌ Meta API Error:", error.response?.data || error.message);
  }
}

// 1. Initialize Mistral Model
const model = new ChatMistralAI({
  modelName: "mistral-small-latest",
  temperature: 0.3,
  apiKey: process.env.MISTRAL_API_KEY,
});

// 2. Global Variable for Restaurant Context
let saltAndSageKnowledgeBase = "";

// 3. Load PDF Document at Startup
async function loadRestaurantDocument(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      console.error(`❌ File not found at path: ${filePath}`);
      console.log("👉 Please make sure 'Salt_and_Sage_Restaurant_Dataset.pdf' is inside the 'data' folder.");
      return;
    }

    const loader = new PDFLoader(filePath);
    const docs = await loader.load();
    saltAndSageKnowledgeBase = docs.map((doc) => doc.pageContent).join("\n\n");
    console.log("✅ Salt & Sage Restaurant knowledge base loaded successfully from PDF!");
  } catch (err) {
    console.error("❌ Failed to load PDF document:", err);
  }
}

// 4. Tool 1: Information Retriever
const getSaltAndSageInfoTool = new DynamicStructuredTool({
  name: "getSaltAndSageInfoTool",
  description:
    "Searches and retrieves official details about Salt & Sage Restaurant, including menu items, pricing, specials, opening/closing hours, location, contact info, and house policies.",
  schema: z.object({
    query: z.string().describe("The user's question or topic of interest."),
  }),
  func: async ({ query }) => {
    if (!saltAndSageKnowledgeBase) {
      return "Knowledge base not initialized. Please ensure the document is uploaded.";
    }
    return saltAndSageKnowledgeBase;
  },
});

// 5. Tool 2: Order & Reservation Tool
const processOrderOrBookingTool = new DynamicStructuredTool({
  name: "processOrderOrBookingTool",
  description:
    "Handles customer table reservations or food orders for Salt & Sage Restaurant once key details are collected.",
  schema: z.object({
    type: z.enum(["reservation", "food_order"]).describe("Whether this is a table reservation or a delivery/takeaway order."),
    customerName: z.string().describe("Customer's name."),
    contactNumber: z.string().describe("Customer's phone/WhatsApp number."),
    details: z.string().describe("Order details (items & quantities) or reservation details (date, time, number of guests)."),
    deliveryAddress: z.string().optional().describe("Physical delivery address if the order is for home delivery."),
  }),
  func: async (args) => {
    const { type, customerName, contactNumber, details, deliveryAddress } = args;

    // Guard Clause: Check for missing fields before executing tool logic
    if (!customerName || customerName.trim() === "" || !contactNumber || contactNumber.trim() === "" || !details || details.trim() === "") {
      return "ERROR: Missing required fields. Please collect the customer's full Name, Contact Number, and Order/Booking details before invoking this tool.";
    }

    const referenceId = "SS-" + Math.floor(100000 + Math.random() * 900000);
    const managerPhone = process.env.MANAGER_WHATSAPP_NUMBER;

    console.log("📌 Tool Invoked with Args:", args);

    let confirmation = "";
    let managerNotification = "";

    if (type === "reservation") {
      confirmation = `✅ *Reservation Confirmed!*\n- Reference ID: ${referenceId}\n- Name: ${customerName}\n- Contact: ${contactNumber}\n- Details: ${details}\n\nWe look forward to welcoming you at Salt & Sage Restaurant!`;

      managerNotification = `📌 *NEW TABLE RESERVATION RECEIVED*\n\n` +
                            `🆔 *Ref ID:* ${referenceId}\n` +
                            `👤 *Name:* ${customerName}\n` +
                            `📞 *Contact:* ${contactNumber}\n` +
                            `📝 *Details:* ${details}`;
    } else {
      confirmation = `🎉 *Order Placed Successfully!*\n- Order ID: ${referenceId}\n- Name: ${customerName}\n- Items: ${details}\n`;
      if (deliveryAddress) {
        confirmation += `- Address: ${deliveryAddress}\n- Estimated Delivery Time: 40–55 minutes.`;
      } else {
        confirmation += `- Type: Takeaway / Dine-In`;
      }

      managerNotification = `🚨 *NEW FOOD ORDER RECEIVED*\n\n` +
                            `🆔 *Order ID:* ${referenceId}\n` +
                            `👤 *Customer Name:* ${customerName}\n` +
                            `📞 *Contact:* ${contactNumber}\n` +
                            `🍔 *Items:* ${details}\n` +
                            `📍 *Address:* ${deliveryAddress || "Takeaway / Dine-In"}`;
    }

    // Direct Alert to Manager
    if (managerPhone && managerPhone.trim() !== "") {
      console.log(`📲 Configured Manager Phone: ${managerPhone}`);
      console.log(`📡 Attempting to dispatch alert to Manager (+${managerPhone})...`);
      await sendMetaWhatsAppMessage(managerPhone, managerNotification);
    } else {
      console.error("❌ CRITICAL: MANAGER_WHATSAPP_NUMBER is missing in .env!");
    }

    return confirmation;
  },
});

const allTools = [getSaltAndSageInfoTool, processOrderOrBookingTool];
const modelWithTools = model.bindTools(allTools);

// --- MEMORY-AWARE EXECUTION PIPELINE ---
async function processUserMessage(input, sessionId = "web_default") {
  try {
    const history = getHistory(sessionId);
    history.push(new HumanMessage(input));

    // First LLM Turn: Determine Intent and Select Tool
    const response = await modelWithTools.invoke(history);
    history.push(response);

    // Check Tool Calls
    if (response.tool_calls && response.tool_calls.length > 0) {
      const toolCall = response.tool_calls[0];
      const targetedTool = allTools.find((t) => t.name === toolCall.name);

      if (targetedTool) {
        const toolResult = await targetedTool.invoke(toolCall.args);

        const toolMessage = new ToolMessage({
          content: String(toolResult),
          tool_call_id: toolCall.id,
          name: toolCall.name,
        });
        history.push(toolMessage);

        // Second LLM Turn: Format Result Cleanly
        const finalResponse = await modelWithTools.invoke(history);
        history.push(finalResponse);

        return finalResponse.content;
      }
    }

    return response.content || "Processing your request...";
  } catch (error) {
    console.error("❌ AI Pipeline Execution Error:", error);
    throw error;
  }
}

// --- ROUTES ---

app.get("/", (req, res) => {
  return res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Web UI Route
app.post("/api/chat", async (req, res) => {
  try {
    const { input, sessionId } = req.body;
    const output = await processUserMessage(input, sessionId || "web_user");
    return res.json({ output });
  } catch (error) {
    console.error("Error in concierge assistant endpoint:", error.message);
    res.status(500).json({ error: "An error occurred while processing your request. Please check API keys." });
  }
});

// --- META WHATSAPP WEBHOOK ROUTES ---

// 1. GET Webhook Verification
app.get("/api/whatsapp/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === process.env.META_VERIFY_TOKEN) {
    console.log("✅ Meta Webhook verified successfully!");
    return res.status(200).send(challenge);
  }
  res.sendStatus(403);
});

// 2. POST Webhook Receiver for WhatsApp Messages
app.post("/api/whatsapp/webhook", async (req, res) => {
  res.sendStatus(200);

  const entry = req.body.entry?.[0];
  const changes = entry?.changes?.[0];
  const message = changes?.value?.messages?.[0];

  if (!message || message.type !== "text") return;

  const senderPhone = message.from;
  const userText = message.text.body.trim();
  const managerPhone = process.env.MANAGER_WHATSAPP_NUMBER;

  if (!chatModes.has(senderPhone)) {
    chatModes.set(senderPhone, "AI");
  }

  // A. COMMAND: TOGGLE AI MODE BACK ON
  if (userText.toLowerCase() === "/ai" || userText.toLowerCase() === "/bot") {
    chatModes.set(senderPhone, "AI");
    await sendMetaWhatsAppMessage(senderPhone, "🤖 *AI Concierge Resumed:* I am back online to assist you!");
    return;
  }

  // B. COMMAND: HUMAN TAKEOVER
  if (userText.toLowerCase() === "/human") {
    chatModes.set(senderPhone, "HUMAN");

    await sendMetaWhatsAppMessage(
      senderPhone,
      "🛎️ *Request Received!* I have connected you with our restaurant manager. They will respond shortly."
    );

    if (managerPhone && senderPhone !== managerPhone) {
      await sendMetaWhatsAppMessage(
        managerPhone,
        `🚨 *HUMAN TAKEOVER REQUEST*\n\n` +
        `👤 *Customer:* +${senderPhone}\n` +
        `💬 *Message:* "${userText}"\n\n` +
        `👉 *Reply directly to assist.* Type /ai to hand back to bot.`
      );
    }
    return;
  }

  // C. IF MODE IS "HUMAN"
  if (chatModes.get(senderPhone) === "HUMAN") {
    if (managerPhone && senderPhone !== managerPhone) {
      await sendMetaWhatsAppMessage(
        managerPhone,
        `💬 *Message from +${senderPhone}:*\n"${userText}"`
      );
    }
    return;
  }

  // D. AI MODE
  try {
    const aiResult = await processUserMessage(userText, senderPhone);
    await sendMetaWhatsAppMessage(senderPhone, String(aiResult));
  } catch (err) {
    console.error("WhatsApp AI Processing Error:", err);
  }
});

// --- SERVER INITIALIZATION ---
app.listen(port, async () => {
  console.log(`Salt & Sage Concierge Server active on http://localhost:${port}`);

  const docPath = path.join(__dirname, "data", "Salt_and_Sage_Restaurant_Dataset.pdf");
  await loadRestaurantDocument(docPath);
});