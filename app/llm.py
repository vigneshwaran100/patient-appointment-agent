"""GenAI LLM Integration Module supporting Gemini and Groq providers with grounded tool execution."""

from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

# Filter out Automatic Function Calling library warning and logger output from google-genai
warnings.filterwarnings("ignore", message=".*automatic function calling.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Automatic function calling.*", category=UserWarning)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def get_chat_model() -> BaseChatModel | None:
    """Initialize and return configured Chat Model (Groq or Gemini).

    Selection strategy:
    1. If LLM_PROVIDER is 'groq', use Groq if key provided.
    2. If LLM_PROVIDER is 'gemini', use Gemini if key provided.
    3. If LLM_PROVIDER is 'auto':
       - Prefers Groq if GROQ_API_KEY is available (high throughput, ultra-fast latency).
       - Falls back to Gemini if GEMINI_API_KEY is available.
    """
    settings = get_settings()
    provider = (settings.llm_provider or "auto").lower()

    # 1. Try Groq
    if provider in {"groq", "auto"} and settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=settings.temperature,
                max_retries=1,
                timeout=5.0,
            )
        except Exception as e:
            logger.warning("Failed to initialize Groq chat model: %s", e)
            if provider == "groq":
                return None

    # 2. Try Gemini
    if provider in {"gemini", "auto"} and settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.temperature,
                max_retries=1,
                timeout=5.0,
            )
        except Exception as e:
            logger.warning("Failed to initialize Gemini chat model: %s", e)
            if provider == "gemini":
                return None

    return None


def get_tool_calling_model() -> Any | None:
    """Return configured Chat Model bound with CLINIC_TOOLS for true LangGraph tool calling."""
    llm = get_chat_model()
    if not llm:
        return None
    from app.tools.registry import CLINIC_TOOLS

    return llm.bind_tools(CLINIC_TOOLS)


def llm_classify_and_extract(
    messages: list[dict[str, Any]], current_state: dict[str, Any]
) -> dict[str, Any] | None:
    """Use LLM (Gemini/Groq) to understand intent and extract clinical entities across dialogue history."""
    llm = get_chat_model()
    if not llm:
        return None

    system_prompt = (
        "You are an expert clinical scheduling AI assistant. Analyze the multi-turn dialogue between "
        "a clinic scheduling agent and a patient.\n\n"
        "Extract clinical scheduling entities and identify the user's intent in the context of the whole conversation.\n"
        "Available Doctors & Specialties:\n"
        "- DOC001: Dr. Alice Smith (Cardiology)\n"
        "- DOC002: Dr. Robert Chen (Dermatology)\n"
        "- DOC003: Dr. Maria Garcia (General Medicine)\n"
        "- DOC004: Dr. James Wilson (Pediatrics)\n"
        "- DOC005: Dr. Linda Taylor (Orthopedics)\n\n"
        "CRITICAL INTENTS:\n"
        "- 'general_inquiry': Greetings like 'Hi', 'Hello', or 'What can you help me with?'. Do NOT invoke DB tools.\n"
        "- 'register': User states they are a new patient, not registered, or is providing name/DOB/phone for registration.\n"
        "- 'check_availability': Inquiring about doctors, specialties, or open slots without confirming a specific booking (e.g., 'Do you have cardiology appointments tomorrow?', 'Do you have any cardiology appointments on 2026-10-10?', 'What slots are open?').\n"
        "- 'book': Intent to schedule/confirm an appointment.\n"
        "- 'reschedule': Changing an existing appointment time/doctor.\n"
        "- 'cancel': Cancelling an existing appointment.\n\n"
        "CRITICAL MULTI-TURN RULES:\n"
        "1. If the assistant previously asked for registration info (name, DOB, phone) and the user provides it, "
        "keep intent='register'.\n"
        "2. If the assistant asked for patient ID to book an appointment, and user answers with their ID, "
        "keep intent='book'.\n"
        "3. If user says 'tomorrow' or relative dates, identify the specialty/doctor and set is_ambiguous_datetime=false if a specific day is requested.\n\n"
        "Output ONLY a raw JSON object with the following schema, with no markdown code blocks:\n"
        "{\n"
        '  "intent": "book" | "check_availability" | "cancel" | "reschedule" | "register" | "general_inquiry" | "unclear",\n'
        '  "patient_id": string or null,\n'
        '  "phone": string or null,\n'
        '  "name": string or null,\n'
        '  "dob": string (YYYY-MM-DD or DD/MM/YYYY) or null,\n'
        '  "specialty": string or null,\n'
        '  "doctor_id": string or null,\n'
        '  "doctor_name": string or null,\n'
        '  "date": string (YYYY-MM-DD) or null,\n'
        '  "slot_datetime": string or null,\n'
        '  "appointment_id": string or null,\n'
        '  "is_ambiguous_datetime": boolean\n'
        "}"
    )

    conv_history = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            conv_history.append(f"Assistant: {content}")
        elif role == "user":
            conv_history.append(f"Patient: {content}")

    user_text = "\n".join(conv_history)
    current_intent = current_state.get("intent")
    context_note = f"\nCurrent Active Intent in State: {current_intent}" if current_intent else ""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Conversation Transcript:\n{user_text}{context_note}"),
            ]
        )
        content = response.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        return data
    except Exception as e:
        logger.debug("LLM extraction failed or timed out, falling back to rule-based parser: %s", e)
        return None


def llm_generate_natural_response(
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    fallback_text: str,
    policies: list[str],
) -> str:
    """Use LLM (Gemini/Groq) to generate empathetic, natural clinical responses strictly grounded in facts."""
    llm = get_chat_model()
    if not llm:
        return fallback_text

    intent = state.get("intent")

    system_prompt = (
        "You are 2Care AI , an empathetic, professional AI receptionist and clinic appointment scheduling assistant.\n"
        "Generate a clear, polite, concise, and contextual response to the patient.\n\n"
        "STRICT GROUNDING & CLINICAL SAFETY RULES:\n"
        "1. NEVER fabricate booking confirmations. You may ONLY confirm an appointment if the booking/reschedule result shows success=True.\n"
        "2. If booking was successful, clearly include the Appointment ID, Doctor, and Date/Time.\n"
        "3. If an appointment slot was unavailable, politely inform the patient and list only the alternative slots provided.\n"
        "4. If patient registration succeeded, confirm their registration and provide their new Patient ID, then ask what specialty they wish to see.\n"
        "5. For greetings or general questions ('Hi', 'What can you help me with?'), respond warmly and concisely explaining how you can help (availability, booking, rescheduling, cancellations, registration). Do NOT repeat the exact same static paragraph every time.\n"
        "6. NEVER provide medical advice, diagnosis, or prescribe medications.\n"
        "7. Strictly adhere to all Active Learned Policies provided below.\n"
    )

    facts = {
        "intent": intent,
        "verified_patient": state.get("patient"),
        "verification_status": state.get("verification"),
        "last_registration_result": state.get("last_registration_result"),
        "last_booking_result": state.get("last_booking_result"),
        "last_cancellation_result": state.get("last_cancellation_result"),
        "available_slots": state.get("available_slots"),
        "active_policies": policies,
        "required_factual_core": fallback_text,
    }

    user_msgs = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-4:]]

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "Dialogue Context:\n" + "\n".join(user_msgs) + "\n\n"
                        f"Factual State & Scoped Tool Results:\n{json.dumps(facts, default=str, indent=2)}\n\n"
                        "Respond to the patient naturally while strictly preserving the facts from the required factual core."
                    )
                ),
            ]
        )
        llm_text = response.content.strip()

        # Factual verification checks
        last_booking = state.get("last_booking_result")
        if last_booking and last_booking.get("success"):
            appt_id = last_booking["appointment"]["appointment_id"]
            if appt_id not in llm_text:
                return fallback_text

        last_reg = state.get("last_registration_result")
        if last_reg and last_reg.get("success"):
            pid = last_reg["patient"]["patient_id"]
            if pid not in llm_text:
                return fallback_text

        return llm_text
    except Exception as e:
        logger.debug("LLM response generation failed, using verified template: %s", e)
        return fallback_text
