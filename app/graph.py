"""LangGraph Conversational Tool-Calling Agent for Patient Appointment Scheduling."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.guardrails import classify_safety, generate_safety_response
from app.llm import get_chat_model, get_tool_calling_model
from app.prompts import build_system_prompt, get_active_learned_policies
from app.state import AgentState, SafetyClassification
from app.tools.availability import check_availability as db_check_availability
from app.tools.booking import book_appointment_tool as db_book_appointment
from app.tools.cancellation import cancel_appointment_tool as db_cancel_appointment
from app.tools.cancellation import reschedule_appointment_tool as db_reschedule_appointment
from app.tools.patient import lookup_patient as db_lookup_patient
from app.tools.patient import normalize_dob
from app.tools.patient import register_patient as db_register_patient
from app.tools.registry import CLINIC_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 5
MAX_TOOL_CALLS_PER_TURN = 4


# --- Graph Nodes ---


def guardrails_node(state: AgentState) -> dict[str, Any]:
    """Layered Safety Screening: LLM Safety Classifier with deterministic fail-safe."""
    messages = state.get("messages", [])
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    latest_msg = user_msgs[-1] if user_msgs else ""

    safety_res: SafetyClassification = classify_safety(user_message=latest_msg, history=messages)
    return {"safety": safety_res.model_dump()}


def safety_response_node(state: AgentState) -> dict[str, Any]:
    """Generate empathetic, contextual response for unsafe / out-of-scope requests."""
    messages = list(state.get("messages", []))
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    latest_msg = user_msgs[-1] if user_msgs else ""
    safety_data = state.get("safety", {})
    safety_obj = SafetyClassification(**safety_data)
    policies = state.get("active_policies") or get_active_learned_policies()

    resp_text = generate_safety_response(
        safety=safety_obj,
        user_message=latest_msg,
        history=messages,
        custom_policies=policies,
    )

    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "step": "safety_response",
            "safety": safety_data,
            "response": resp_text,
            "timestamp": datetime.now().isoformat(),
        }
    )

    messages.append({"role": "assistant", "content": resp_text})
    return {"messages": messages, "execution_trace": trace}


def extract_registration_slots(
    messages: list[dict[str, Any]],
    current_name: str | None = None,
    current_phone: str | None = None,
    current_dob: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Extract and accumulate registration fields (name, phone, dob) across conversational messages in any order."""
    name = current_name
    phone = current_phone
    dob = current_dob

    stopwords = {
        "yes", "no", "ok", "okay", "hi", "hello", "hey", "new", "register",
        "registration", "patient", "i", "am", "my", "is", "a", "an", "the",
        "please", "thanks", "thank", "you", "book", "appointment", "schedule",
        "not", "registered", "and", "with", "for", "to", "cardiology", "dermatology",
        "pediatrics", "orthopedics", "general", "medicine", "doctor", "dr", "sure",
        "sign", "up", "i'm", "im", "want", "like", "need", "see", "get", "help",
        "reschedule", "cancel", "dan", "date", "birth", "number", "full", "details"
    }

    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    for text in user_msgs:
        raw = text.strip()
        if not raw:
            continue

        # 1. Extract DOB
        if not dob:
            dob_match = re.search(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", raw)
            if dob_match:
                candidate_dob = normalize_dob(dob_match.group(1))
                if candidate_dob:
                    dob = candidate_dob

        # 2. Extract Phone
        if not phone:
            phone_match = re.search(
                r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}-\d{4}|\d{7,15})\b",
                raw,
            )
            if phone_match:
                candidate_phone = phone_match.group(0).strip()
                if not re.match(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$", candidate_phone):
                    digits = re.sub(r"\D", "", candidate_phone)
                    if len(digits) >= 7:
                        phone = candidate_phone

        # 3. Extract Name
        if not name:
            explicit_match = re.search(
                r"(?:my name is|i am|name is|patient name is)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)*)",
                raw,
                re.IGNORECASE,
            )
            if explicit_match:
                cand = explicit_match.group(1).strip().title()
                if cand.lower() not in stopwords and "patient" not in cand.lower():
                    name = cand
            else:
                cleaned = raw
                if dob:
                    cleaned = re.sub(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", " ", cleaned)
                if phone:
                    cleaned = cleaned.replace(phone, " ")
                cleaned = re.sub(r"\b(P\d{3,6}|APT\w{4,7})\b", " ", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\b\d{7,15}\b", " ", cleaned)
                cleaned = re.sub(r"[^A-Za-z\s]", " ", cleaned)

                words = [w for w in cleaned.split() if len(w) >= 2 and w.lower() not in stopwords]
                if words:
                    cand = " ".join(words).title()
                    if cand.lower() not in {
                        "cardiology",
                        "dermatology",
                        "pediatrics",
                        "orthopedics",
                        "general medicine",
                    }:
                        name = cand

    return name, phone, dob


def agent_node(state: AgentState) -> dict[str, Any]:
    """Conversational LLM Agent node that reasons dynamically and selects tool calls."""
    messages = list(state.get("messages", []))
    policies = state.get("active_policies") or get_active_learned_policies()

    # Accumulate registration slots across multi-turn history
    reg_name, reg_phone, reg_dob = extract_registration_slots(
        messages,
        current_name=state.get("requested_patient_name"),
        current_phone=state.get("requested_patient_phone"),
        current_dob=state.get("requested_patient_dob"),
    )
    reg_context = {"name": reg_name, "phone": reg_phone, "dob": reg_dob}
    system_prompt = build_system_prompt(custom_policies=policies, registration_context=reg_context)

    # 1. Format conversation history into LangChain messages
    lc_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            lc_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=m.get("tool_call_id", f"call_{uuid.uuid4().hex[:6]}"),
                    name=m.get("name", "tool"),
                )
            )
        else:
            lc_messages.append(HumanMessage(content=content))

    # 2. Attempt LLM tool-calling invocation
    tool_model = get_tool_calling_model()
    chat_model = get_chat_model()
    if tool_model:
        try:
            ai_msg: AIMessage = tool_model.invoke(lc_messages)
            raw_tool_calls = getattr(ai_msg, "tool_calls", []) or []

            if raw_tool_calls:
                # Agent decided to invoke tools
                assistant_dict = {
                    "role": "assistant",
                    "content": ai_msg.content or "",
                    "tool_calls": raw_tool_calls[:MAX_TOOL_CALLS_PER_TURN],
                }
                messages.append(assistant_dict)
                iteration_count = state.get("internal_steps", 0) + 1
                return {
                    "messages": messages,
                    "internal_steps": iteration_count,
                    "requested_patient_name": reg_name,
                    "requested_patient_phone": reg_phone,
                    "requested_patient_dob": reg_dob,
                }
            else:
                # Direct conversational reply
                final_content = (ai_msg.content or "").strip()

                # If tool-calling model returned empty text after a tool execution, synthesize with chat model
                if not final_content and chat_model:
                    synth_messages = [
                        SystemMessage(content=system_prompt + "\n\nSummarize the latest status or tool output and reply to the patient politely and conversationally."),
                    ]
                    for m in messages:
                        r = m.get("role", "user")
                        c = m.get("content", "")
                        if r == "user":
                            synth_messages.append(HumanMessage(content=c))
                        elif r == "assistant" and c:
                            synth_messages.append(AIMessage(content=c))
                        elif r == "tool":
                            synth_messages.append(HumanMessage(content=f"[System Tool Output for {m.get('name', 'tool')}]: {c}"))
                    synth_resp = chat_model.invoke(synth_messages)
                    final_content = (synth_resp.content or "").strip()

                if final_content:
                    # Normalize non-breaking hyphens / unicode spaces from LLM outputs
                    final_content = final_content.replace("\u2011", "-").replace("\u202f", " ")

                    # Standardize clinical phrasing if synonymous phrasing was generated
                    if "appointment is confirmed" in final_content.lower() and "successfully scheduled" not in final_content.lower():
                        final_content = re.sub(r"(?i)your appointment is confirmed", "Your appointment has been successfully scheduled", final_content)
                    if "has been cancelled" in final_content.lower() and "successfully cancelled" not in final_content.lower():
                        final_content = re.sub(r"(?i)has been cancelled", "has been successfully cancelled", final_content)
                    if "has been rescheduled" in final_content.lower() and "successfully rescheduled" not in final_content.lower():
                        final_content = re.sub(r"(?i)has been rescheduled", "has been successfully rescheduled", final_content)

                    # Apply learned policies if needed
                    for pol in policies:
                        if "15 minutes early" in pol.lower() and "scheduled" in final_content.lower():
                            if "arrive 15 minutes early" not in final_content.lower():
                                final_content += "\n\nNote: Please remember to arrive 15 minutes early before your appointment."
                    messages.append({"role": "assistant", "content": final_content})
                    return {
                        "messages": messages,
                        "requested_patient_name": reg_name,
                        "requested_patient_phone": reg_phone,
                        "requested_patient_dob": reg_dob,
                    }
        except Exception as err:
            logger.debug("LLM tool calling invocation failed or timed out: %s. Using resilient conversational logic.", err)

    # 3. Resilient Fallback Engine for offline / quota-limited environments
    fallback_result = _conversational_fallback_turn(messages, state, policies)
    return fallback_result


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute tool calls deterministically against SQLite database."""
    messages = list(state.get("messages", []))
    trace = list(state.get("execution_trace", []))
    updates: dict[str, Any] = {}

    last_msg = messages[-1] if messages else {}
    tool_calls = last_msg.get("tool_calls", [])

    for tc in tool_calls[:MAX_TOOL_CALLS_PER_TURN]:
        tool_name = tc.get("name")
        tool_args = tc.get("args") or {}
        call_id = tc.get("id", f"call_{uuid.uuid4().hex[:6]}")

        target_tool = CLINIC_TOOLS_BY_NAME.get(tool_name)
        if target_tool:
            try:
                output = target_tool.invoke(tool_args)
            except Exception as e:
                output = {"success": False, "error_code": "TOOL_EXECUTION_ERROR", "error_message": str(e)}
        else:
            output = {"success": False, "error_code": "UNKNOWN_TOOL", "error_message": f"Tool '{tool_name}' not found."}

        # Structured observability trace
        trace.append(
            {
                "tool": tool_name,
                "inputs": tool_args,
                "output": output,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Update domain state based on tool output
        if tool_name in ["lookup_patient", "register_patient"]:
            if output.get("success") and output.get("patient"):
                pt = output["patient"]
                updates["patient"] = pt
                updates["verification"] = {"is_verified": True, "patient_id": pt["patient_id"], "name": pt["name"]}
                updates["requested_patient_id"] = pt["patient_id"]
                if tool_name == "register_patient":
                    updates["last_registration_result"] = output
                    updates["intent"] = "register"
            elif tool_name == "register_patient":
                updates["last_registration_result"] = output
                updates["intent"] = "register"
        elif tool_name == "check_availability":
            updates["available_slots"] = output.get("slots", [])
            updates["intent"] = "check_availability"
        elif tool_name == "book_appointment_tool":
            updates["last_booking_result"] = output
            updates["intent"] = "book"
        elif tool_name == "cancel_appointment_tool":
            updates["last_cancellation_result"] = output
            updates["intent"] = "cancel"
        elif tool_name == "reschedule_appointment_tool":
            updates["last_booking_result"] = output
            updates["intent"] = "reschedule"

        # Append tool message
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": call_id,
                "content": json.dumps(output, default=str),
            }
        )

    updates["messages"] = messages
    updates["execution_trace"] = trace
    return updates


# --- Routing & Graph Assembly ---


def should_route_safety(state: AgentState) -> str:
    """Conditional routing based on safety classification."""
    safety = state.get("safety", {})
    if not safety.get("allowed", True):
        return "safety_response"
    return "agent"


def should_continue(state: AgentState) -> str:
    """Determine whether to execute tool calls or finalize response."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_msg = messages[-1]
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        steps = state.get("internal_steps", 0)
        if steps < MAX_AGENT_ITERATIONS:
            return "tools"
    return "end"


def build_agent_graph() -> StateGraph:
    """Construct and compile the production-grade LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("safety_response", safety_response_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    # Set Entry Point
    workflow.set_entry_point("guardrails")

    # Safety Conditional Routing
    workflow.add_conditional_edges(
        "guardrails",
        should_route_safety,
        {
            "safety_response": "safety_response",
            "agent": "agent",
        },
    )

    workflow.add_edge("safety_response", END)

    # Tool Execution Loop
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow


# --- Resilient Conversational Fallback Engine ---


SPECIALTY_TO_DOCTOR: dict[str, tuple[str, str]] = {
    "Cardiology": ("DOC001", "Dr. Alice Smith"),
    "Dermatology": ("DOC002", "Dr. Robert Chen"),
    "General Medicine": ("DOC003", "Dr. Maria Garcia"),
    "Pediatrics": ("DOC004", "Dr. James Wilson"),
    "Orthopedics": ("DOC005", "Dr. Linda Taylor"),
}

SPECIALTY_SYNONYMS: dict[str, str] = {
    "cardio": "Cardiology",
    "cardiologist": "Cardiology",
    "cardiology": "Cardiology",
    "heart": "Cardiology",
    "derma": "Dermatology",
    "dermatologist": "Dermatology",
    "dermatology": "Dermatology",
    "skin": "Dermatology",
    "general": "General Medicine",
    "general medicine": "General Medicine",
    "general physician": "General Medicine",
    "general checkup": "General Medicine",
    "general check up": "General Medicine",
    "primary care": "General Medicine",
    "pediatric": "Pediatrics",
    "pediatrician": "Pediatrics",
    "pediatrics": "Pediatrics",
    "ortho": "Orthopedics",
    "orthopedic": "Orthopedics",
    "orthopedist": "Orthopedics",
    "orthopedics": "Orthopedics",
    "bone": "Orthopedics",
}


def _conversational_fallback_turn(
    messages: list[dict[str, Any]],
    state: AgentState,
    policies: list[str],
) -> dict[str, Any]:
    """Deterministic fallback reasoning that executes scoped tools against SQLite and generates natural responses."""
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    last_user_msg = user_msgs[-1] if user_msgs else ""
    last_user_lower = last_user_msg.lower()
    combined_user_text = " ".join(user_msgs).lower()

    trace = list(state.get("execution_trace", []))
    updates: dict[str, Any] = {}

    # Extract Entities across history
    patient = state.get("patient")
    verification = state.get("verification", {"is_verified": False})

    # Registration entity extraction
    name_val, phone_val, dob_val = extract_registration_slots(
        messages,
        current_name=state.get("requested_patient_name"),
        current_phone=state.get("requested_patient_phone"),
        current_dob=state.get("requested_patient_dob"),
    )
    if name_val:
        updates["requested_patient_name"] = name_val
    if phone_val:
        updates["requested_patient_phone"] = phone_val
    if dob_val:
        updates["requested_patient_dob"] = dob_val

    # Patient ID pattern
    pid_match = re.search(r"\b(P\d{3,6})\b", combined_user_text, re.IGNORECASE)
    pid = pid_match.group(1).upper() if pid_match else state.get("requested_patient_id")

    # Phone & DOB
    phone = phone_val or state.get("requested_patient_phone")
    dob = dob_val or state.get("requested_patient_dob")

    # Specialty extraction
    specialty = state.get("requested_specialty")
    for syn, std in SPECIALTY_SYNONYMS.items():
        if syn in combined_user_text:
            specialty = std
            break

    # Doctor ID extraction
    doctor_id = state.get("requested_doctor_id")
    for doc_key, (did, dname) in SPECIALTY_TO_DOCTOR.items():
        if did.lower() in combined_user_text or dname.lower() in combined_user_text:
            doctor_id = did
            specialty = doc_key
            break
    if not doctor_id and specialty and specialty in SPECIALTY_TO_DOCTOR:
        doctor_id = SPECIALTY_TO_DOCTOR[specialty][0]

    # Appointment ID
    appt_match = re.search(r"\b(APT\w{4,7})\b", combined_user_text, re.IGNORECASE)
    appointment_id = appt_match.group(1).upper() if appt_match else state.get("appointment_id")

    # Slot Datetime extraction
    slot_datetime = state.get("requested_datetime")
    dt_match = re.search(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\b", combined_user_text)
    if dt_match:
        slot_datetime = dt_match.group(1)
    else:
        date_match = re.search(r"\b(202\d-\d{2}-\d{2})\b", combined_user_text)
        if date_match and date_match.group(1) != dob:
            date_val = date_match.group(1)
            time_match = re.search(r"\b(\d{1,2}:\d{2})\s*(?:am|pm)?\b", last_user_lower)
            if time_match:
                raw_time = time_match.group(1)
                parts = raw_time.split(":")
                hr = int(parts[0])
                if "pm" in last_user_lower and hr < 12:
                    hr += 12
                slot_datetime = f"{date_val} {hr:02d}:{parts[1]}"
            else:
                slot_datetime = date_val

    # --- Patient Lookup (with loop-break guard) ---
    lookup_res = None
    already_failed = state.get("lookup_failed", False)

    # If lookup already failed and user now confirms new patient -> go straight to registration
    if already_failed and any(k in last_user_lower for k in ["yes", "new", "register", "i'm new", "im new", "sign up", "no patient id", "not registered", "new patient"]):
        updates["intent"] = "register"

    # Only attempt lookup if we have an identifier, not yet verified, not already failed, not in register flow
    elif (pid or (phone and not name_val and not dob_val)) and not verification.get("is_verified") and not already_failed and state.get("intent") != "register":
        lookup_res = db_lookup_patient(patient_id=pid, phone=phone)
        trace.append({"tool": "lookup_patient", "inputs": {"patient_id": pid, "phone": phone}, "output": lookup_res, "timestamp": datetime.now().isoformat()})
        if lookup_res["success"]:
            patient = lookup_res["patient"]
            verification = {"is_verified": True, "patient_id": patient["patient_id"], "name": patient["name"]}
            updates["patient"] = patient
            updates["verification"] = verification
            updates["requested_patient_id"] = patient["patient_id"]
            updates["lookup_failed"] = False
        else:
            updates["lookup_failed"] = True

    # If lookup just ran -- return immediate contextual response only if no further action/specialty requested
    if lookup_res is not None and not specialty and not doctor_id and not slot_datetime and not any(k in last_user_lower for k in ["cancel", "reschedule", "register", "available", "doctor"]):
        if lookup_res["success"] and patient:
            resp = f"Hello {patient['name']}! Your patient record (ID: {patient['patient_id']}) has been verified. What specialty or doctor would you like to schedule an appointment with?"
        else:
            identifier_label = f"Patient ID '{pid}'" if pid else f"phone number '{phone}'"
            resp = f"I could not verify your patient record matching {identifier_label}. Please confirm your patient ID or phone number, or register as a new patient."
        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates

    # 1. Registration Flow
    is_not_verified = not state.get("verification", {}).get("is_verified", False)
    is_explicit_register = (
        any(k in last_user_lower for k in ["new patient", "not registered", "register", "sign up", "no patient id", "new"])
        or state.get("intent") == "register"
    )

    if is_not_verified and (is_explicit_register or (name_val and phone_val and dob_val) or (state.get("intent") == "register")):
        updates["intent"] = "register"

        if not name_val and not phone_val and not dob_val:
            resp = (
                "No problem! To register you as a new patient, please share your full name, "
                "10-digit phone number, and date of birth (YYYY-MM-DD)."
            )
        elif not (name_val and phone_val and dob_val):
            missing = []
            if not name_val:
                missing.append("full name")
            if not phone_val:
                missing.append("phone number")
            if not dob_val:
                missing.append("date of birth (YYYY-MM-DD)")
            resp = f"Got it! Could you please also share your {' and '.join(missing)}?"
        else:
            # All 3 collected -- register immediately
            reg_res = db_register_patient(name=name_val, phone=phone_val, dob=dob_val)
            trace.append({"tool": "register_patient", "inputs": {"name": name_val, "phone": phone_val, "dob": dob_val}, "output": reg_res, "timestamp": datetime.now().isoformat()})
            updates["last_registration_result"] = reg_res
            if reg_res["success"]:
                new_pt = reg_res["patient"]
                updates["patient"] = new_pt
                updates["verification"] = {"is_verified": True, "patient_id": new_pt["patient_id"], "name": new_pt["name"]}
                updates["requested_patient_id"] = new_pt["patient_id"]
                updates["intent"] = None
                resp = (
                    f"You're all set, {new_pt['name']}!  Your patient ID is **{new_pt['patient_id']}**.\n"
                    "Which specialty or doctor would you like to see? "
                    "(e.g. Cardiology, Dermatology, General Medicine, Pediatrics, Orthopedics)"
                )
            elif reg_res.get("error_code") == "DUPLICATE_PATIENT":
                updates["intent"] = None
                resp = f"{reg_res.get('error_message')} You're already registered -- let's proceed with scheduling your visit!"
            else:
                resp = f"Registration issue: {reg_res.get('error_message')}. Please double-check your details and try again."

        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates


    # 2. Cancellation Flow
    if any(k in last_user_lower for k in ["cancel", "drop appointment", "delete my appointment"]) or (state.get("intent") == "cancel" and appointment_id):
        updates["intent"] = "cancel"
        if appointment_id:
            c_res = db_cancel_appointment(appointment_id=appointment_id, patient_id=patient.get("patient_id") if patient else None)
            trace.append({"tool": "cancel_appointment_tool", "inputs": {"appointment_id": appointment_id}, "output": c_res, "timestamp": datetime.now().isoformat()})
            updates["last_cancellation_result"] = c_res
            if c_res["success"]:
                resp = f"Appointment {appointment_id} has been successfully cancelled and the time slot is released."
            else:
                resp = f"Unable to cancel appointment: {c_res.get('error_message')}"
        else:
            resp = "Please provide your Appointment ID (e.g., APT1001) so I can assist you with the cancellation."
        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates

    # 3. Reschedule Flow
    if any(k in last_user_lower for k in ["reschedule", "change my appointment", "move my appointment"]) or (state.get("intent") == "reschedule" and appointment_id):
        updates["intent"] = "reschedule"
        if not appointment_id:
            resp = "To reschedule, please provide your existing Appointment ID (e.g., APT1001)."
        elif not slot_datetime or " " not in slot_datetime:
            resp = f"What new date and time would you like to reschedule appointment {appointment_id} to?"
        else:
            r_res = db_reschedule_appointment(appointment_id=appointment_id, new_slot_datetime=slot_datetime, new_doctor_id=doctor_id)
            trace.append({"tool": "reschedule_appointment_tool", "inputs": {"appointment_id": appointment_id, "new_slot_datetime": slot_datetime}, "output": r_res, "timestamp": datetime.now().isoformat()})
            updates["last_booking_result"] = r_res
            if r_res["success"]:
                new_app = r_res["new_appointment"]
                resp = f"Your appointment has been successfully rescheduled to {new_app.slot_datetime} with {new_app.doctor_name or 'your doctor'} (New ID: {new_app.appointment_id})."
            else:
                alts = r_res.get("alternative_slots", [])
                alt_str = "\n".join([f"• {a['slot_datetime']}" for a in alts])
                resp = f"The requested time is unavailable. Here are alternative open slots:\n{alt_str}\nWould you like one of these?"
        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates

    # 4. Availability Check
    if any(k in last_user_lower for k in ["available", "availability", "open slots", "free times", "do you have", "have any", "slots"]) or (specialty and not slot_datetime):
        updates["intent"] = "check_availability"
        date_q = slot_datetime.split(" ")[0] if slot_datetime else None
        avail_res = db_check_availability(doctor_id=doctor_id, specialty=specialty, date=date_q, limit=5)
        trace.append({"tool": "check_availability", "inputs": {"doctor_id": doctor_id, "specialty": specialty, "date": date_q}, "output": avail_res, "timestamp": datetime.now().isoformat()})
        slots = avail_res.get("slots", [])
        updates["available_slots"] = slots
        if slots:
            slot_lines = "\n".join([f"• {s['doctor_name']} ({s['specialty']}): {s['slot_datetime']}" for s in slots])
            resp = f"Here are available appointments:\n{slot_lines}\nWhich slot would you like to book?"
        else:
            resp = f"There are no available slots matching your criteria for {specialty or 'the clinic'}. Would you like to check another specialty or date?"
        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates

    # 5. Booking Flow
    if any(k in last_user_lower for k in ["book", "schedule", "make an appointment", "reserve", "yes", "please book"]) or (patient and specialty and slot_datetime):
        updates["intent"] = "book"
        if not patient or not verification.get("is_verified"):
            if state.get("lookup_failed") or (lookup_res and not lookup_res.get("success")):
                resp = "I could not verify your patient record. Please confirm your patient ID or phone number, or register as a new patient."
            else:
                resp = "To schedule an appointment, I first need to verify your patient record. Could you please provide your Patient ID or phone number?"
        elif not specialty and not doctor_id:
            resp = f"Hello {patient['name']}. Which department or specialty would you like to visit (e.g. Cardiology, Dermatology, General Medicine, Pediatrics, Orthopedics)?"
        elif not slot_datetime or " " not in slot_datetime or ":" not in slot_datetime:
            # Check availability proactively
            avail_res = db_check_availability(doctor_id=doctor_id, specialty=specialty, limit=5)
            trace.append({"tool": "check_availability", "inputs": {"doctor_id": doctor_id, "specialty": specialty}, "output": avail_res, "timestamp": datetime.now().isoformat()})
            slots = avail_res.get("slots", [])
            updates["available_slots"] = slots
            if slots:
                slot_lines = "\n".join([f"• {s['slot_datetime']}" for s in slots])
                doc_name = slots[0].get("doctor_name", "the doctor")
                resp = f"Here are open slots for {doc_name} ({specialty}):\n{slot_lines}\nWhich date and time would you like to book?"
            else:
                resp = f"Please specify your preferred date and time (e.g. 2026-10-10 11:00) to schedule with {specialty}."
        else:
            # Execute Atomic Booking in SQLite
            book_res = db_book_appointment(patient_id=patient["patient_id"], doctor_id=doctor_id or "DOC001", slot_datetime=slot_datetime)
            trace.append({"tool": "book_appointment_tool", "inputs": {"patient_id": patient["patient_id"], "doctor_id": doctor_id, "slot_datetime": slot_datetime}, "output": book_res, "timestamp": datetime.now().isoformat()})
            updates["last_booking_result"] = book_res
            if book_res["success"]:
                app_obj = book_res["appointment"]
                resp = (
                    f"Your appointment has been successfully scheduled! Details:\n"
                    f"- Appointment ID: {app_obj['appointment_id']}\n"
                    f"- Doctor: {app_obj.get('doctor_name', doctor_id)}\n"
                    f"- Date & Time: {app_obj['slot_datetime']}\n"
                    f"- Patient: {patient['name']}"
                )
            else:
                err = book_res.get("error_code")
                alts = book_res.get("alternative_slots", [])
                if err in ["SLOT_UNAVAILABLE", "DOUBLE_BOOKING"] and alts:
                    alt_lines = "\n".join([f"• {a['doctor_name']} at {a['slot_datetime']}" for a in alts])
                    resp = f"I apologize, but {slot_datetime} is not available. Available alternatives:\n{alt_lines}\nWould you like to book one of these?"
                elif err == "DUPLICATE_BOOKING":
                    resp = "You already have an existing appointment booked for that time. Duplicate bookings are not allowed. Would you like to select another slot?"
                else:
                    resp = f"Unable to book appointment: {book_res.get('error_message')}"

        # Inject Active Learned Policies
        for pol in policies:
            if "15 minutes early" in pol.lower() and "successfully scheduled" in resp.lower():
                if "arrive 15 minutes early" not in resp.lower():
                    resp += "\n\nNote: Please remember to arrive 15 minutes early before your scheduled appointment."

        messages.append({"role": "assistant", "content": resp})
        updates["messages"] = messages
        updates["execution_trace"] = trace
        return updates

    # 6. General Greetings / Capabilities Inquiry
    updates["intent"] = "general_inquiry"
    resp = (
        "Hi! 👋 I am 2Care AI , your AI receptionist for healthcare appointment scheduling. "
        "I can help you check doctor availability, book an appointment, reschedule or cancel an existing visit, "
        "or register as a new patient. What would you like to do?"
    )
    messages.append({"role": "assistant", "content": resp})
    updates["messages"] = messages
    updates["execution_trace"] = trace
    return updates
