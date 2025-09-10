#general_chat_section_routing.py
"""
SIMPLIFIED general_chat_section_routing.py

Now ONLY handles:
1. General chat (when current_section is None)
2. Initial routing TO sections (sets current_section and next_action)

All section-to-section routing, exit detection, and internal section logic
is now handled by the section nodes themselves.
"""

import os
import json
import re
from typing import Any, Dict, Optional, List, Tuple
from difflib import SequenceMatcher

# litellm (sync/async wrappers) - use completion or acompletion based on your runtime preference
from litellm import completion, acompletion

#local file imports
from agents.resume_builder_state import ResumeBuilderState

# -----------------------
# Constants
# -----------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OFFLINE_MODE = GOOGLE_API_KEY is None
MAX_ROUTING_ATTEMPTS = 3  # Prevent infinite loops

def safe_extract_text(resp: Any) -> Optional[str]:
    """Extract the assistant text from common litellm ModelResponse shapes."""
    try:
        # Handle ModelResponse objects directly
        if hasattr(resp, "choices") and resp.choices:
            c0 = resp.choices[0]
            if hasattr(c0, "message") and c0.message:
                return getattr(c0.message, "content", None)
        # Handle streaming chunk-like objects or other common structures
        if hasattr(resp, "choices") and resp.choices:
            c0 = resp.choices[0]
            # Check for delta content first (common in streaming)
            if hasattr(c0, "delta") and c0.delta:
                return getattr(c0.delta, "content", None)
            # Fallback to message content if delta is not present
            if hasattr(c0, "message") and c0.message:
                return getattr(c0.message, "content", None)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # Handle dictionary responses
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and isinstance(ch["message"], dict):
                        return ch["message"].get("content")
            if "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("content")
            if "content" in resp: # Direct content in dict
                return resp.get("content")
    except Exception as e:
        # Log the exception if any error occurs during extraction
        print(f"Warning: Error extracting text from response: {e}")
        pass
    # Return None if content cannot be extracted
    return None

def maybe_print_usage(resp: Any, label: str = ""):
    """Best-effort token usage printer for common shapes returned by litellm."""
    try:
        # 1) ModelResponse.usage
        if hasattr(resp, "usage") and resp.usage:
            u = resp.usage
            prompt = getattr(u, "prompt_tokens", getattr(u, "prompt_token_count", None))
            completion_t = getattr(u, "completion_tokens", getattr(u, "candidates_token_count", None))
            total = getattr(u, "total_tokens", getattr(u, "total_token_count", None))
            print(f"[Token Usage {label}] prompt={prompt} completion={completion_t} total={total}")
            return
        # 2) dict usage
        if isinstance(resp, dict) and "usage" in resp:
            u = resp["usage"]
            print(f"[Token Usage {label}] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={u.get('total_tokens')}")
    except Exception as e:
        # Log any errors during usage printing
        print(f"Warning: Error printing token usage: {e}")
        pass

def normalize_section_name(section_name: str, available_sections: List[str]) -> Optional[str]:
    """
    Normalize and match section names with fuzzy matching for typos.
    Returns the correct section name or None if no good match found.
    """
    if not section_name or not available_sections:
        return None
    
    # Clean the input
    clean_input = section_name.lower().strip()
    
    # Exact match first
    if clean_input in available_sections:
        return clean_input
    
    # Common variations and aliases
    section_aliases = {
        'skill': 'skills',
        'experience': 'experiences', 
        'exp': 'experiences',
        'work': 'experiences',
        'edu': 'education',
        'school': 'education',
        'project': 'projects',
        'cert': 'certificates',
        'certification': 'certificates',
        'certs': 'certificates',
        'pub': 'publications',
        'publication': 'publications',
        'papers': 'publications',
        'lang': 'languages',
        'language': 'languages',
        'rec': 'recommendations',
        'recommendation': 'recommendations',
        'refs': 'recommendations',
        'references': 'recommendations',
        'contact': 'contact',
        'contacts': 'contact',
        'summary': 'summary',
        'about': 'summary',
        'custom': 'custom',
        'other': 'custom',
        'additional': 'custom'
    }
    
    # Check aliases
    if clean_input in section_aliases:
        canonical = section_aliases[clean_input]
        if canonical in available_sections:
            return canonical
    
    # Fuzzy matching for typos (similarity > 0.6)
    best_match = None
    best_similarity = 0.0
    
    for section in available_sections:
        similarity = SequenceMatcher(None, clean_input, section).ratio()
        if similarity > best_similarity and similarity > 0.6:  # 60% similarity threshold
            best_similarity = similarity
            best_match = section
    
    return best_match

def extract_and_validate_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract JSON from raw text and validate basic structure.
    Raises ValueError if no valid JSON found or required keys missing.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from LLM")
    
    # Try to find JSON block in response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
    if not json_match:
        # If no JSON found, treat entire text as answer
        return {"action": "answer", "route": None, "answer": raw_text.strip()}
    
    json_text = json_match.group(0)
    try:
        parsed = json.loads(json_text)
        
        # Validate required keys
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        
        if "action" not in parsed:
            raise ValueError("Missing required 'action' key in JSON response")
        
        # Valid actions for general chat only
        valid_actions = {"answer", "route"}
        if parsed["action"] not in valid_actions:
            print(f"Warning: Unexpected action '{parsed['action']}', treating as 'answer'")
            parsed["action"] = "answer"
        
        return parsed
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def safe_initialize_answers(state: ResumeBuilderState, section_name: str, questions: List[str]) -> None:
    """Safely initialize answers array for a section with proper synchronization."""
    if section_name not in state.recommended_answers:
        state.recommended_answers[section_name] = [""] * len(questions)
    else:
        # Ensure array length matches questions count
        current_answers = state.recommended_answers[section_name]
        if len(current_answers) != len(questions):
            # Resize array, preserving existing answers
            new_answers = [""] * len(questions)
            for i in range(min(len(current_answers), len(questions))):
                new_answers[i] = current_answers[i]
            state.recommended_answers[section_name] = new_answers

def detect_question_matches(user_answer: str, questions: List[str]) -> List[Tuple[int, str, float]]:
    """
    Detect which questions the user's answer might be responding to using keyword matching.
    Returns list of (question_index, question_text, confidence_score) sorted by confidence.
    """
    matches = []
    user_words = set(user_answer.lower().split())
    
    for idx, question in enumerate(questions):
        question_words = set(question.lower().split())
        
        # Calculate keyword overlap
        common_words = user_words & question_words
        if len(common_words) > 0:
            # Simple confidence based on word overlap ratio
            confidence = len(common_words) / max(len(question_words), 1)
            matches.append((idx, question, confidence))
    
    # Sort by confidence (highest first)
    return sorted(matches, key=lambda x: x[2], reverse=True)

async def call_llm_json_decision(system_prompt: str, user_payload: Dict[str, Any], max_tokens: int = 300) -> Dict[str, Any]:
    """
    Call the LLM (async) and expect JSON decision output with robust error handling.
    """
    # Build message: use a compact JSON payload to keep tokens small
    user_text = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    extra = {}
    if GOOGLE_API_KEY:
        extra["api_key"] = GOOGLE_API_KEY

    # Use async completion (acompletion) so this node can be async-compatible in the graph
    if OFFLINE_MODE:
        # Provide deterministic offline answer so app still works
        return {"action": "answer", "route": None, "answer": "(Offline) I received your query and will help once an API key is configured."}
    
    try:
        resp = await acompletion(model=LLM_MODEL, messages=messages, max_completion_tokens=max_tokens, **extra)
        maybe_print_usage(resp, "router")
        raw = safe_extract_text(resp) or ""
        
        # Use robust JSON extraction and validation
        return extract_and_validate_json(raw)
        
    except Exception as e:
        # Connectivity or auth issue -> degrade gracefully
        print(f"LLM call error: {e}")
        return {"action": "answer", "route": None, "answer": f"I encountered an error processing your request. Please try again."}

# -----------------------
# Node: general_chat_and_section_routing
# -----------------------
async def general_chat_and_section_routing(state: ResumeBuilderState, config: Dict[str, Any]) -> Any:
    """
    SIMPLIFIED routing node that ONLY handles general chat and initial routing TO sections.
    
    This node is ONLY called when:
    1. User is in general chat (current_section is None)  
    2. User is returning from a section that exited to general chat
    
    All section-to-section routing is handled by section_chat_node internally.
    """
    # Initialize routing attempt counter if not exists
    if not hasattr(state, 'routing_attempts'):
        state.routing_attempts = 0
    
    # Prevent infinite loops
    if state.routing_attempts >= MAX_ROUTING_ATTEMPTS:
        print(f"Maximum routing attempts ({MAX_ROUTING_ATTEMPTS}) reached, ending conversation")
        state.current_section = None
        state.routing_attempts = 0
        state.next_action = None
        state.context.append(state.make_message(
            "assistant",
            "Let's start fresh - how can I help you with your resume?"
        ))
        return state
    
    state.routing_attempts += 1
    
    # This should ONLY be called for general chat scenarios
    if state.current_section is not None:
        print(f"Warning: GeneralChat called while in section '{state.current_section}' - this shouldn't happen")
        # Reset to general chat state
        state.current_section = None
        state.next_action = None
    
    # Detect if this is the first turn (no user message yet)
    last_msg = state.context[-1] if state.context else None
    first_turn = not state.context or (last_msg and getattr(last_msg, "role", "") != "user")
    user_text = ""
    
    if not first_turn:
        user_text = getattr(last_msg, "content", "") or ""
    else:
        # Seed a synthetic user query to drive initial LLM greeting
        user_text = "INITIAL_GREETING: Greet user, summarize JD, show sections with alignment scores."

    # Build compact payload for LLM - section summaries for general chat overview
    compact_sections = {}
    for s, data in state.section_objects.items():
        compact_sections[s] = {
            "alignment_score": data.get("alignment_score"),
            "missing_requirements": data.get("missing_requirements", [])[:2]  # Only first 2 for overview
        }

    # General chat system prompt - simplified since we only handle initial routing
    system_prompt = f"""You are a resume assistant in GENERAL CHAT mode.

AVAILABLE SECTIONS: {json.dumps(compact_sections, separators=(",", ":"))}
JD SUMMARY: {(state.jd_summary or "No JD provided")[:300]}

RESPONSE FORMAT: {{"action": "answer|route", "route": "section_name_or_null", "answer": "response_text"}}

RULES:
1. If user_text starts with 'INITIAL_GREETING': Provide friendly welcome, JD summary, section alignment overview
2. action='route' ONLY when user explicitly wants to work on/edit a specific section
3. action='answer' for questions, general chat, or unclear intent  
4. When routing, use exact section name from available sections
5. Keep responses helpful, under 150 words
6. Show section alignment scores when relevant
7. This is ONLY for general chat - section work happens elsewhere

Available sections: {list(compact_sections.keys())}"""
    
    payload = {
        "user_query": user_text,
        "sections_summary": compact_sections,
        "is_initial_greeting": first_turn
    }

    # Call LLM for general chat / initial routing decision
    try:
        parsed = await call_llm_json_decision(system_prompt, payload, max_tokens=400)
        
        # Reset routing attempts on successful processing
        state.routing_attempts = 0
        
        # Process LLM decision
        action = parsed.get("action")
        route_to = parsed.get("route")
        answer_text = parsed.get("answer", "")
        
        if action == "route" and route_to:
            # Route to a section - this is the ONLY routing this node does
            available_sections = list(compact_sections.keys())
            normalized_section = normalize_section_name(route_to, available_sections)
            
            if normalized_section:
                state.current_section = normalized_section
                state.next_action = "section_chat"  # Start with section chat
                print(f"Routing from general chat to section: {normalized_section}")
                if normalized_section != route_to:
                    print(f"Note: Corrected '{route_to}' to '{normalized_section}'")
                return state
            else:
                # No good match found - stay in general chat
                state.context.append(state.make_message(
                    "assistant",
                    f"I couldn't find section '{route_to}'. Available: {', '.join(sorted(available_sections))}. Which would you like to work on?"
                ))
                state.next_action = None
                return state
        
        else:
            # action == "answer" or fallback - general chat response
            state.next_action = None
            if answer_text:
                state.context.append(state.make_message("assistant", answer_text))
            else:
                state.context.append(state.make_message("assistant", "How can I help you with your resume today?"))
            return state
            
    except ValueError as e:
        print(f"JSON parsing error in general chat: {e}")
        state.context.append(state.make_message("assistant", "Could you please rephrase your request?"))
        state.next_action = None
        return state
    except Exception as e:
        print(f"Unexpected error in general chat: {e}")
        state.context.append(state.make_message("assistant", "How can I help you with your resume?"))
        state.next_action = None
        return state