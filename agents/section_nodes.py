# agents/section_nodes.py
"""
Specialized section nodes with internal routing - no need to go back to AnalyzeUserQuery.
Sections handle their own routing including section-to-section switches.
"""

import json
import re
from typing import Any, Dict, List
from agents.resume_builder_state import ResumeBuilderState
from agents.general_chat_section_routing import (
    call_llm_json_decision, safe_extract_text, maybe_print_usage, 
    OFFLINE_MODE, GOOGLE_API_KEY, LLM_MODEL, safe_initialize_answers,
    detect_question_matches, normalize_section_name
)
from litellm import acompletion

async def section_chat_node(state: ResumeBuilderState, config: Dict[str, Any]) -> ResumeBuilderState:
    """
    Handle conversations within a section + internal routing (section switches, exits).
    This node now handles ALL section routing decisions internally.
    """
    print(f"\n--- Section Chat: {state.current_section} ---")
    
    if not state.current_section:
        # Shouldn't happen, but handle gracefully
        state.context.append(state.make_message("assistant", "Please select a section to work on."))
        state.next_action = "exit_to_general"
        return state
    
    # Get section data
    section_data = state.section_objects.get(state.current_section, {})
    recommended_questions = section_data.get("recommended_questions", [])
    available_sections = list(state.section_objects.keys())
    
    # Initialize answers if needed
    safe_initialize_answers(state, state.current_section, recommended_questions)
    current_answers = state.recommended_answers.get(state.current_section, [])
    
    # Get the last user message
    last_msg = state.context[-1] if state.context else None
    is_first_entry = not last_msg or getattr(last_msg, "role", "") != "user"
    user_text = ""
    
    if is_first_entry:
        user_text = f"SECTION_ENTRY: User just entered {state.current_section} section."
    else:
        user_text = getattr(last_msg, "content", "") or ""
    
    # Check if all questions are answered
    all_questions_answered = (
        len(recommended_questions) > 0 and 
        len(current_answers) == len(recommended_questions) and
        all(answer and len(str(answer).strip()) > 0 for answer in current_answers)
    )
    
    # Enhanced system prompt with internal routing logic
    system_prompt = f"""You are managing the {state.current_section} section with FULL ROUTING CAPABILITY.

SECTION INFO:
- Current questions: {recommended_questions}
- Current answers: {current_answers} 
- All questions answered: {all_questions_answered}
- Available sections: {available_sections}

RESPONSE FORMAT:
{{"action": "stay|switch_section|exit_section|trigger_updater|trigger_applier", "target_section": "section_name_or_null", "answer": "response_text", "updated_answers": [...] }}

ROUTING RULES:
1. action='switch_section' if user wants to go to a DIFFERENT section (set target_section)
2. action='exit_section' if user wants general chat/main menu  
3. action='trigger_updater' if all questions answered and user hasn't seen updated content
4. action='trigger_applier' if user says "apply", "yes" (to apply), "save changes"
5. action='stay' for normal chat within current section

SPECIAL CASE: If user mentions the SAME section they're already in, just acknowledge and stay (action='stay')

CHAT RULES:
1. If user_text starts with "SECTION_ENTRY":
   - Welcome to {state.current_section} section
   - Show alignment score: {section_data.get('alignment_score', 'Not calculated')}
   - List unanswered questions in numbered format
   
2. If user is answering questions:
   - Update 'updated_answers' array with new answers in correct positions
   - Acknowledge briefly, show remaining questions
   
3. Always end with numbered list of remaining unanswered questions (if any)
4. Keep responses under 120 words, be conversational
5. Handle section switches directly - don't defer to other systems"""

    # Prepare payload
    payload = {
        "user_message": user_text,
        "current_section": state.current_section,
        "available_sections": available_sections,
        "questions": recommended_questions,
        "answers": current_answers,
        "all_answered": all_questions_answered
    }
    
    try:
        parsed = await call_llm_json_decision(system_prompt, payload, max_tokens=500)
        
        # Handle answer updates first
        if 'updated_answers' in parsed and isinstance(parsed['updated_answers'], list):
            updated_answers = parsed['updated_answers']
            
            # Update answers in state
            for i, answer in enumerate(updated_answers):
                if i < len(current_answers) and answer and answer.strip():
                    state.recommended_answers[state.current_section][i] = answer.strip()
            
            print(f"Updated answers for {state.current_section}")
        
        # Handle routing decisions
        action = parsed.get("action", "stay")
        target_section = parsed.get("target_section")
        answer_text = parsed.get("answer", "")
        
        if action == "switch_section" and target_section:
            # Handle section switching internally
            normalized_section = normalize_section_name(target_section, available_sections)
            
            if normalized_section and normalized_section != state.current_section:
                # Switch to different section
                state.current_section = normalized_section
                state.next_action = "section_chat"  # Continue in section chat
                print(f"Switching to section: {normalized_section}")
                if answer_text:
                    state.context.append(state.make_message("assistant", answer_text))
                return state
            elif normalized_section == state.current_section:
                # User mentioned same section they're already in
                state.next_action = "section_chat"
                same_section_msg = answer_text or f"You're already in the {state.current_section} section. How can I help you improve it?"
                state.context.append(state.make_message("assistant", same_section_msg))
                return state
            else:
                # Invalid section name
                state.next_action = "section_chat"
                error_msg = f"I couldn't find section '{target_section}'. Available: {', '.join(sorted(available_sections))}. Staying in {state.current_section}."
                state.context.append(state.make_message("assistant", error_msg))
                return state
                
        elif action == "exit_section":
            # Exit to general chat
            state.current_section = None
            state.next_action = "exit_to_general"
            exit_msg = answer_text or "Returning to general chat. How can I help with your resume?"
            state.context.append(state.make_message("assistant", exit_msg))
            return state
            
        elif action == "trigger_updater":
            # All questions answered - trigger updater
            state.next_action = "section_updater"
            if answer_text:
                state.context.append(state.make_message("assistant", answer_text))
            return state
            
        elif action == "trigger_applier":
            # User wants to apply changes
            state.next_action = "section_applier"
            if answer_text:
                state.context.append(state.make_message("assistant", answer_text))
            return state
            
        else:
            # action == "stay" or fallback - continue in section
            state.next_action = None  # Clear next_action to end execution
            if answer_text:
                state.context.append(state.make_message("assistant", answer_text))
            return state
        
    except Exception as e:
        print(f"Error in section_chat_node: {e}")
        state.context.append(state.make_message(
            "assistant", 
            f"I'm here to help with your {state.current_section} section. What would you like to know?"
        ))
        state.next_action = "section_chat"
        return state

async def section_updater_node(state: ResumeBuilderState, config: Dict[str, Any]) -> ResumeBuilderState:
    """
    Generate updated section content based on answered questions.
    """
    print(f"\n--- Section Updater: {state.current_section} ---")
    
    if not state.current_section:
        state.next_action = "exit_to_general"
        return state
    
    # Get current data
    section_data = state.section_objects.get(state.current_section, {})
    recommended_questions = section_data.get("recommended_questions", [])
    current_answers = state.recommended_answers.get(state.current_section, [])
    original_content = state.resume_sections.get(state.current_section, "")
    
    # System prompt for content generation
    system_prompt = f"""Generate updated {state.current_section} content based on answers.

ORIGINAL CONTENT: {original_content}

QUESTIONS & ANSWERS:
{chr(10).join(f"Q{i+1}: {q}\\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(recommended_questions, current_answers)) if a)}

RESPONSE FORMAT:
{{"updated_content": "new_content", "summary": "brief_summary"}}

RULES:
1. Keep same format/structure as original
2. Enhance with details from answers
3. Don't remove good existing content
4. Be concise and professional"""

    payload = {
        "original_content": original_content,
        "questions": recommended_questions,
        "answers": current_answers
    }
    
    try:
        parsed = await call_llm_json_decision(system_prompt, payload, max_tokens=600)
        
        updated_content = parsed.get("updated_content", "")
        summary = parsed.get("summary", "Content updated")
        
        if updated_content:
            # Store proposed content
            state.proposed_section_content = updated_content
            
            response_msg = (
                f"{summary}\n\n"
                f"**Updated {state.current_section}:**\n\n"
                f"{updated_content}\n\n"
                f"Apply these changes? (say 'yes' or 'apply')"
            )
        else:
            response_msg = f"Having trouble updating {state.current_section}. Let's continue."
        
        state.context.append(state.make_message("assistant", response_msg))
        state.next_action = "section_chat"  # Return to section chat for confirmation
        return state
        
    except Exception as e:
        print(f"Error in section_updater_node: {e}")
        state.context.append(state.make_message(
            "assistant",
            f"Trouble updating {state.current_section} content. Let's continue working on it."
        ))
        state.next_action = "section_chat"
        return state

async def section_applier_node(state: ResumeBuilderState, config: Dict[str, Any]) -> ResumeBuilderState:
    """
    Apply changes and re-analyze section alignment with JD.
    """
    print(f"\n--- Section Applier: {state.current_section} ---")
    
    if not state.current_section:
        state.next_action = "exit_to_general"
        return state
    
    # Get content to apply
    content_to_apply = getattr(state, 'proposed_section_content', None)
    if not content_to_apply:
        content_to_apply = state.resume_sections.get(state.current_section, "")
    
    if not content_to_apply:
        state.context.append(state.make_message(
            "assistant",
            f"No content to apply for {state.current_section}. Please try again."
        ))
        state.next_action = "section_chat"
        return state
    
    # Apply changes and re-analyze
    await apply_section_changes_internal(state, state.current_section, content_to_apply)
    
    # Clean up
    if hasattr(state, 'proposed_section_content'):
        delattr(state, 'proposed_section_content')
    
    # Return to section chat with fresh data
    state.next_action = "section_chat"
    return state

async def apply_section_changes_internal(state: ResumeBuilderState, section_name: str, updated_content: str):
    """Apply changes and re-analyze section."""
    print(f"\n🔄 APPLYING changes to {section_name}...")
    
    # Update resume content
    if not hasattr(state, 'resume_sections'):
        state.resume_sections = {}
    state.resume_sections[section_name] = updated_content
    
    # Heavy analysis prompt (only used here)
    analysis_prompt = f"""Analyze updated resume section alignment with job requirements.

JOB DESCRIPTION: {state.jd_summary or ""}
SECTION: {section_name}
CONTENT: {updated_content}

RESPONSE FORMAT (JSON):
{{"alignment_score": <0-100>, "missing_requirements": ["req1", "req2"], "recommended_questions": ["q1", "q2"], "analysis_summary": "brief_summary"}}

Focus on most important gaps, max 3-4 requirements, 2-4 targeted questions."""
    
    try:
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": f"Section: {section_name}\nContent: {updated_content}"}
        ]
        
        extra = {}
        if GOOGLE_API_KEY:
            extra["api_key"] = GOOGLE_API_KEY

        if OFFLINE_MODE:
            analysis_result = {
                "alignment_score": 75,
                "missing_requirements": ["More examples needed"],
                "recommended_questions": [f"Add more examples to {section_name}?"],
                "analysis_summary": "Offline mode"
            }
        else:
            resp = await acompletion(model=LLM_MODEL, messages=messages, max_completion_tokens=400, **extra)
            maybe_print_usage(resp, f"apply_{section_name}")
            raw = safe_extract_text(resp) or ""
            
            # Extract JSON
            analysis_result = {}
            try:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group(0))
            except:
                pass
            
            # Defaults
            analysis_result.setdefault("alignment_score", 70)
            analysis_result.setdefault("missing_requirements", [])
            analysis_result.setdefault("recommended_questions", [])
            analysis_result.setdefault("analysis_summary", "Updated successfully")
        
        # Update state
        if section_name not in state.section_objects:
            state.section_objects[section_name] = {}
        
        state.section_objects[section_name].update({
            "alignment_score": analysis_result["alignment_score"],
            "missing_requirements": analysis_result["missing_requirements"],
            "recommended_questions": analysis_result["recommended_questions"]
        })
        
        # Reset answers for new questions
        if analysis_result["recommended_questions"]:
            state.recommended_answers[section_name] = [""] * len(analysis_result["recommended_questions"])
        else:
            state.recommended_answers[section_name] = []
        
        # Confirmation message
        confirmation_msg = (
            f"✅ Applied changes to {section_name}!\n\n"
            f"New alignment: {analysis_result['alignment_score']}%\n"
            f"{analysis_result.get('analysis_summary', '')}\n\n"
        )
        
        if analysis_result.get("recommended_questions"):
            confirmation_msg += f"Continue with {len(analysis_result['recommended_questions'])} more questions?"
        else:
            confirmation_msg += "Section complete! Switch to another section or continue refining."
        
        state.context.append(state.make_message("assistant", confirmation_msg))
        print(f"✅ {section_name} updated - Score: {analysis_result['alignment_score']}%")
        
    except Exception as e:
        print(f"❌ Error applying changes: {e}")
        error_msg = f"✅ Changes saved to {section_name}, but analysis failed. Continue working on this section."
        state.context.append(state.make_message("assistant", error_msg))