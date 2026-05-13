import json

from ollama import chat

# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------
# Options: "qwen3:1.7b" or "deepseek-r1:1.5b"
MODEL_NAME = "qwen3:1.7b"
# ---------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------

SYSTEM_PROMPT = """
You are an AI-powered Advanced Driver Assistance System (ADAS).

Your role:
- analyze driving conditions
- assess road risk
- provide safe driving recommendations
- generate concise driver alerts

Rules:
- prioritize safety
- keep explanations short
- do not hallucinate
- do not generate unnecessary text
- maintain professional driving-assistant tone

Always provide EXACTLY these fields:

Risk Level:
Hazard Alert:
Recommended Speed:
Driving Suggestion:
Short Explanation:
Object Detection Alert:
Voice Alert:

IMPORTANT:
- Voice Alert must be VERY SHORT.
- Voice Alert should be suitable for speech output.
- Maximum 1 short sentence for Voice Alert.
"""
# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------

def get_llm_decision(context):

    """
    Input:
        context dictionary

    Returns:
        LLM text response
    """

    try:

        # ---------------------------------------------------
        # BUILD USER PROMPT
        # ---------------------------------------------------

        user_prompt = f"""
Analyze the following driving conditions:

{json.dumps(context, indent=2)}

Generate a driving safety assessment.
"""

        # ---------------------------------------------------
        # OLLAMA INFERENCE
        # ---------------------------------------------------

        response = chat(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",

                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",

                    "content": user_prompt
                }
            ]
        )

        # ---------------------------------------------------
        # RETURN RESPONSE
        # ---------------------------------------------------

        return response.message.content

    except Exception as e:

        return f"LLM Error: {str(e)}"