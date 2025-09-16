import subprocess
import json
import re

def check_ollama_available(model):
    """Check if Ollama is running and model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=300
        )
        if result.returncode != 0:
            return False, "Ollama is not running"
        
        # Check if model is available
        output = result.stdout.decode('utf-8')
        if model not in output:
            available_models = []
            for line in output.split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    if model_name != "NAME":  # Skip header
                        available_models.append(model_name)
            return False, f"Model '{model}' not found. Available: {', '.join(available_models)}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def check_claim_with_ollama(claim, hits, model="llama3.1"):
    """
    Run claim + evidence chunks through Ollama to classify truthfulness.
    Enhanced version with better context handling and formatting.
    Returns: { verdict: "SUPPORT"|"REFUTE"|"NOT FOUND", confidence: int, explanation: str }
    """
    # Check if Ollama and model are available
    available, message = check_ollama_available(model)
    if not available:
        return {
            "verdict": "ERROR",
            "confidence": 0,
            "explanation": message
        }
    
    # Build evidence context with better formatting
    evidence_sections = []
    for i, h in enumerate(hits[:10]):  # Limit to top 10 for context length
        section = f"""=== Evidence {i+1} ===
Location: Page {h['metadata'].get('page')}, Lines {h['metadata'].get('start_line')}-{h['metadata'].get('end_line')}
Relevance Score: {1 - h.get('distance', 0):.3f}

{h['document']}
"""
        evidence_sections.append(section)
    
    context_text = "\n".join(evidence_sections)
    
    # Enhanced prompt with better instructions
    prompt = f"""You are an expert legal fact-checker analyzing deposition testimony.

CLAIM TO VERIFY:
"{claim}"

AVAILABLE EVIDENCE:
{context_text}

INSTRUCTIONS:
1. Carefully read through all evidence sections
2. Determine if the evidence SUPPORTS, REFUTES, or provides NO CLEAR ANSWER to the claim
3. Consider the context and exact wording in depositions
4. Rate your confidence from 0-100 based on how clear the evidence is

RESPONSE FORMAT (respond with valid JSON only):
{{
  "verdict": "SUPPORT",
  "confidence": 85,
  "explanation": "The evidence clearly shows..."
}}

Valid verdict values: SUPPORT, REFUTE, NOT_FOUND
"""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=500  # Increased timeout for larger contexts
        )
        
        if result.returncode != 0:
            return {
                "verdict": "ERROR", 
                "confidence": 0, 
                "explanation": f"Ollama error: {result.stderr.decode('utf-8')}"
            }

        raw = result.stdout.decode("utf-8").strip()
        
        # Try to extract JSON from response (sometimes models add extra text)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            # Validate and clean the response
            verdict = parsed.get("verdict", "UNKNOWN").upper()
            if verdict not in ["SUPPORT", "REFUTE", "NOT_FOUND"]:
                verdict = "NOT_FOUND"
            
            confidence = parsed.get("confidence", 0)
            if not isinstance(confidence, int) or confidence < 0 or confidence > 100:
                confidence = 50
            
            explanation = parsed.get("explanation", "No explanation provided")
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation
            }
        else:
            # Fallback parsing if no JSON found
            return {
                "verdict": "UNKNOWN", 
                "confidence": 0, 
                "explanation": f"Could not parse response: {raw[:200]}..."
            }
            
    except subprocess.TimeoutExpired:
        return {
            "verdict": "ERROR", 
            "confidence": 0, 
            "explanation": "Ollama request timed out"
        }
    except Exception as e:
        return {
            "verdict": "ERROR", 
            "confidence": 0, 
            "explanation": f"Error: {str(e)}"
        }
