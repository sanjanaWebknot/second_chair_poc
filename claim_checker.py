import subprocess
import json
import re
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from models import ClaimCheckResponse

def create_claim_check_prompt():
    """Create the prompt template for claim checking."""
    template = """You are an expert legal fact-checker analyzing deposition testimony with advanced date and number matching capabilities.

CLAIM TO VERIFY:
"{claim}"

AVAILABLE EVIDENCE:
{evidence}

CRITICAL INSTRUCTIONS:

1. SEMANTIC DATE/NUMBER MATCHING RULES:
   • Treat equivalent date/number formats as IDENTICAL:
     - "'65" = "1965" = "65" (when referring to years)
     - "late 60s" = "late 60's" = "late sixties" = "1960s" = "'67" = "'68" = "'69"
     - "early 70s" = "early seventies" = "'70" = "'71" = "'72" = "1970s"  
     - "mid 80s" = "middle eighties" = "'84" = "'85" = "'86" = "1980s"
     - "12th Aug 2025" = "12/08/2025" = "August 12, 2025" = "Aug 12, 2025"
     - "first" = "1st", "second" = "2nd", "twenty-five" = "25"
   
   • Context-aware year interpretation:
     - In legal/historical contexts, assume 1900s for 2-digit years unless context suggests otherwise
     - "'65" in deposition likely means "1965", not "2065" or age "65"

2. ANALYSIS PROCESS:
   • Step 1: Identify all dates/numbers in both claim and evidence
   • Step 2: Apply semantic matching rules to normalize formats mentally
   • Step 3: Compare normalized versions for factual alignment
   • Step 4: Determine SUPPORT/REFUTE/NOT_FOUND based on semantic content

3. VERDICT CRITERIA:
   • SUPPORT: Evidence clearly confirms the claim (considering format equivalence)
   • REFUTE: Evidence clearly contradicts the claim (considering format equivalence)
   • NOT_FOUND: No relevant evidence or evidence is ambiguous/insufficient

4. CONFIDENCE SCORING:
   • 90-100: Very clear evidence with exact semantic match
   • 70-89: Strong evidence with minor format differences (now resolved by rules)
   • 50-69: Moderate evidence, some ambiguity remains
   • 30-49: Weak evidence, significant uncertainty
   • 0-29: Very unclear or conflicting evidence

5. OUTPUT FORMAT:
   • You MUST respond with ONLY the JSON object as specified below
   • Do NOT include any explanatory text before or after the JSON
   • Do NOT wrap the JSON in code blocks or markdown
   • Do NOT add any commentary or analysis outside the JSON

{format_instructions}

Remember: Focus on SEMANTIC MEANING, not exact text matching. Different date/number formats can represent the same factual information. Respond with ONLY the JSON object."""

    return PromptTemplate(
        template=template,
        input_variables=["claim", "evidence"],
        partial_variables={"format_instructions": "{format_instructions}"}
    )

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
            for line in output.split('\n')[1:]:  
                if line.strip():
                    model_name = line.split()[0]
                    if model_name != "NAME":  # Skip header
                        available_models.append(model_name)
            return False, f"Model '{model}' not found. Available: {', '.join(available_models)}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def check_claim_with_ollama_chain(claim, hits, model="llama3.1"):
    """
    Run claim + evidence chunks through Ollama using LangChain chain pattern.
    Returns: ClaimCheckResponse object with validated fields.
    """
    # Check if Ollama and model are available
    available, message = check_ollama_available(model)
    if not available:
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation=message
        )
    
    # Format evidence sections
    evidence_sections = []
    for i, h in enumerate(hits[:10]): 
        section = f"""=== Evidence {i+1} ===
        Location: Page {h['metadata'].get('page')}, Lines {h['metadata'].get('start_line')}-{h['metadata'].get('end_line')}
        Relevance Score: {1 - h.get('distance', 0):.3f}

        {h['document']}

        """
        evidence_sections.append(section)
    
    context_text = "\n".join(evidence_sections)
    
    try:
        # Check if OllamaLLM is available
        if OllamaLLM is None:
            return ClaimCheckResponse(
                verdict="ERROR",
                confidence=0,
                explanation="OllamaLLM not available. Please install langchain-ollama package."
            )
        
        # Initialize Ollama LLM
        llm = OllamaLLM(model=model)
        
        # Initialize PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=ClaimCheckResponse)
        
        # Create prompt template
        prompt_template = create_claim_check_prompt()
        
        # Create the chain: prompt | model | parser
        chain = prompt_template | llm | parser
        
        # Invoke the chain
        result = chain.invoke({
            "claim": claim,
            "evidence": context_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
        
    except Exception as e:
        # Enhanced error handling for LangChain chain
        error_msg = str(e)
        print(f"Chain processing error: {error_msg}")
        
        # If the error is related to parsing, try to extract JSON from the error message
        if "parsing" in error_msg.lower() or "json" in error_msg.lower():
            # Try to extract JSON from the error message itself
            json_match = re.search(r'\{.*\}', error_msg, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    
                    # Handle key variations by position (model always returns: verdict, confidence, explanation)
                    def normalize_json_keys(data):
                        """Normalize JSON keys based on position since model returns in order: verdict, confidence, explanation."""
                        items = list(data.items())
                        normalized = {}
                        
                        # Map by position: first = verdict, second = confidence, third = explanation
                        if len(items) >= 1:
                            normalized["verdict"] = items[0][1]
                        if len(items) >= 2:
                            normalized["confidence"] = items[1][1]
                        if len(items) >= 3:
                            normalized["explanation"] = items[2][1]
                        
                        # Keep any additional keys as-is
                        for i in range(3, len(items)):
                            normalized[items[i][0]] = items[i][1]
                            
                        return normalized
                    
                    normalized_json = normalize_json_keys(parsed_json)
                    
                    return ClaimCheckResponse(
                        verdict=normalized_json.get("verdict", "UNKNOWN"),
                        confidence=int(normalized_json.get("confidence", 50)),
                        explanation=normalized_json.get("explanation", "No explanation available")
                    )
                except (json.JSONDecodeError, ValueError):
                    pass
        
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation=f"Chain processing error: {error_msg}"
        )

def check_claim_with_ollama(claim, hits, model="llama3.1"):
    """
    Run claim + evidence chunks through Ollama to classify truthfulness.
    Enhanced version with PydanticOutputParser for structured output.
    Returns: ClaimCheckResponse object with validated fields.
    """
    # Initialize PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=ClaimCheckResponse)
    
    # Check if Ollama and model are available
    available, message = check_ollama_available(model)
    if not available:
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation=message
        )
    
    evidence_sections = []
    for i, h in enumerate(hits[:10]): 
        section = f"""=== Evidence {i+1} ===
        Location: Page {h['metadata'].get('page')}, Lines {h['metadata'].get('start_line')}-{h['metadata'].get('end_line')}
        Relevance Score: {1 - h.get('distance', 0):.3f}

        {h['document']}

        """
        evidence_sections.append(section)
    
    context_text = "\n".join(evidence_sections)
    
    # Enhanced prompt with PydanticOutputParser format instructions
    prompt = f"""You are an expert legal fact-checker analyzing deposition testimony with advanced date and number matching capabilities.

    CLAIM TO VERIFY:
    "{claim}"

    AVAILABLE EVIDENCE:
    {context_text}

    CRITICAL INSTRUCTIONS:

    1. SEMANTIC DATE/NUMBER MATCHING RULES:
    • Treat equivalent date/number formats as IDENTICAL:
        - "'65" = "1965" = "65" (when referring to years)
        - "late 60s" = "late 60's" = "late sixties" = "1960s" = "'67" = "'68" = "'69"
        - "early 70s" = "early seventies" = "'70" = "'71" = "'72" = "1970s"  
        - "mid 80s" = "middle eighties" = "'84" = "'85" = "'86" = "1980s"
        - "12th Aug 2025" = "12/08/2025" = "August 12, 2025" = "Aug 12, 2025"
        - "first" = "1st", "second" = "2nd", "twenty-five" = "25"
    
    • Context-aware year interpretation:
        - In legal/historical contexts, assume 1900s for 2-digit years unless context suggests otherwise
        - "'65" in deposition likely means "1965", not "2065" or age "65"

    2. ANALYSIS PROCESS:
    • Step 1: Identify all dates/numbers in both claim and evidence
    • Step 2: Apply semantic matching rules to normalize formats mentally
    • Step 3: Compare normalized versions for factual alignment
    • Step 4: Determine SUPPORT/REFUTE/NOT_FOUND based on semantic content

    3. VERDICT CRITERIA:
    • SUPPORT: Evidence clearly confirms the claim (considering format equivalence)
    • REFUTE: Evidence clearly contradicts the claim (considering format equivalence)
    • NOT_FOUND: No relevant evidence or evidence is ambiguous/insufficient

    4. CONFIDENCE SCORING:
    • 90-100: Very clear evidence with exact semantic match
    • 70-89: Strong evidence with minor format differences (now resolved by rules)
    • 50-69: Moderate evidence, some ambiguity remains
    • 30-49: Weak evidence, significant uncertainty
    • 0-29: Very unclear or conflicting evidence

    5. OUTPUT FORMAT:
    • You MUST respond with ONLY the JSON object as specified below
    • Do NOT include any explanatory text before or after the JSON
    • Do NOT wrap the JSON in code blocks or markdown
    • Do NOT add any commentary or analysis outside the JSON

    {parser.get_format_instructions()}

    Remember: Focus on SEMANTIC MEANING, not exact text matching. Different date/number formats can represent the same factual information. Respond with ONLY the JSON object."""


    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=500  # Increased timeout for larger contexts
        )
        
        if result.returncode != 0:
            return ClaimCheckResponse(
                verdict="ERROR",
                confidence=0,
                explanation=f"Ollama error: {result.stderr.decode('utf-8')}"
            )

        raw = result.stdout.decode("utf-8").strip()
        
        # Clean the raw response to remove control characters and fix common issues
        cleaned_raw = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw)  # Remove control characters
        cleaned_raw = cleaned_raw.replace('\n', ' ').replace('\r', ' ')  # Replace newlines
        
        # Debug: print what we got from the model
        print(f"Raw model response: {cleaned_raw[:200]}{'...' if len(cleaned_raw) > 200 else ''}")
        
        # Enhanced JSON extraction with multiple fallback strategies
        def extract_json_from_response(text):
            """Extract JSON from response using multiple strategies."""
            strategies = [
                # Strategy 1: Look for JSON wrapped in ```json``` code blocks
                r'```json\s*(\{.*?\})\s*```',
                # Strategy 2: Look for JSON wrapped in ``` code blocks
                r'```\s*(\{.*?\})\s*```',
                # Strategy 3: Look for JSON after "here is" or similar phrases
                r'(?:here is|following|output|result).*?(\{.*?\})',
                # Strategy 4: Look for JSON at the end of the response
                r'(\{.*?\})\s*$',
                # Strategy 5: Look for any JSON object in the text
                r'(\{.*?\})',
            ]
            
            for pattern in strategies:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean the match
                        clean_match = match.strip()
                        # Remove any trailing text after the JSON
                        if clean_match.count('{') > clean_match.count('}'):
                            # Find the last complete JSON object
                            brace_count = 0
                            end_pos = 0
                            for i, char in enumerate(clean_match):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_pos = i + 1
                                        break
                            clean_match = clean_match[:end_pos]
                        
                        parsed = json.loads(clean_match)
                        if isinstance(parsed, dict) and 'verdict' in parsed:
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            return None
        
        # Try to extract JSON using enhanced strategies
        extracted_json = extract_json_from_response(cleaned_raw)
        if extracted_json:
            try:
                # Handle key variations by position (model always returns: verdict, confidence, explanation)
                def normalize_json_keys(data):
                    """Normalize JSON keys based on position since model returns in order: verdict, confidence, explanation."""
                    items = list(data.items())
                    normalized = {}
                    
                    # Map by position: first = verdict, second = confidence, third = explanation
                    if len(items) >= 1:
                        normalized["verdict"] = items[0][1]
                    if len(items) >= 2:
                        normalized["confidence"] = items[1][1]
                    if len(items) >= 3:
                        normalized["explanation"] = items[2][1]
                    
                    # Keep any additional keys as-is
                    for i in range(3, len(items)):
                        normalized[items[i][0]] = items[i][1]
                        
                    return normalized
                
                # Apply the normalization function to the extracted JSON
                normalized_json = normalize_json_keys(extracted_json)
                
                return ClaimCheckResponse(
                    verdict=normalized_json.get("verdict", "UNKNOWN"),
                    confidence=int(normalized_json.get("confidence", 50)),
                    explanation=normalized_json.get("explanation", "No explanation available")
                )
            except (ValueError, TypeError) as e:
                print(f"Error creating ClaimCheckResponse from extracted JSON: {e}")
        
        # Fallback: try PydanticOutputParser on the cleaned raw text
        try:
            parsed_response = parser.parse(cleaned_raw)
            return parsed_response
        except Exception as e:
            print(f"Pydantic parsing error: {e}")
            print(f"Problematic response: {cleaned_raw[:300]}...")
            
            # Final fallback: try to extract JSON manually with simple regex
            json_match = re.search(r'\{.*\}', cleaned_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    
                    # Handle key variations by position (model always returns: verdict, confidence, explanation)
                    def normalize_json_keys(data):
                        """Normalize JSON keys based on position since model returns in order: verdict, confidence, explanation."""
                        items = list(data.items())
                        normalized = {}
                        
                        # Map by position: first = verdict, second = confidence, third = explanation
                        if len(items) >= 1:
                            normalized["verdict"] = items[0][1]
                        if len(items) >= 2:
                            normalized["confidence"] = items[1][1]
                        if len(items) >= 3:
                            normalized["explanation"] = items[2][1]
                        
                        # Keep any additional keys as-is
                        for i in range(3, len(items)):
                            normalized[items[i][0]] = items[i][1]
                            
                        return normalized
                    
                    normalized_json = normalize_json_keys(parsed_json)
                    
                    return ClaimCheckResponse(
                        verdict=normalized_json.get("verdict", "UNKNOWN"),
                        confidence=int(normalized_json.get("confidence", 50)),
                        explanation=normalized_json.get("explanation", "No explanation available")
                    )
                except (json.JSONDecodeError, ValueError) as json_error:
                    print(f"JSON fallback error: {json_error}")
            
            # Ultimate fallback: return error response
            return ClaimCheckResponse(
                verdict="ERROR",
                confidence=0,
                explanation=f"Failed to parse response: {str(e)}"
            )
            
    except subprocess.TimeoutExpired:
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation="Ollama request timed out"
        )
    except Exception as e:
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation=f"Error: {str(e)}"
        )
