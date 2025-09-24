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

{format_instructions}

Remember: Focus on SEMANTIC MEANING, not exact text matching. Different date/number formats can represent the same factual information."""

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
        return ClaimCheckResponse(
            verdict="ERROR",
            confidence=0,
            explanation=f"Chain processing error: {str(e)}"
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

    {parser.get_format_instructions()}

    Remember: Focus on SEMANTIC MEANING, not exact text matching. Different date/number formats can represent the same factual information."""


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
        
        # Use PydanticOutputParser to parse the response
        try:
            parsed_response = parser.parse(cleaned_raw)
            return parsed_response
        except Exception as e:
            print(f"Pydantic parsing error: {e}")
            print(f"Problematic response: {cleaned_raw[:300]}...")
            
            # Fallback: try to extract JSON manually and create ClaimCheckResponse
            json_match = re.search(r'\{.*\}', cleaned_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return ClaimCheckResponse(
                        verdict=parsed_json.get("verdict", "UNKNOWN"),
                        confidence=int(parsed_json.get("confidence", 50)),
                        explanation=parsed_json.get("explanation", "No explanation provided")
                    )
                except (json.JSONDecodeError, ValueError) as json_error:
                    print(f"JSON fallback error: {json_error}")
            
            # Final fallback: return error response
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
