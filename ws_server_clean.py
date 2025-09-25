# ws_server_robust.py - Robust Q&A pair processing with proper state management
import os
import uuid
import json
import asyncio
import re
import time
from typing import List, Dict, Any, Optional
from collections import deque
from enum import Enum
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import chromadb
from embeddings import get_sentence_transformer  
from chroma_utils import create_chroma_client    
from claim_checker import check_claim_with_ollama 
from models import ClaimCheckResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set tokenizers parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_DIR = "./chroma_db_second_chair"
COLLECTION_NAME = "second_chair_depositions"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
PDF_OUT = "example_testimonial_transcript.pdf"

app = FastAPI()

# Initialize embedder and chroma
embedder = get_sentence_transformer(EMBED_MODEL_NAME)
chroma_client = create_chroma_client(CHROMA_DIR)
if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
else:
    collection = chroma_client.create_collection(name=COLLECTION_NAME,
                                                  metadata={"source": "second_chair_ws"})

# Global state
running_transcript: List[Dict[str, Any]] = []
transcript_lock = asyncio.Lock()
global_sequence_counter = 0

# Connection tracking
active_connections: Dict[str, Dict] = {}
connection_lock = asyncio.Lock()

# Transcript session state
transcript_session_active = False
transcript_session_id: Optional[str] = None
transcript_start_time: Optional[datetime] = None
session_lock = asyncio.Lock()

class PairState(Enum):
    FORMING = "forming"
    READY_TO_PROCESS = "ready_to_process"
    PROCESSING = "processing"
    COMPLETED = "completed"

class QAPairBuffer:
    """Robust Q&A pair detection with state management for fast streams."""
    
    def __init__(self, max_pairs: int = 20):
        self.max_pairs = max_pairs
        self.word_buffer: List[str] = []
        self.pairs_buffer: List[Dict[str, Any]] = []  
        self.current_q: str = ""
        self.current_a: str = ""
        self.in_answer: bool = False
        self.pair_counter: int = 0
        
    def add_word(self, word: str) -> List[Dict[str, Any]]:
        """Add word and return pairs ready for processing."""
        self.word_buffer.append(word)
        ready_pairs = []
        
        # Check for Q. pattern - exact match for robustness
        if self._is_question_marker(word):
            # Complete previous pair if we have both Q and A
            if self.current_q and self.current_a and self.in_answer:
                completed_pair = self._create_pair(self.current_q, self.current_a)
                self._add_pair_to_buffer(completed_pair)
                print(f"📝 PAIR_COMPLETE: {completed_pair['id']} | Q: {self.current_q[:30]}... | A: {self.current_a[:30]}...")
                
                # Mark older pairs as ready (keep newest forming)
                ready_pairs.extend(self._mark_older_pairs_ready())
            
            # Start new question
            self.current_q = ""
            self.current_a = ""
            self.in_answer = False
            
        # Check for A. pattern
        elif self._is_answer_marker(word):
            if self.current_q:
                self.current_a = ""
                self.in_answer = True
                
        else:
            # Regular word - add to current Q or A (skip legal markers)
            if not self._is_legal_marker(word):
                if self.in_answer and self.current_q:
                    self._add_to_answer(word)
                elif not self.in_answer:
                    self._add_to_question(word)
        
        return ready_pairs
    
    def get_buffer_pressure(self) -> float:
        """Get buffer pressure (0.0 to 1.0)."""
        return len(self.pairs_buffer) / self.max_pairs
    
    def _mark_older_pairs_ready(self) -> List[Dict[str, Any]]:
        """Mark older forming pairs as ready based on buffer pressure."""
        ready_pairs = []
        forming_pairs = [p for p in self.pairs_buffer if p["state"] == PairState.FORMING]
        
        # Always keep the newest pair forming (it might still be receiving words)
        if len(forming_pairs) > 1:
            # Mark all but the newest as ready
            pairs_to_ready = forming_pairs[:-1]
            for pair in pairs_to_ready:
                pair["state"] = PairState.READY_TO_PROCESS
                ready_pairs.append(pair)
                print(f"✅ MARKED_READY: {pair['id']} (seq: {pair['sequence_number']})")
        
        # If buffer is getting very full, mark all forming pairs as ready
        elif self.get_buffer_pressure() > 0.8:
            for pair in forming_pairs:
                pair["state"] = PairState.READY_TO_PROCESS
                ready_pairs.append(pair)
                print(f"🔄 PRESSURE_READY: {pair['id']} (buffer at {self.get_buffer_pressure():.1%})")
        
        return ready_pairs
    
    def _is_question_marker(self, word: str) -> bool:
        """Check if word is Q or Q."""
        return word.strip() in ["Q.", "Q"]
    
    def _is_answer_marker(self, word: str) -> bool:
        """Check if word is A or A."""
        return word.strip() in ["A.", "A"]
    
    def _is_legal_marker(self, word: str) -> bool:
        """Check for common legal transcript markers to skip."""
        legal_patterns = [
            r'^MR\.',
            r'^MS\.',
            r'^THE$',
            r'^COURT:?$',
            r'^BY$',
            r'^\([^)]*\)$',  # Parenthetical
        ]
        
        for pattern in legal_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return True
        return False
    
    def _add_to_question(self, word: str):
        """Add word to current question."""
        if self.current_q:
            self.current_q += f" {word}"
        else:
            self.current_q = word
    
    def _add_to_answer(self, word: str):
        """Add word to current answer."""
        if self.current_a:
            self.current_a += f" {word}"
        else:
            self.current_a = word
    
    def _create_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Create Q&A pair with global sequence number."""
        global global_sequence_counter
        # Use blocking call since we need immediate sequence number
        global_sequence_counter += 1
        sequence_num = global_sequence_counter
        
        self.pair_counter += 1
        return {
            "id": f"pair_{self.pair_counter}_{uuid.uuid4().hex[:8]}",
            "sequence_number": sequence_num,
            "question": question.strip(),
            "answer": answer.strip(),
            "timestamp": datetime.utcnow().isoformat(),
            "word_count": len(answer.split()),
            "state": PairState.FORMING
        }
    
    def _add_pair_to_buffer(self, pair: Dict[str, Any]):
        """Add pair to buffer with overflow protection."""
        self.pairs_buffer.append(pair)
        
        # Remove old pairs if buffer gets too big
        while len(self.pairs_buffer) > self.max_pairs:
            removed = self.pairs_buffer.pop(0)
            print(f"🗑️ BUFFER_OVERFLOW: Removed {removed['id']}")
    
    def get_ready_pairs(self) -> List[Dict[str, Any]]:
        """Get all pairs ready for processing."""
        return [p for p in self.pairs_buffer if p["state"] == PairState.READY_TO_PROCESS]
    
    def mark_processing(self, pair_id: str):
        """Mark pair as processing."""
        for pair in self.pairs_buffer:
            if pair["id"] == pair_id:
                pair["state"] = PairState.PROCESSING
                break
    
    def mark_completed(self, pair_id: str):
        """Remove completed pair from buffer."""
        self.pairs_buffer = [p for p in self.pairs_buffer if p["id"] != pair_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        states = {}
        for pair in self.pairs_buffer:
            state = pair["state"].value
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_words": len(self.word_buffer),
            "buffer_size": len(self.pairs_buffer),
            "states": states,
            "current_question": self.current_q[:50] if self.current_q else "",
            "current_answer": self.current_a[:50] if self.current_a else "",
            "in_answer_mode": self.in_answer
        }

def embed_sync(texts: List[str]):
    """Synchronous wrapper for embedding."""
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

def add_chunk_to_chroma(chunk: Dict[str,Any]):
    """Add chunk to Chroma with error handling."""
    doc_text = chunk.get("clean_text") or chunk.get("raw_text") or ""
    emb = embed_sync([doc_text])[0]
    try:
        metadata = {}
        for key, value in chunk.items():
            if key not in ["chunk_id", "clean_text", "raw_text"] and value is not None:
                metadata[key] = str(value)
        
        collection.add(
            ids=[chunk["chunk_id"]],
            metadatas=[metadata],
            documents=[doc_text],
            embeddings=[emb]
        )
    except Exception as e:
        print(f"CHROMA_ERROR: {e}")

def retrieve_top_k_sync(query_text: str, k: int = TOP_K):
    """Retrieve top-k similar chunks."""
    qv = embed_sync([query_text])[0]
    results = collection.query(query_embeddings=[qv], n_results=k, include=["metadatas","documents","distances"])
    hits = []
    if results and "ids" in results and results["ids"]:
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })
    return hits

def make_chunk_from_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Create chunk from Q&A pair."""
    raw_text = f"Q. {pair['question']}\nA. {pair['answer']}"
    return {
        "chunk_id": str(uuid.uuid4()),
        "raw_text": raw_text,
        "clean_text": raw_text,
        "type": "qna",
        "source": "live_stream",
        "pair_id": pair["id"],
        "sequence_number": pair["sequence_number"],
        "timestamp": pair["timestamp"]
    }

def regenerate_pdf(pairs: List[Dict[str, Any]], session_id: Optional[str] = None):
    """Generate PDF from Q&A pairs."""
    if session_id:
        pdf_filename = f"transcript_session_{session_id}.pdf"
    else:
        pdf_filename = PDF_OUT
        
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    line_height = 12
    c.setFont("Times-Roman", 11)
    
    header = f"Testimonial Transcript — {datetime.utcnow().isoformat()}Z"
    if session_id:
        header += f" (Session: {session_id})"
    c.drawString(margin, y, header)
    y -= 2 * line_height

    for pair in pairs:
        if y < margin + 50:
            c.showPage()
            y = height - margin
            c.setFont("Times-Roman", 11)
        
        meta_line = f"[Q&A #{pair.get('sequence_number', '?')}] {pair['timestamp']} ({pair['word_count']} words)"
        c.setFont("Times-Roman", 8)
        c.drawString(margin, y, meta_line)
        y -= line_height
        
        c.setFont("Times-Roman", 11)
        c.drawString(margin, y, f"Q. {pair['question'][:150]}")
        y -= line_height
        
        c.drawString(margin, y, f"A. {pair['answer'][:150]}")
        y -= line_height * 2
    
    c.save()
    print(f"📄 PDF_SAVED: {pdf_filename} with {len(pairs)} pairs")

# Session management functions
async def start_transcript_session():
    """Start a new transcript session."""
    global transcript_session_active, transcript_session_id, transcript_start_time
    
    async with session_lock:
        if transcript_session_active:
            return {"status": "error", "message": "Transcript session already active"}
        
        transcript_session_active = True
        transcript_session_id = f"session_{uuid.uuid4().hex[:8]}"
        transcript_start_time = datetime.utcnow()
        
        print(f"🟢 TRANSCRIPT_START: {transcript_session_id}")
        return {
            "status": "success", 
            "message": "Transcript session started",
            "session_id": transcript_session_id,
            "start_time": transcript_start_time.isoformat()
        }

async def end_transcript_session():
    """End the current transcript session and save PDF."""
    global transcript_session_active, transcript_session_id, transcript_start_time
    
    async with session_lock:
        if not transcript_session_active:
            return {"status": "error", "message": "No active transcript session"}
        
        session_id = transcript_session_id
        start_time = transcript_start_time
        
        # Save current transcript to PDF
        async with transcript_lock:
            await asyncio.to_thread(regenerate_pdf, running_transcript, session_id)
        
        # Reset session state
        transcript_session_active = False
        transcript_session_id = None
        transcript_start_time = None
        
        print(f"🛑 TRANSCRIPT_END: {session_id} - PDF saved")
        return {
            "status": "success",
            "message": "Transcript session ended and PDF saved",
            "session_id": session_id,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": datetime.utcnow().isoformat(),
            "pairs_saved": len(running_transcript)
        }

async def get_transcript_session_status():
    """Get current transcript session status."""
    async with session_lock:
        return {
            "active": transcript_session_active,
            "session_id": transcript_session_id,
            "start_time": transcript_start_time.isoformat() if transcript_start_time else None,
            "transcript_size": len(running_transcript)
        }

async def process_pair_with_fact_check(pair: Dict[str, Any], websocket: WebSocket):
    """Process one Q&A pair through fact-checking and send result back."""
    start_time = time.time()
    pair_id = pair["id"]
    
    print(f"🔄 PROCESSING: {pair_id} (seq: {pair['sequence_number']})")
    
    try:
        # Add to Chroma
        chunk = make_chunk_from_pair(pair)
        await asyncio.to_thread(add_chunk_to_chroma, chunk)
        
        # Get similar chunks
        hits = await asyncio.to_thread(retrieve_top_k_sync, chunk["clean_text"], TOP_K)
        
        # Fact check (no timeout - let it complete naturally)
        verdict_response = await asyncio.to_thread(
            check_claim_with_ollama, 
            f"Q: {pair['question']}\nA: {pair['answer']}", 
            hits, 
            "phi3:mini"
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Enhanced error handling for claim checker response
        try:
            # Handle Pydantic object properly with validation
            if not hasattr(verdict_response, 'verdict'):
                print(f"⚠️ WARNING: {pair_id} - Invalid verdict_response object, skipping")
                return
            
            verdict = verdict_response.verdict
            confidence = getattr(verdict_response, 'confidence', 0)
            explanation = getattr(verdict_response, 'explanation', "No explanation available")
            
            # Validate verdict is a string and not empty
            if not isinstance(verdict, str) or not verdict.strip():
                print(f"⚠️ WARNING: {pair_id} - Invalid verdict '{verdict}', skipping")
                return
            
            # Validate confidence is a number
            try:
                confidence = int(confidence) if confidence is not None else 0
                if not (0 <= confidence <= 100):
                    print(f"⚠️ WARNING: {pair_id} - Invalid confidence {confidence}, using 0")
                    confidence = 0
            except (ValueError, TypeError):
                print(f"⚠️ WARNING: {pair_id} - Invalid confidence type, using 0")
                confidence = 0
            
            # Validate explanation is a string
            if not isinstance(explanation, str):
                explanation = str(explanation) if explanation is not None else "No explanation available"
            
            result = {
                "type": "fact_check_result",
                "pair_id": pair_id,
                "sequence_number": pair["sequence_number"],
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation,
                "question": pair["question"],
                "answer": pair["answer"],
                "evidence_count": len(hits),
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Validate result JSON before sending
            try:
                json.dumps(result)  # Test if result can be serialized
            except (TypeError, ValueError) as json_error:
                print(f"⚠️ WARNING: {pair_id} - JSON serialization error: {json_error}, skipping")
                return
            
            # Send result back only if WebSocket is connected and result is valid
            if websocket.client_state.name == 'CONNECTED':
                try:
                    await websocket.send_text(json.dumps(result))
                    verdict_emoji = "✅" if verdict == "SUPPORT" else "❌" if verdict == "REFUTE" else "❓"
                    print(f"{verdict_emoji} RESULT: {pair_id} -> {verdict} ({confidence}%) in {processing_time}ms")
                except Exception as send_error:
                    print(f"⚠️ WARNING: {pair_id} - Error sending result: {send_error}")
                    return
            else:
                print(f"⚠️ WARNING: {pair_id} - WebSocket not connected, skipping send")
                
        except Exception as parsing_error:
            print(f"⚠️ WARNING: {pair_id} - Error parsing claim checker response: {parsing_error}, skipping")
            return
        
    except Exception as e:
        # Only log the error, don't send it to frontend
        print(f"💥 ERROR: {pair_id} -> {e}")
        # Continue processing without sending error to frontend

async def background_fact_checker(buffer: QAPairBuffer, websocket: WebSocket):
    """Background worker - processes ready pairs one by one during active sessions."""
    print("FACT_CHECK_WORKER: Started")
    
    try:
        while True:
            # Only process if transcript session is active
            if transcript_session_active:
                # Get ready pairs from buffer
                ready_pairs = buffer.get_ready_pairs()
                
                # Process each ready pair (one by one)
                for pair in ready_pairs:
                    buffer.mark_processing(pair["id"])
                    
                    try:
                        await process_pair_with_fact_check(pair, websocket)
                    except Exception as e:
                        print(f"TASK_ERROR: {pair['id']} -> {e}")
                    finally:
                        buffer.mark_completed(pair["id"])
            
            # Check buffer every 100ms
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        print("FACT_CHECK_WORKER: Cancelled")
        raise

async def background_appender(buffer: QAPairBuffer):
    """Background worker for appending to transcript during active sessions."""
    print("APPEND_WORKER: Started")
    
    try:
        while True:
            # Only process if transcript session is active
            if transcript_session_active:
                # Get ready pairs from buffer
                ready_pairs = buffer.get_ready_pairs()
                
                for pair in ready_pairs:
                    buffer.mark_processing(pair["id"])
                    
                    try:
                        async with transcript_lock:
                            running_transcript.append(pair)
                            print(f"📄 APPENDED: {pair['id']} (seq: {pair['sequence_number']}) -> size: {len(running_transcript)}")
                    except Exception as e:
                        print(f"APPEND_ERROR: {pair['id']} -> {e}")
                    finally:
                        buffer.mark_completed(pair["id"])
            
            # Check buffer every 100ms
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        print("APPEND_WORKER: Cancelled")
        raise

@app.websocket("/ws/check")
async def ws_check(websocket: WebSocket):
    """Check endpoint with robust fact-checking and event handling."""
    await websocket.accept()
    print("WS_CHECK_CONNECTED: Fact-checking endpoint ready")
    
    buffer = QAPairBuffer(max_pairs=30)  # Larger buffer for fast streams
    worker_task = asyncio.create_task(background_fact_checker(buffer, websocket))
    
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"⚠️ WARNING: Invalid JSON received: {e}")
                print(f"   Raw message: {msg[:100]}{'...' if len(msg) > 100 else ''}")
                continue

            # Handle transcript events with error handling
            if payload.get("type") == "transcript.start":
                try:
                    result = await start_transcript_session()
                    response = {
                        "type": "transcript_event",
                        "event": "start_transcript",
                        **result
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error handling transcript.start: {e}")
                    continue
                continue
                
            elif payload.get("type") == "transcript.end":
                try:
                    result = await end_transcript_session()
                    response = {
                        "type": "transcript_event",
                        "event": "end_transcript",
                        **result
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error handling transcript.end: {e}")
                    continue
                continue

            # Handle regular word processing (only during active sessions)
            word = payload.get("word", "").strip()
            if not word:
                continue
            
            # Only process words during active transcript sessions
            if transcript_session_active:
                # Add word to buffer
                ready_pairs = buffer.add_word(word)
                
                # Send stats periodically with error handling
                if len(buffer.word_buffer) % 25 == 0:
                    try:
                        stats = buffer.get_stats()
                        response = {
                            "type": "word_ack",
                            "words_processed": stats["total_words"],
                            "buffer_size": stats["buffer_size"],
                            "states": stats["states"],
                            "current_q": stats["current_question"],
                            "current_a": stats["current_answer"],
                            "session_active": transcript_session_active
                        }
                        # Validate response JSON before sending
                        json.dumps(response)  # Test serialization
                        await websocket.send_text(json.dumps(response))
                    except Exception as e:
                        print(f"⚠️ WARNING: Error sending stats: {e}")
                        continue
            else:
                # Send warning if trying to process words outside session
                try:
                    response = {
                        "type": "warning",
                        "message": "No active transcript session - start a session first"
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error sending warning: {e}")
                    continue

    except WebSocketDisconnect:
        print("WS_CHECK_DISCONNECTED")
        worker_task.cancel()
    except Exception as e:
        print(f"💥 WS_CHECK_ERROR: {e}")
        worker_task.cancel()

@app.websocket("/ws/append")
async def ws_append(websocket: WebSocket):
    """Append endpoint for transcript building with event handling."""
    await websocket.accept()
    print("WS_APPEND_CONNECTED: Transcript building ready")
    
    buffer = QAPairBuffer(max_pairs=30)
    worker_task = asyncio.create_task(background_appender(buffer))
    
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"⚠️ WARNING: Invalid JSON received: {e}")
                print(f"   Raw message: {msg[:100]}{'...' if len(msg) > 100 else ''}")
                continue

            # Handle transcript events with error handling
            if payload.get("type") == "transcript.start":
                try:
                    result = await start_transcript_session()
                    response = {
                        "type": "transcript_event",
                        "event": "start_transcript",
                        **result
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error handling transcript.start: {e}")
                    continue
                continue
                
            elif payload.get("type") == "transcript.end":
                try:
                    result = await end_transcript_session()
                    response = {
                        "type": "transcript_event",
                        "event": "end_transcript",
                        **result
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error handling transcript.end: {e}")
                    continue
                continue

            # Handle regular word processing (only during active sessions)
            word = payload.get("word", "").strip()
            if not word:
                continue
            
            # Only process words during active transcript sessions
            if transcript_session_active:
                # Add word to buffer
                ready_pairs = buffer.add_word(word)
                
                # Send stats periodically (no fact-check results)
                if len(buffer.word_buffer) % 25 == 0:
                    stats = buffer.get_stats()
                    await websocket.send_text(json.dumps({
                        "type": "word_ack",
                        "words_processed": stats["total_words"],
                        "buffer_size": stats["buffer_size"],
                        "transcript_size": len(running_transcript),
                        "session_active": transcript_session_active
                    }))
            else:
                # Send warning if trying to process words outside session
                try:
                    response = {
                        "type": "warning",
                        "message": "No active transcript session - start a session first"
                    }
                    # Validate response JSON before sending
                    json.dumps(response)  # Test serialization
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"⚠️ WARNING: Error sending warning: {e}")
                    continue

    except WebSocketDisconnect:
        print("WS_APPEND_DISCONNECTED")
        worker_task.cancel()
    except Exception as e:
        print(f"💥 WS_APPEND_ERROR: {e}")
        worker_task.cancel()

@app.get("/status")
async def get_status():
    session_status = await get_transcript_session_status()
    return {
        "status": "running",
        "transcript_size": len(running_transcript),
        "global_sequence": global_sequence_counter,
        **session_status
    }

@app.post("/start_transcript")
async def start_transcript():
    """Start a new transcript session via REST API."""
    return await start_transcript_session()

@app.post("/end_transcript")
async def end_transcript():
    """End the current transcript session via REST API."""
    return await end_transcript_session()

@app.get("/transcript_status")
async def get_transcript_status():
    """Get current transcript session status via REST API."""
    return await get_transcript_session_status()

@app.post("/clear")
async def clear_all():
    global running_transcript, global_sequence_counter
    async with transcript_lock:
        running_transcript.clear()
        global_sequence_counter = 0
    await asyncio.to_thread(regenerate_pdf, [])
    return {"status": "success", "message": "Everything cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)