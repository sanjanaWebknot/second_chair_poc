# ws_server_enhanced.py - Enhanced with concurrent processing and deduplication
import os
import uuid
import json
import asyncio
import re
import time
from typing import List, Dict, Any, Optional
from collections import deque
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import chromadb
from embeddings import get_sentence_transformer  
from chroma_utils import create_chroma_client    
from claim_checker import check_claim_with_ollama 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Set tokenizers parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_DIR = "./chroma_db_second_chair"
COLLECTION_NAME = "second_chair_depositions"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
REFUTE_CONFIDENCE_THRESHOLD = 60   
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

class PairState(Enum):
    FORMING = "forming"
    READY_TO_PROCESS = "ready_to_process"
    PROCESSING = "processing"
    COMPLETED = "completed"

class QAPairBuffer:
    """Enhanced Q&A pair detection with concurrent processing support."""
    
    def __init__(self, max_pairs: int = 10):
        self.max_pairs = max_pairs
        self.word_buffer: List[str] = []
        self.pairs_buffer: List[Dict[str, Any]] = []  
        self.current_q: str = ""
        self.current_a: str = ""
        self.in_answer: bool = False
        self.pair_counter: int = 0
        self.sequence_counter: int = 0
        self.processing_tasks: Dict[str, asyncio.Task] = {}  
        
    def add_word(self, word: str) -> List[Dict[str, Any]]:
        """Add word and return any pairs ready for processing."""
        ready_pairs = []
        
        # Debug every 20 words
        if len(self.word_buffer) % 20 == 0:
            print(f"🐛 DEBUG: Word '{word}' | Current Q: '{self.current_q[:30]}...' | Current A: '{self.current_a[:30]}...' | In Answer: {self.in_answer}")
        
        if re.search(r'[Qq]\.', word) or word.lower() in ['q.', 'q']:
            print(f"🔍 Found Q. pattern in word: '{word}'")
            
            if self.current_q and self.current_a and self.in_answer:
                completed_pair = self._create_pair(self.current_q, self.current_a)
                self._add_pair_to_buffer(completed_pair)
                print(f"📦 COMPLETED PAIR: {completed_pair['id']} Q={self.current_q[:30]}... A={self.current_a[:30]}...")
                
                ready_pairs.extend(self._mark_ready_pairs())
            
            self.current_q = ""
            self.current_a = ""
            self.in_answer = False
            print(f"🔤 STARTING NEW Q")
            
        elif re.search(r'[Aa]\.', word) or word.lower() in ['a.', 'a']:
            print(f"Found A. pattern in word: '{word}'")
            if self.current_q:
                self.current_a = ""
                self.in_answer = True
                print(f"STARTING NEW A")
            else:
                print(f"Found A. but no current question!")
                
        else:
            # Regular word - add to current Q or A
            if self.in_answer:
                self._add_to_answer(word)
            elif self.current_q or not self.in_answer:  # Building question
                self._add_to_question(word)
        
        self.word_buffer.append(word)
        return ready_pairs
    
    def _mark_ready_pairs(self) -> List[Dict[str, Any]]:
        """Mark pairs as ready to process (all but the last one)."""
        ready_pairs = []
        
        forming_pairs = [p for p in self.pairs_buffer if p["state"] == PairState.FORMING]
        if len(forming_pairs) > 1:  # Keep the last forming pair
            for pair in forming_pairs[:-1]:
                pair["state"] = PairState.READY_TO_PROCESS
                ready_pairs.append(pair)
                print(f"Marked {pair['id']} as READY_TO_PROCESS")
        
        return ready_pairs
    
    def _add_pair_to_buffer(self, pair: Dict[str, Any]):
        """Add pair to buffer with state management."""
        pair["state"] = PairState.FORMING
        self.pairs_buffer.append(pair)
        
        while len(self.pairs_buffer) > self.max_pairs:
            removed = self.pairs_buffer.pop(0)
            print(f"🗑️ Removed old pair from buffer: {removed['id']}")
    
    def _add_to_question(self, word: str):
        """Add word to current question."""
        if (re.search(r'[QqAa]\.', word) or 
            word.lower() in ['q.', 'q', 'a.', 'a']):
            return
            
        if self.current_q:
            self.current_q += f" {word}"
        else:
            self.current_q = word
    
    def _add_to_answer(self, word: str):
        """Add word to current answer."""
        if (re.search(r'[QqAa]\.', word) or 
            word.lower() in ['q.', 'q', 'a.', 'a']):
            return
            
        if self.current_a:
            self.current_a += f" {word}"
        else:
            self.current_a = word
    
    def _create_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Create a Q&A pair with unique ID and sequence number."""
        self.pair_counter += 1
        self.sequence_counter += 1
        return {
            "id": f"pair_{self.pair_counter}_{uuid.uuid4().hex[:8]}",
            "sequence_number": self.sequence_counter,
            "question": question.strip(),
            "answer": answer.strip(),
            "timestamp": datetime.utcnow().isoformat(),
            "word_count": len(answer.split()),
            "state": PairState.FORMING
        }
    
    def get_ready_pairs(self) -> List[Dict[str, Any]]:
        """Get all pairs ready for processing."""
        return [p for p in self.pairs_buffer if p["state"] == PairState.READY_TO_PROCESS]
    
    def mark_processing(self, pair_id: str):
        """Mark pair as currently being processed."""
        for pair in self.pairs_buffer:
            if pair["id"] == pair_id:
                pair["state"] = PairState.PROCESSING
                print(f" Marked {pair_id} as PROCESSING")
                break
    
    def mark_completed(self, pair_id: str):
        """Mark pair as completed and remove from buffer."""
        self.pairs_buffer = [p for p in self.pairs_buffer if p["id"] != pair_id]
        print(f"Completed and removed {pair_id} from buffer")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        states = {}
        for pair in self.pairs_buffer:
            state = pair["state"].value
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_pairs": len(self.pairs_buffer),
            "words_processed": len(self.word_buffer),
            "states": states,
            "active_tasks": len(self.processing_tasks)
        }

def embed_sync(texts: List[str]):
    """Synchronous wrapper for embedding."""
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

def add_chunk_to_chroma(chunk: Dict[str,Any]):
    """Add chunk to Chroma with proper metadata handling."""
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
        print("Chroma add error (ignored):", e)

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

def regenerate_pdf(pairs: List[Dict[str, Any]]):
    """Generate PDF from Q&A pairs."""
    c = canvas.Canvas(PDF_OUT, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    line_height = 12
    c.setFont("Times-Roman", 11)
    
    header = f"Testimonial Transcript — {datetime.utcnow().isoformat()}Z"
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
        
        # Answer
        c.drawString(margin, y, f"A. {pair['answer'][:150]}")
        y -= line_height * 2
    
    c.save()

async def process_pair_with_fact_check(pair: Dict[str, Any], websocket: WebSocket) -> Dict[str, Any]:
    """Process a single Q&A pair with fact checking."""
    start_time = time.time()
    pair_id = pair["id"]
    
    print(f"⚙️ Processing pair {pair_id} (seq: {pair['sequence_number']})...")
    
    try:
        chunk = make_chunk_from_pair(pair)
        
        await asyncio.to_thread(add_chunk_to_chroma, chunk)
        hits = await asyncio.to_thread(retrieve_top_k_sync, chunk["clean_text"], TOP_K)
        verdict = await asyncio.to_thread(check_claim_with_ollama, 
                                        f"Q: {pair['question']}\nA: {pair['answer']}", 
                                        hits, "phi3:mini")
        
        processing_time = int((time.time() - start_time) * 1000)
        print(f" Processed {pair_id} → {verdict.get('verdict')} ({verdict.get('confidence')}%) in {processing_time}ms")
        
        result = {
            "type": "fact_check_result",
            "pair_id": pair_id,
            "sequence_number": pair["sequence_number"],
            "verdict": verdict.get("verdict", "UNKNOWN"),
            "confidence": verdict.get("confidence", 0),
            "explanation": verdict.get("explanation", "No explanation provided"),
            "question": pair["question"],
            "answer": pair["answer"],
            "evidence_count": len(hits),
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Stream result immediately
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_text(json.dumps(result))
        
        return result
        
    except Exception as e:
        error_result = {
            "type": "fact_check_error",
            "pair_id": pair_id,
            "sequence_number": pair["sequence_number"],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_text(json.dumps(error_result))
        
        print(f" Error processing {pair_id}: {e}")
        return error_result

async def background_processor(buffer: QAPairBuffer, websocket: WebSocket):
    """Background coroutine that continuously processes ready pairs."""
    print("Background processor started")
    
    try:
        while True:
            # Get pairs ready for processing
            ready_pairs = buffer.get_ready_pairs()
            
            if ready_pairs:
                print(f"Found {len(ready_pairs)} pairs ready for processing")
                
                # Process pairs concurrently
                tasks = []
                for pair in ready_pairs:
                    # Mark as processing to prevent duplicate processing
                    buffer.mark_processing(pair["id"])
                    
                    # Create processing task
                    task = asyncio.create_task(
                        process_pair_with_fact_check(pair, websocket)
                    )
                    tasks.append((pair["id"], task))
                    buffer.processing_tasks[pair["id"]] = task
                
                # Wait for all tasks to complete
                for pair_id, task in tasks:
                    try:
                        await task
                        buffer.mark_completed(pair_id)
                        if pair_id in buffer.processing_tasks:
                            del buffer.processing_tasks[pair_id]
                    except Exception as e:
                        print(f"Task error for {pair_id}: {e}")
                        buffer.mark_completed(pair_id)  # Still remove from buffer
                        if pair_id in buffer.processing_tasks:
                            del buffer.processing_tasks[pair_id]
            
            # Sleep briefly before checking again
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        print("Background processor cancelled")
        # Clean up any remaining tasks
        for task in buffer.processing_tasks.values():
            task.cancel()
        raise
    except Exception as e:
        print(f"Background processor error: {e}")

@app.websocket("/ws/check")
async def ws_check(websocket: WebSocket):
    """Check endpoint - word by word streaming with concurrent processing."""
    await websocket.accept()
    print("🔍 CHECK endpoint connected")
    
    buffer = QAPairBuffer(max_pairs=10)
    
    processor_task = asyncio.create_task(background_processor(buffer, websocket))
    
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except Exception:
                await websocket.send_text(json.dumps({"type":"error","message":"invalid_json"}))
                continue

            word = payload.get("word", "").strip()
            if not word:
                continue
            
            ready_pairs = buffer.add_word(word)
            
            if len(buffer.word_buffer) % 10 == 0:
                stats = buffer.get_buffer_stats()
                await websocket.send_text(json.dumps({
                    "type": "word_ack",
                    "words_processed": stats["words_processed"],
                    "buffer_stats": stats,
                    "current_q": buffer.current_q[:50] if buffer.current_q else "",
                    "current_a": buffer.current_a[:50] if buffer.current_a else ""
                }))

    except WebSocketDisconnect:
        print("CHECK endpoint disconnected")
        processor_task.cancel()
    except Exception as e:
        print(f"Error in CHECK: {e}")
        processor_task.cancel()
        await websocket.close()

@app.websocket("/ws/append")
async def ws_append(websocket: WebSocket):
    """Append endpoint - word by word streaming with PDF building."""
    await websocket.accept()
    print("APPEND endpoint connected")
    
    buffer = QAPairBuffer(max_pairs=10)
    
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except Exception:
                await websocket.send_text(json.dumps({"type":"error","message":"invalid_json"}))
                continue

            word = payload.get("word", "").strip()
            if not word:
                continue
            
            ready_pairs = buffer.add_word(word)
            
            for pair in ready_pairs:
                if pair["state"] == PairState.READY_TO_PROCESS:
                    async with transcript_lock:
                        running_transcript.append(pair)
                        print(f"📄 Added pair {pair['id']} (seq: {pair['sequence_number']}) to transcript (total: {len(running_transcript)})")
                        
                        await asyncio.to_thread(regenerate_pdf, running_transcript)
                    
                    await websocket.send_text(json.dumps({
                        "type": "pair_stored",
                        "pair_id": pair["id"],
                        "sequence_number": pair["sequence_number"],
                        "question": pair["question"][:50] + "..." if len(pair["question"]) > 50 else pair["question"],
                        "answer": pair["answer"][:50] + "..." if len(pair["answer"]) > 50 else pair["answer"],
                        "transcript_size": len(running_transcript)
                    }))
            
            if len(buffer.word_buffer) % 15 == 0:
                stats = buffer.get_buffer_stats()
                await websocket.send_text(json.dumps({
                    "type": "word_ack",
                    "words_processed": stats["words_processed"],
                    "buffer_stats": stats,
                    "transcript_size": len(running_transcript)
                }))

    except WebSocketDisconnect:
        print("APPEND endpoint disconnected")
    except Exception as e:
        print(f"Error in APPEND: {e}")
        await websocket.close()

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "transcript_size": len(running_transcript)
    }

@app.post("/clear")
async def clear_all():
    global running_transcript
    async with transcript_lock:
        running_transcript.clear()
    await asyncio.to_thread(regenerate_pdf, [])
    return {"status": "success", "message": "Everything cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)