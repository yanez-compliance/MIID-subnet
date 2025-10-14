#!/usr/bin/env python3
"""
Name Variant Generation Service

This service provides name variants to miners' name variant pool requests.
- GET /{original_name}: Provides name variants pool except for already consumed variations
- POST /{list of variants}: Marks variants as consumed for 20 minutes

Architecture:
- Queue-based whole pool generation with timeout management
- Multi-CPU pool generation with RAM monitoring
- File-based caching for each name
- Concurrent instance pool generation with whole pool background generation
"""

import asyncio
from asyncio.tasks import Task
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable
import hashlib
from contextlib import asynccontextmanager
import re
import time
import datetime
import copy

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging  # add this near the top


TRACE_LOG_FORMAT = (
    f"%(asctime)s | %(levelname)s | %(name)s:%(filename)s:%(lineno)s | %(message)s"
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # For app/uvicorn error logs (with timestamps)
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": TRACE_LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
        # For access logs — no client IP, no request line/URL
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": "%(levelprefix)s %(asctime)s status=%(status_code)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        # important: don't propagate or you'll get duplicate access lines
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},

        # optional: your own app logger so your prints become timestamped logs if you switch to logging
        "nvgen": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}

log = logging.getLogger("nvgen")
# Suppress gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'error'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '2'
os.environ.setdefault("GRPC_PYTHON_LOG_LEVEL", "ERROR")

API_TITLE = "Name Variant Generation Service"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
A FastAPI-based service that provides name variants to miners' name variant pool requests.

## Features

- Phonetic-aware name variant generation using Soundex, Metaphone, and NYSIIS algorithms
- Queue-based pool generation with timeout management
- Multi-CPU pool generation with RAM monitoring
- File-based caching for each name
- Consumed variant tracking with automatic expiration
- Resource monitoring to prevent system overload
"""

HOST = "0.0.0.0"
PORT = 8000

# Global state
pending_requests: Dict[str, asyncio.Event] = {}
answer_candidate_cache = {}
map_cache = {}
# Query parsing system
query_queue = asyncio.Queue()
parsed_query_cache = {}
query_processing_events = {}
worker_task = None
worker_running = False

class QueryParseRequest(BaseModel):
    query_text: str
    max_retries: Optional[int] = 10

class QueryParseResponse(BaseModel):
    query_text: str
    parsed_params: Dict
    status: str

async def query_parse_worker():
    """Background worker that processes query parsing requests"""
    global worker_running
    worker_running = True
    log.info("🔧 Query parse worker started")
    
    while worker_running:
        try:
            # Get request from queue with timeout
            request = await query_queue.get()
            
            query_text = request["query_text"]
            event = request["event"]
            cache_key = make_key([], query_text)
            
            version = ""
            try:
                strategy_path = Path(os.path.dirname(__file__), "versions.json")
                if strategy_path.exists():
                    with open(strategy_path, 'r') as f:
                        strategy_data = json.load(f)
                        version = strategy_data.get("versions", {"parser": strategy_data.get("version", "v1")}).get("parser", "")
            except Exception as e:
                # log.error(f"Error reading version.json: {e}")
                pass
            parsed_params = None
            # Import the appropriate module dynamically based on version
            if version not in (None, "", "default", "v1"):
                try:
                    # accept "v2" or "2", normalize to _v2
                    log.info(f"📝 Processing query parse request with version {version}: {cache_key[:8]}...")
                    ver = str(version).lstrip("_")
                    if not ver.startswith("v"):
                        ver = "v" + ver
                    modname = f"MIID.miner.parse_query_gpt_{ver}"
                    import importlib
                    module = importlib.import_module(modname)
                    importlib.reload(module)
                    func = getattr(module, "query_parser")
                    parsed_params = await func(
                        query_text=query_text,
                        max_retries=request.get("max_retries", 10)
                    )
                    log.info(f"✅ Query parsed successfully: {cache_key[:8]}...")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.info(f"❌ Query parsed failed: {cache_key[:8]}...")
            if not parsed_params:
                try:
                    log.info(f"📝 Processing query parse request with version default: {cache_key[:8]}...")
                    from MIID.miner.parse_query_gpt import query_parser
                    parsed_params = await query_parser(
                        query_text=query_text,
                        max_retries=request.get("max_retries", 10)
                    )
                    log.info(f"✅ Query parsed successfully: {cache_key[:8]}...")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.info(f"❌ Query parsed failed: {cache_key[:8]}...")
            # Cache the result
            parsed_query_cache[cache_key] = {
                "query_text": query_text,
                "parsed_params": parsed_params,
                "timestamp": asyncio.get_event_loop().time()
            }
        except asyncio.TimeoutError:
            # No requests in queue, continue loop
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            log.info(f"❌ Worker error: {e}")
            continue
        finally:
            # Signal completion
            event.set()
            query_queue.task_done()

    log.info("🔧 Query parse worker stopped")


def _load_nonlatin_module():
    """Dynamically load non-latin parser module based on versions.json."""
    try:
        strategy_path = Path(os.path.dirname(__file__), "versions.json")
        version = "v1"
        if strategy_path.exists():
            with open(strategy_path, 'r') as f:
                strategy_data = json.load(f)
                version = strategy_data.get("versions", {"nonlatin_parser": strategy_data.get("version", "v1")}).get("nonlatin_parser", "v1")
        ver = str(version).lstrip("_")
        if not ver.startswith("v"):
            ver = "v" + ver
        modname = f"MIID.miner.nonlatin_parser_{ver}"
        import importlib
        module = importlib.import_module(modname)
        importlib.reload(module)
        return module
    except Exception as e:
        # Fallback to v1 if anything goes wrong
        bt.logging.error(f"Error loading non-latin parser module: {e}")
        try:
            from MIID.miner import nonlatin_parser_v1 as module  # type: ignore
            return module
        except Exception:
            return None


def is_latin_name(name: str) -> bool:
    """Versioned wrapper for latin-script detection."""
    module = _load_nonlatin_module()
    if module and hasattr(module, "is_latin_name"):
        try:
            return bool(module.is_latin_name(name))  # type: ignore[attr-defined]
        except Exception:
            pass
    # Fallback: ASCII-only heuristic
    try:
        return name.isascii()
    except Exception:
        return True

async def get_transliteration(name: str) -> str:
    """Versioned wrapper for transliteration of a single name. Returns input if no transliterator available."""
    module = _load_nonlatin_module()
    if not module:
        return name
    # Prefer common function names if provided by the module
    translit_fn = None
    for fn_name in ("transliterate_to_latin",):
        if hasattr(module, fn_name):
            translit_fn = getattr(module, fn_name)
            break
    if not translit_fn:
        return name
    try:
        result = translit_fn(name)
        if asyncio.iscoroutine(result):
            return await result
        return str(result)
    except Exception:
        return name


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    global worker_task
    
    # Startup
    log.info("🚀 Starting nvgen service...")
    worker_task = asyncio.create_task(query_parse_worker())
    
    yield
    
    # Shutdown
    log.info("🛑 Shutting down nvgen service...")
    global worker_running
    worker_running = False
    
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass



app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

def run_single_generation(args):
    """Helper function for multiprocessing that generates variations for a single name"""
    i, name, query_params, timeout, key = args
    # Read version string from strategy.json file if exists, else use version v1
    version = ""
    try:
        strategy_path = Path(os.path.dirname(__file__), "versions.json")
        if strategy_path.exists():
            with open(strategy_path, 'r') as f:
                strategy_data = json.load(f)
                version = strategy_data.get("versions", {"generator": strategy_data.get("version", "v1")}).get("generator", "")
    except Exception as e:
        # log.error(f"Error reading version.json: {e}")
        pass
    # Import the appropriate module dynamically based on version
    if version in (None, "", "default", "v1"):
        modname = "MIID.miner.generate_name_variations"
    else:
        # accept "v2" or "2", normalize to _v2
        ver = str(version).lstrip("_")
        if not ver.startswith("v"):
            ver = "v" + ver
        modname = f"MIID.miner.generate_name_variations_{ver}"
    try:
        log.info(f"Running {modname} for {name}")
        import importlib
        module = importlib.import_module(modname)
        importlib.reload(module)
        func = getattr(module, "generate_name_variations")
        return func(
            original_name=name,
            query_params=query_params,
            timeout=timeout,
            key=key
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        from MIID.miner.generate_name_variations import generate_name_variations
        return generate_name_variations(
            original_name=name,
            query_params=query_params,
            timeout=timeout,
            key=key
        )

def make_key(names: List[str], query_template: str) -> str:
    import hashlib
    return hashlib.sha256(str(names).encode() + query_template.encode()).hexdigest()[:6]

async def get_or_parse_query(query_text: str, max_retries: int = 10) -> Dict:
    """Get parsed query from cache or queue for parsing"""
    cache_key = make_key([], query_text)
    
    # Check cache first
    if cache_key in parsed_query_cache:
        cached_result = parsed_query_cache[cache_key]
        if cached_result.get("parsed_params") is not None:
            log.info(f"📋 Cache hit for query: {cache_key[:8]}...")
            return cached_result["parsed_params"]
        elif cached_result.get("error"):
            log.info(f"❌ Cache hit with error for query: {cache_key[:8]}...")
            raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    # Check if already being processed
    if cache_key in query_processing_events:
        log.info(f"⏳ Query already being processed: {cache_key[:8]}...")
        event = query_processing_events[cache_key]
        await event.wait()
        
        # Check cache again after waiting
        if cache_key in parsed_query_cache:
            cached_result = parsed_query_cache[cache_key]
            if cached_result.get("parsed_params") is not None:
                return cached_result["parsed_params"]
            elif cached_result.get("error"):
                raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    # Create new processing event
    event = asyncio.Event()
    query_processing_events[cache_key] = event
    
    
    # Add to queue
    
    await query_queue.put({
        "query_text": query_text,
        "event": event,
        "max_retries": max_retries
    })
    
    log.info(f"📝 Queued query for parsing: {cache_key[:8]}...")
    
    # Wait for completion
    await event.wait()
    
    # Clean up event
    del query_processing_events[cache_key]
    
    # Return result
    if cache_key in parsed_query_cache:
        cached_result = parsed_query_cache[cache_key]
        if cached_result.get("parsed_params") is not None:
            return cached_result["parsed_params"]
        elif cached_result.get("error"):
            raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    raise HTTPException(status_code=500, detail="Query parsing failed")

@app.post("/parse_query")
async def parse_query_endpoint(request: QueryParseRequest) -> QueryParseResponse:
    """Parse a query template using the background worker"""
    try:
        parsed_params = await get_or_parse_query(request.query_text, request.max_retries)
        return QueryParseResponse(
            query_text=request.query_text,
            parsed_params=parsed_params,
            status="success"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/parse_query/{query_hash}")
async def get_parsed_query(query_hash: str) -> QueryParseResponse:
    """Get a parsed query from cache by its hash"""
    if query_hash in parsed_query_cache:
        cached_result = parsed_query_cache[query_hash]
        return QueryParseResponse(
            query_text=cached_result["query_text"],
            parsed_params=cached_result["parsed_params"],
            status="cached" if cached_result.get("parsed_params") else "error"
        )
    else:
        raise HTTPException(status_code=404, detail="Query not found in cache")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    total_queries = len(parsed_query_cache)
    successful_parses = sum(1 for result in parsed_query_cache.values() if result.get("parsed_params") is not None)
    failed_parses = sum(1 for result in parsed_query_cache.values() if result.get("error"))
    
    return {
        "total_queries": total_queries,
        "successful_parses": successful_parses,
        "failed_parses": failed_parses,
        "queue_size": query_queue.qsize(),
        "pending_processing": len(query_processing_events)
    }

async def calculate_answer_candidate(names: List[str], query_template: str, query_params: Optional[Dict] = None, timeout: Optional[float] = None) -> List[str]:
    """
    Calculate answer candidate from names and query template
    """
    try:
        if query_params is None:
            query_params = await get_or_parse_query(query_template, max_retries=1)
        if query_params is None:
            log.info(f"Failed to parse query: {query_template}")
            return []
    except Exception as e:
        log.info(f"Failed to parse query: {e}")
        return []
    
    task_args = []
    max_cpus = os.cpu_count()
    max_workers = max(1, min(8, len(names)))
    timeout_per_worker = int(timeout / max_workers)
    key = make_key(names, query_template)
    for i, name in enumerate(names):
        args = (i, name, query_params, timeout_per_worker, key)
        task_args.append(args)
    
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, run_single_generation, args) for args in task_args]
        try:
            results = await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            log.info(f"Timeout waiting for {len(tasks)} tasks to complete")
            return []
        return results

from MIID.miner.generate_name_variations import AnswerCandidate
class AnswerCandidateForNoisy:
    def __init__(self, task_key: str, answer_candidates: List[AnswerCandidate], validator_uid: int, query_template: str):
        self.task_key = task_key
        self.answer_candidates = answer_candidates
        self.serial = 0
        self.buckets_exact = {}
        self.answer_list = []
        self.miner_list = []
        self.first_time = time.time()
        self.start_at = time.time()
        self.validator_uid = validator_uid
        self.query_template = query_template

    def calc_reward_and_penalty(self, answers: List[Dict[str, List[str]]], miners: List[int]):
        from types import SimpleNamespace
        responses = {}
        responses = [SimpleNamespace(
            variations=answer
        ) for answer in answers]
        # Calculate rule-based metadata
        rule_based = {
            "selected_rules": self.answer_candidates[0].query_params["selected_rules"],
            "rule_percentage": self.answer_candidates[0].query_params["rule_percentage"] * 100
        }
        import bittensor as bt
        debug_level = bt.logging.get_level()
        bt.logging.setLevel('WARNING')
        from MIID.validator.reward_v12 import get_name_variation_rewards
        _, metrics = get_name_variation_rewards(
            None,
            seed_names=list(answers[0].keys()), 
            responses=responses,
            uids=miners,
            variation_count=self.answer_candidates[0].query_params["variation_count"],
            phonetic_similarity=self.answer_candidates[0].query_params["phonetic_similarity"],
            orthographic_similarity=self.answer_candidates[0].query_params["orthographic_similarity"],
            rule_based=rule_based,
        )
        penalty = False
        for metric in metrics:
            if 'collusion' in metric['penalties'] or 'duplication' in metric['penalties']:
                penalty = True
                break
        return metrics, penalty
    
    def get_next_answer_for(self, miner_uid: int) -> Set[str]:
        COLLUSION_GROUP_SIZE_THRESHOLD = 1
        bucket_no = 0
        while True:
            answer = {}
            metric = {}
            try_count = 100
            f_serial = os.path.join(os.path.dirname(__file__), "tasks", f"serial_{self.task_key}.txt")
            if os.path.exists(f_serial):
                with open(f_serial, "r") as f:
                    self.serial = int(f.read())
            else:
                self.serial = 0
            while try_count >= 0:
                self.serial += 1
                try_count -= 1
                from MIID.miner.kth_plan import kth_plan
                noisy_plan, noisy_count = kth_plan(len(self.answer_candidates), max(0, self.serial - COLLUSION_GROUP_SIZE_THRESHOLD) + 1)
                for i,cand in enumerate(self.answer_candidates):
                    answer[cand.name] = cand.get_next_answer(noisy_plan[i])
                # metrics, penalty = self.calc_reward_and_penalty(self.answer_list + [answer], self.miner_list + [miner_uid])
                # if not penalty:
                #     self.answer_list.append(answer)
                #     self.miner_list.append(miner_uid)
                #     self.metrics = metrics
                #     break
            metrics = [{}]
            self.metrics = metrics
            with open(f_serial, "w") as f:
                f.write(str(self.serial))
            if try_count < 0:
                for cand in self.answer_candidates:
                    cand.increase_bucket_no()
                bucket_no += 1
                if bucket_no >= len(self.answer_candidates[0].bucket):
                    break
                continue
            return answer, metrics[-1]
        return answer, metrics[-1]
        


class TaskRequest(BaseModel):
    names: List[str]
    query_template: str
    query_params: Optional[Dict] = None
    timeout: Optional[float] = None
    miner_uid: Optional[int] = None
    validator_uid: Optional[int] = None

def save_result(answer_candidate: AnswerCandidateForNoisy, miner_uid: int):
    timestamp = datetime.datetime.fromtimestamp(answer_candidate.start_at).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        os.path.dirname(__file__),
        "tasks",
        f"validator_{answer_candidate.validator_uid}",        
        f"{timestamp}-{answer_candidate.task_key}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        # with open(os.path.join(run_dir, f"serial_{answer_candidate.serial:02d}-miner_{miner_uid}.json"), 'w', encoding="utf-8") as f:
        #     output_json = answer_candidate.metrics[-1]
        #     json.dump(output_json, f, indent=4)

        with open(os.path.join(run_dir, f"_task.json"), 'w', encoding="utf-8") as f:
            json.dump(
                {
                    "names": [ cand.name for cand in answer_candidate.answer_candidates ],
                    "query_template": answer_candidate.query_template,
                    "query_template_hash": answer_candidate.task_key,
                    "query_params": answer_candidate.answer_candidates[0].query_params
                }, f, indent=4)

        metrics = answer_candidate.metrics[-1]
        count_matrix = {}
        phonetic_score = {}
        orthographic_score = {}
        nonrule_count = {}
        # for name in metrics['name_metrics']:
        #     name_metrics = metrics['name_metrics'][name]
        #     for sub_name in [f"first_name", "last_name"]:
        #         sub_count_matrix = [[0 for _ in range(8)] for _ in range(4)]
        #         sub_name_data = name_metrics.get(sub_name, [])
        #         if sub_name_data:
        #             variation_scores = sub_name_data['metrics'].get('variations', [])
        #             for variation in variation_scores:
        #                 from MIID.miner.pool_generator import orth_level, orth_sim, phon_class, seed_codes
                        
        #                 o_level = orth_level(orth_sim(name.split(" ")[0] if sub_name == "first_name" else name.split(" ")[1],variation['variation']))
        #                 p_level = phon_class(seed_codes(name.split(" ")[0] if sub_name == "first_name" else name.split(" ")[1]), variation['variation'])
        #                 if o_level is None:
        #                     o_level = 3
        #                 sub_count_matrix[o_level][p_level] += 1
        #         count_matrix[name + " - " + sub_name] = sub_count_matrix
        #         phonetic_score[name + " - " + sub_name] = sub_name_data['metrics']['similarity']['phonetic']
        #         orthographic_score[name + " - " + sub_name] = sub_name_data['metrics']['similarity']['orthographic']
                # nonrule_count[name + " - " + sub_name] = sub_name_data['metrics']['count']['actual']

        output = ""
        for subname in count_matrix:
            mat = count_matrix[subname]
            count = nonrule_count[subname]
            phonetic = phonetic_score[subname]
            orthographic = orthographic_score[subname]
            output += f"{subname}\n - Count: {count}\n - Phonetic: {phonetic:.2f}\n - Orthographic: {orthographic:.2f}\n"
            from MIID.miner.utils import _mat_str84
            output += _mat_str84(mat)

        with open(os.path.join(run_dir, f"_count_matrix.txt"), 'w', encoding="utf-8") as f:
            f.write(output)

    except Exception as e:
        log.error(f"Error saving serial_{answer_candidate.serial}-miner_{miner_uid}.json: {e}")
        pass
    

@app.post("/task")
async def solve_task(request: TaskRequest, background_tasks: BackgroundTasks = None):
    """
    POST /task: Solve task with name variations generation
    
    Solve task by generating name variants pool for each name in names, then solving the query template with the pool.
    If timeout parameter is specified, use it for instance pool generation instead of default timeout.
    Handles concurrent requests by making other requests wait for the first one to complete.
    """
    def clear_cache():
        if len(answer_candidate_cache) >= 100:
            try:
                os.remove(os.path.join(os.path.dirname(__file__), "tasks", f"serial_{list(answer_candidate_cache.keys())[0]}.txt"))
            except Exception as e:
                pass
            del answer_candidate_cache[list(answer_candidate_cache.keys())[0]]
            del pending_requests[list(pending_requests.keys())[0]]
    clear_cache()
    names = request.names    
    query_template = request.query_template
    query_params = request.query_params
    timeout = request.timeout
    non_latin_names_map = {}
    
    task_key = make_key(names, query_template)
    start_at = time.time()

    if task_key in pending_requests:
        log.info(
            f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
            f"Waiting pending request hit for {task_key}")
        await asyncio.wait_for(pending_requests[task_key].wait(), timeout=timeout)
        if task_key in answer_candidate_cache:
            log.info(
                f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
                f"Finished waiting pending request hit. Using cached answer candidate for {task_key}")
            answer_candidate = answer_candidate_cache[task_key]
        if task_key in map_cache:
            non_latin_names_map = map_cache[task_key]
    elif task_key in answer_candidate_cache:
        log.info(
            f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
            f"Using cached answer candidate for {task_key}")
        answer_candidate = answer_candidate_cache[task_key]
        if task_key in map_cache:
            non_latin_names_map = map_cache[task_key]
    else:
        latin_names = [name for name in names if is_latin_name(name)]
        non_latin_names = [name for name in names if not is_latin_name(name)]
        
        for name in non_latin_names:
            converted_name = await get_transliteration(name)
            non_latin_names_map[name] = converted_name
        
        non_latin_names = list(non_latin_names_map.values())
        
        log.info(f"Non-latin names: {non_latin_names_map}")

        log.info(
            f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
            f"Calculate answer candidate for {task_key} with timeout {timeout: .1f}s")
        pending_requests[task_key] = asyncio.Event()
        try:
            answer_candidate_latin = await calculate_answer_candidate(latin_names, query_template, query_params, timeout)
            
            # Make a copy of query_params with rule count set to 0
            
            query_params_no_rule = None
            if query_params is not None:
                query_params_no_rule = copy.deepcopy(query_params)
                query_params_no_rule["rule_percentage"] = 0.0
                query_params_no_rule["selected_rules"] = []
            
            answer_candidate_non_latin = await calculate_answer_candidate(non_latin_names, query_template, query_params_no_rule, timeout)
            answer_candidate = answer_candidate_latin + answer_candidate_non_latin
            
            answer_candidate = AnswerCandidateForNoisy(task_key, answer_candidate, validator_uid=request.validator_uid, query_template=query_template)
            answer_candidate_cache[task_key] = answer_candidate
            map_cache[task_key] = non_latin_names_map
            pending_requests[task_key].set()
            log.info(
                f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
                f"Finished calculating answer candidate for {task_key}")
        except asyncio.TimeoutError:
            pending_requests[task_key].set()
            log.info(
                f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
                f"Timeout waiting for pool generation for {task_key}")
            raise HTTPException(status_code=408, detail="Request timeout waiting for pool generation")
    
    answer, metric = answer_candidate.get_next_answer_for(request.miner_uid)
    
    # convert non-latin names back to original names
    answer_temp = copy.deepcopy(answer)
    for name in answer:
        if name in non_latin_names_map.values():
            key = list(non_latin_names_map.keys())[list(non_latin_names_map.values()).index(name)]
            answer_temp[key] = answer[name]
            del answer_temp[name]

    answer = answer_temp
    
    log.info(f"non-latin names map: {non_latin_names_map}")
    log.info(f"names list: {names}")
    log.info(f"name variation key: {answer.keys()}")
    missing_names = set(names) - set(answer.keys())
    log.info(f"missing names: {missing_names}")
    
    log.info(
        f"Miner {request.miner_uid} (validator: {request.validator_uid}, task:{task_key}): "
        f"Answer candidate: {answer_candidate.serial}, "
        f"Timeout: {time.time() - start_at: .1f}s, "
        f"Metric: {metric.get('final_reward', '0')}")
    save_result(answer_candidate, request.miner_uid)
    return {name: list(answer[name]) for name in answer}, metric, answer_candidate.answer_candidates[0].query_params

if __name__ == "__main__":
    import argparse
    port = argparse.ArgumentParser()
    import bittensor as bt
    bt.logging.add_args(port)
    port.add_argument("--port", type=int, default=PORT)
    args = port.parse_args()
    PORT = args.port
    log.info(f"Starting nvgen service on port {PORT}")

    try:
        import uvicorn
        # Start the server
        uvicorn.run(
            app, 
            host=HOST, 
            port=PORT,
            log_level="info",
            access_log=False,
            log_config=LOGGING_CONFIG,
            limit_concurrency=1000,
            limit_max_requests=None,
        )
        
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        log.info("\n🛑 KeyboardInterrupt received, shutting down gracefully...")
        log.info("👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        log.info(f"\n❌ Error starting service: {e}")
        sys.exit(1)
