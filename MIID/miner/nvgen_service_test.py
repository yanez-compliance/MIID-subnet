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
import json
import os
import bittensor as bt
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from MIID.miner.nvgen_service import TaskRequest, solve_task, log, query_parse_worker

async def test():
    base_query_files = [
        os.path.join(os.path.dirname(__file__), "1.json"),
    ]
    asyncio.create_task(query_parse_worker())
    dup = 1
    query_files = []
    for q in base_query_files:
        query_files.extend([q] * dup)
    
    tasks = []
    for i in range(len(query_files)):
        with open(query_files[i], 'r') as f:
            query_data = json.load(f)
            task = TaskRequest(
                names=query_data['names'],
                query_template=query_data.get('query_template', None),
                query_params=query_data.get('query_params', None),
                miner_uid=i,
                validator_uid=0,
                timeout=700.0)
            tasks.append(solve_task(task))
    results = await asyncio.gather(*tasks)

if __name__ == "__main__":
    
    if not log.handlers:  # prevent duplicates when dictConfig runs later
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.info("nvgen service test started")
    asyncio.run(test())