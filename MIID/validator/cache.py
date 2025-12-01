"""
LRU Cache implementation for caching Nominatim API results.
"""
from collections import OrderedDict
from typing import Union


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation with maximum size limit.
    Evicts least recently used entries when the cache reaches max_size.
    """
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, Union[float, str, dict]] = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Union[float, str, dict, None]:
        """Get value from cache and move it to the end (most recently used)."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Union[float, str, dict]) -> None:
        """Add or update value in cache."""
        if key in self.cache:
            # Update existing entry and move to end
            self.cache.move_to_end(key)
        else:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
        self.cache[key] = value
        # Move to end to mark as most recently used
        self.cache.move_to_end(key)
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        self.cache.clear()

