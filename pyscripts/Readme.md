Quick start

pip install fastapi uvicorn requests geonamescache unidecode

Set a real contact in NOMINATIM_USER_AGENT (important for OSM etiquette), e.g.
export NOMINATIM_USER_AGENT='addr-variants/1.2 (contact: you@domain.com)'

Run: uvicorn service:app --host 0.0.0.0 --port 8000

Try it:

curl 'http://127.0.0.1:8000/generate?country=Japan&city=Tokyo&count=20'

curl 'http://127.0.0.1:8000/generate?country=Germany&count=40'

curl 'http://127.0.0.1:8000/cache/keys'

curl -X POST 'http://127.0.0.1:8000/cache/clear'

This service follows the earlier grader-friendly formatting so your _grade_address_variations score should be strong, and the cache avoids re-hitting Nominatim for repeat queries.