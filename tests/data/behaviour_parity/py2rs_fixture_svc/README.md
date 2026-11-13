# py2rs-fixture-svc (vendored test copy)

Copy of the PRIVATE fixture `conveyor-coding/py2rs-fixture-svc` at commit `9aad739`
(src, migrations, alembic.ini, parity/requests.jsonl; imports re-sorted by this repo's ruff,
nothing else changed). The behaviour-parity driver tests (py2rust A5c) run this real service
under the real driver against an ephemeral `postgres:18`, so CI exercises record/replay
without the fixture repository. It runs on this repo's own venv (fastapi, sqlalchemy,
psycopg, alembic, httpx, uvicorn are all dependencies here). Never edit to make a test pass:
the mutation test copies it to a temp dir and mutates the copy.
