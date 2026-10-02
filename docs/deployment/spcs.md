# Run turbovec on Snowpark Container Services

This is a minimal, single-tenant demonstration of `IdMapIndex` behind a private Snowpark Container Services (SPCS) endpoint. A Snowflake service function calls a small HTTP adapter; it does not make turbovec a Snowflake-native index. See [issue #95](https://github.com/RyanCodrai/turbovec/issues/95) for the deployment context.

The example uses an eight-dimensional index and one service instance to make verification inexpensive. It stores vectors **only in process memory**: restarting, suspending, or rescheduling the service loses the index. This demo is not a reliable SQL interface: service-function calls can be retried or land on different instances, even when a single instance is configured. Do not use this example for production data or as a multi-tenant access-control boundary. A caller-controlled `tenant_id` or `allowlist` is not authorization; a production service must derive permitted IDs from a trusted policy source and fail closed.

## Prerequisites

- An account and role with permission to create an image repository, compute pool, service, and service function in a chosen database/schema. SPCS availability and instance-family support vary by region.
- Docker, the Snowflake CLI (`snow`), and an authenticated Snowflake connection. Build a `linux/amd64` image; SPCS requires this platform.
- A schema you can use for the demonstration. The SQL below assumes `USE DATABASE` and `USE SCHEMA` have already selected it. Pick a unique resource prefix if names collide.

The example uses the published `turbovec==1.0.0` Python distribution and the `IdMapIndex`/`search`/`add_with_ids` calls in the current [Python API reference](../api.md). Pin and review all dependencies for a production build.

## Container

Place these two files in an empty local directory. This small adapter accepts a single-row Snowflake service-function envelope (`{"data": [[row_number, payload]]}`) and returns the corresponding row. It intentionally has **no public ingress**, tenant field, persistence endpoint, or arbitrary filtering API.

`Dockerfile`:

```dockerfile
FROM --platform=linux/amd64 python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir turbovec==1.0.0 numpy==2.4.3 fastapi==0.115.0 uvicorn==0.49.0
COPY service.py /app/service.py
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

`service.py`:

```python
import numpy as np
from fastapi import FastAPI, HTTPException
from turbovec import IdMapIndex

app = FastAPI()
index = IdMapIndex(dim=8, bit_width=4)


def vector(value):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(400, "expected an 8-dimensional numeric vector")
    if array.shape != (8,) or not np.isfinite(array).all() or not np.any(array):
        raise HTTPException(400, "expected a nonzero, finite 8-dimensional vector")
    return array / np.linalg.norm(array)


def rows(request):
    data = request.get("data")
    if not isinstance(data, list) or any(
        not isinstance(row, list) or len(row) != 2
        or not isinstance(row[0], int) or not isinstance(row[1], dict)
        for row in data
    ):
        raise HTTPException(400, "expected data rows [row_number, payload]")
    if len(data) != 1:
        raise HTTPException(400, "demo accepts exactly one row per request")
    return data


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/sf/add")
def add(request: dict):
    result = []
    for row_number, payload in rows(request):
        ids = payload.get("ids")
        vectors = payload.get("vectors")
        if (not isinstance(ids, list) or not isinstance(vectors, list)
                or not ids or len(ids) != len(vectors)
                or any(type(value) is not int or value < 0 or value >= 2**64 for value in ids)
                or len(set(ids)) != len(ids)):
            raise HTTPException(400, "expected equal-length vectors and unique uint64 ids")
        batch = np.stack([vector(value) for value in vectors])
        index.add_with_ids(batch, np.asarray(ids, dtype=np.uint64))
        result.append([row_number, {"added": len(ids)}])
    return {"data": result}


@app.post("/sf/search")
def search(request: dict):
    result = []
    for row_number, payload in rows(request):
        k = payload.get("k", 5)
        if type(k) is not int or k < 1 or k > 100:
            raise HTTPException(400, "k must be between 1 and 100")
        query = vector(payload.get("query"))[None, :]
        scores, ids = index.search(query, k=k)
        result.append([row_number, {
            "ids": [int(value) for value in ids[0]],
            "scores": [float(value) for value in scores[0]],
        }])
    return {"data": result}
```

This adapter deliberately accepts exactly one row per request to avoid partial writes or inconsistent search results when Snowflake batches service-function calls. It is not hardened against oversized requests, concurrent writes, authorization failures, or automatic retry of side-effecting SQL calls. Use controlled test input only. Do not treat a successful filtered search as proof of tenant isolation.

## Build and push

Create the image repository in your chosen database/schema:

```sql
CREATE IMAGE REPOSITORY IF NOT EXISTS TURBOVEC_DEMO_REPO;
SHOW IMAGE REPOSITORIES;
```

Copy the `repository_url` from the `SHOW` output (lowercase hostname, with the database, schema, and repository path). Replace the placeholder below with **your own** value; do not paste credentials into shell history:

```bash
docker build --platform linux/amd64 -t turbovec-spcs-demo:local .
snow spcs image-registry login --connection YOUR_CONNECTION
# Example URL shape: org-account.registry.snowflakecomputing.com/db/schema/turbovec_demo_repo
docker tag turbovec-spcs-demo:local YOUR_REPOSITORY_URL/turbovec-spcs-demo:v1
docker push YOUR_REPOSITORY_URL/turbovec-spcs-demo:v1
```

For a local smoke test before pushing, run `docker run --rm -p 127.0.0.1:8000:8000 turbovec-spcs-demo:local` in another terminal, then:

```bash
curl -fsS http://localhost:8000/health
curl -fsS -X POST http://localhost:8000/sf/add -H 'Content-Type: application/json' -d '{"data":[[0,{"ids":[101,102],"vectors":[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]]}]]}'
curl -fsS -X POST http://localhost:8000/sf/search -H 'Content-Type: application/json' -d '{"data":[[0,{"query":[1,0,0,0,0,0,0,0],"k":2}]]}'
```

The second request should report `added: 2`; the third should rank ID `101` first. Local `curl` uses an intentionally unauthenticated test adapter; do **not** expose its port outside a trusted machine.

## Create the service and SQL functions

Substitute the image path after `image:` with `/DB/SCHEMA/TURBOVEC_DEMO_REPO/turbovec-spcs-demo:v1` (the image path, **not** the registry hostname). Change `CPU_X64_S` if your account offers a different suitable x86 instance family. Run only in your test database/schema:

```sql
CREATE COMPUTE POOL IF NOT EXISTS TURBOVEC_DEMO_POOL
  MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_S
  AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 300;

CREATE SERVICE TURBOVEC_DEMO_SERVICE
  IN COMPUTE POOL TURBOVEC_DEMO_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: turbovec
      image: /DB/SCHEMA/TURBOVEC_DEMO_REPO/turbovec-spcs-demo:v1
      readinessProbe:
        port: 8000
        path: /health
  endpoints:
    - name: api
      port: 8000
      public: false
$$
  MIN_INSTANCES = 1 MAX_INSTANCES = 1;

CREATE FUNCTION TURBOVEC_DEMO_ADD(payload VARIANT)
  RETURNS VARIANT
  SERVICE = TURBOVEC_DEMO_SERVICE
  ENDPOINT = api
  MAX_BATCH_ROWS = 1
  MAX_BATCH_RETRIES = 0
  AS '/sf/add';

CREATE FUNCTION TURBOVEC_DEMO_SEARCH(payload VARIANT)
  RETURNS VARIANT
  SERVICE = TURBOVEC_DEMO_SERVICE
  ENDPOINT = api
  MAX_BATCH_ROWS = 1
  AS '/sf/search';
```

Service functions marshal arguments into an HTTP POST batch and expect `data` rows with the same row numbers in the response. `MAX_BATCH_ROWS = 1` matches the adapter's single-row constraint, but it is not a guarantee of sequential calls or durable state. `MAX_BATCH_RETRIES = 0` on `ADD` prevents configured batch retries, but callers can still retry a query. `ADD` mutates in-memory state: call it only as a manually controlled demonstration, not inside a repeatable bulk query or a pipeline with automatic retries. A failed/retried call may add twice or reach a different instance after a restart. This example provides no exactly-once ingestion guarantee.

After `SELECT SYSTEM$GET_SERVICE_STATUS('TURBOVEC_DEMO_SERVICE')` shows the service ready, test:

```sql
SELECT TURBOVEC_DEMO_ADD(OBJECT_CONSTRUCT(
  'ids', ARRAY_CONSTRUCT(101, 102),
  'vectors', ARRAY_CONSTRUCT(
    ARRAY_CONSTRUCT(1,0,0,0,0,0,0,0),
    ARRAY_CONSTRUCT(0,1,0,0,0,0,0,0)
  )
));

SELECT TURBOVEC_DEMO_SEARCH(OBJECT_CONSTRUCT(
  'query', ARRAY_CONSTRUCT(1,0,0,0,0,0,0,0), 'k', 2
));
```

Expect `added: 2` and ID `101` first. If the second call returns no IDs, check service readiness/logs and whether it restarted between calls. Do not increase `MIN_INSTANCES` or `MAX_INSTANCES` beyond 1: each replica owns a separate in-memory index.

## Operational boundaries and cleanup

- Keep `public: false`. Grant service-function `USAGE` only to trusted roles; this example has no application-level authorization. In a real multi-tenant system, resolve tenant identity and allowed document IDs from a trusted source and enforce that mapping before `IdMapIndex.search(..., allowlist=...)`. Never accept an allowlist or tenant label as proof of access.
- Persist and restore the index before relying on it across service restarts. A stage volume is not a POSIX disk and does not support atomic renames, random writes, or appends; choose a storage strategy compatible with turbovec's snapshot semantics. Add failure/retry and replica tests before production use.
- Costs continue while compute runs. After testing, remove only the demo resources **you created** (and only after checking for dependents):

```sql
DROP FUNCTION IF EXISTS TURBOVEC_DEMO_SEARCH(VARIANT);
DROP FUNCTION IF EXISTS TURBOVEC_DEMO_ADD(VARIANT);
DROP SERVICE IF EXISTS TURBOVEC_DEMO_SERVICE;
DROP COMPUTE POOL IF EXISTS TURBOVEC_DEMO_POOL;
DROP IMAGE REPOSITORY IF EXISTS TURBOVEC_DEMO_REPO;
```

References: [SPCS service specifications](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/specification-reference), [service functions and service lifecycle](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-services), [image registry](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-registry-repository), [stage volume limitations](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/snowflake-stage-volume).
