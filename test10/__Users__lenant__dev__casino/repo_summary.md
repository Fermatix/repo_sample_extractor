# Repository Summary

## 1. Overview

This is a web application for evaluating and managing conversations, rules, and attributes in a customer service / compliance context. The backend is written in Python (FastAPI) with SQLAlchemy/Pydantic for data modeling. The frontend is a React/TypeScript SPA (Vite). The project includes LLM-based evaluation pipelines (OpenAI/Gemini integrations), rule parsing and extraction, conversation analysis, and project management workflows.

**Stack:** Python, FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Docker, PostgreSQL  
**Build:** pyproject.toml (Python), package.json + Vite (frontend), Makefile, Docker Compose

## 2. Repository Structure

```
├── pyproject.toml          # Python project config
├── docker-compose.yml      # Main services
├── Makefile                # Dev tasks
├── migrations/             # Alembic DB migrations
├── src/                    # Python backend
│   ├── api/                # LLM callers (OpenAI, Gemini)
│   ├── config/             # App configuration
│   ├── constants.py        # Domain constants
│   ├── database/           # DB connection / sessions
│   ├── entities/           # Domain entity classes
│   ├── exceptions.py       # Custom exceptions
│   ├── mappers/            # Data mapping layer
│   ├── repositories/       # Data access layer
│   ├── schemas.py          # Pydantic models (~3K LOC)
│   ├── services/           # Business logic (~23K LOC)
│   ├── usecases/           # Use-case orchestration
│   ├── utils/              # Utility functions
│   ├── web/                # FastAPI routes
│   └── workers/            # Background workers
├── src/web/frontend/src/   # React/TypeScript frontend
│   ├── components/         # UI components
│   ├── pages/              # Page-level components
│   ├── services/           # API client code
│   ├── contexts/           # React contexts
│   └── hooks/              # Custom React hooks
├── tests/                  # Python tests (pytest)
├── pii-redactor/           # PII redaction service
├── rules/                  # Rule files (DSL-like)
└── ops/                    # Operations (Grafana, etc.)
```

## 3. Sampled Content

| Layer       | Files | LOC    | % of Sample |
|-------------|-------|--------|-------------|
| data        | 1     | 2944   | 57%         |
| util        | 3     | 1881   | 37%         |
| api         | 2     | 179    | 3%          |
| test        | 1     | 147    | 3%          |
| **Total**   | **7** | **5151** | **100%**  |

- **data/schemas.py** — the largest Pydantic schema file covering request/response models for conversations, alerts, projects, rules, attributes, and evaluations.
- **util/constants.py** — domain constants and enums for conversation statuses, alert types, evaluation scores.
- **util/dependencies.py** — FastAPI dependency injection setup for database sessions, auth, and services.
- **util/exceptions.py** — custom exception hierarchy.
- **api/llm_caller.py + gemini_caller.py** — abstraction layer for LLM providers.
- **test (e2e_python)** — an end-to-end test for a debugging workflow.

Tests are underrepresented (3%) — the repo has extensive tests (~91K LOC total) but most are very large files (4K–10K LOC each). Only a handful of files were sampled.

## 4. Notes

- The repository is a monolith with a clear layered architecture (repositories → services → API routes → usecases).
- Frontend code (TypeScript/React) was **not sampled** due to budget constraints. The frontend is roughly comparable in size.
- The largest files (services, repositories, tests) were not fully sampled because of their size — partial extracts would be needed for fair representation.
- Rule files (in `rules/`) and the PII redactor sub-project were also skipped.
- The codebase uses both synchronous and async patterns, with FastAPI async endpoints and some synchronous service methods.
- All identifiers are in non-ASCII Latin script; no recognizable company, product, or geographic names are present in the sampled files.
