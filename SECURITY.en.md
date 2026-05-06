<!-- translation: machine + human review pending -->

# SECURITY — Picarones (English)

> French version: [`SECURITY.md`](SECURITY.md) (canonical).
> Detailed threat model: [`docs/security/threat-model.md`](docs/security/threat-model.md).
>
> This is a summary translation focused on what an English-speaking
> auditor needs.  The canonical FR version remains authoritative
> for institutional sign-off.  Full alignment scheduled for a
> dedicated human-review sprint.

## Reporting a vulnerability

If you discover a security vulnerability in Picarones, please **do
not file a public GitHub issue**.  Instead, use one of the following
private channels:

- **GitHub Security Advisories** (preferred):
  https://github.com/maribakulj/Picarones/security/advisories/new
- **`/.well-known/security.txt`** on any institutional deployment
  (RFC 9116) — the contact address is documented there.

We acknowledge reports within **72 hours** and aim to ship a fix
within **30 days** for HIGH severity issues, **90 days** for MEDIUM.
A coordinated disclosure agreement is offered for non-trivial issues.

## Supported versions

| Version | Status | Security fixes |
|---------|--------|----------------|
| 1.x (current) | Active | Yes |
| 0.x | End of life | No — please upgrade |
| Pre-release branches | Best effort | On request |

## Deployment contexts

Picarones is designed for three deployment contexts:

1. **Developer machine** (Codespaces, laptop) — local access only,
   relaxed defaults to keep iteration fast.
2. **Institutional server** (intranet, scientific cluster) —
   authenticated internal access, with cost guards (rate limit, body
   size limit, max concurrent jobs).
3. **Public space** (HuggingFace Space, online demo) — anyone can
   reach the API; cloud API keys (OpenAI, Anthropic, Mistral, Azure…)
   must NOT be exposed to financial DoS.

## Security controls — quick reference

| Variable | Default | Effect |
|----------|---------|--------|
| `PICARONES_PUBLIC_MODE` | off | If `1`/`true`, refuses cloud OCR/LLM with shared keys and enables rate limit |
| `PICARONES_MAX_UPLOAD_MB` | `100` | Max upload size in MiB |
| `PICARONES_MAX_CONCURRENT_JOBS` | `2` | Max parallel benchmark jobs (in-process semaphore) |
| `PICARONES_RATE_LIMIT_PER_HOUR` | `5` (public mode) | Max jobs per IP per hour, `0` disables |
| `PICARONES_CSP` | hardened policy | Override Content-Security-Policy |
| `PICARONES_CSRF_REQUIRED` | off | If `1`/`true`, enables CSRF protection (double-submit cookie + HMAC) |
| `PICARONES_CSRF_SECRET` | auto | HMAC secret for CSRF tokens; **must be set in production** |

## In-process middlewares

The `picarones.interfaces.web.security` module provides four
middlewares that institutional operators wire via `create_app(...)`:

- `SecurityHeadersMiddleware` — adds CSP, X-Frame-Options,
  X-Content-Type-Options, Referrer-Policy, Permissions-Policy to
  every response.
- `BodySizeLimitMiddleware` — rejects requests where
  `Content-Length` exceeds a threshold.  **Known limitation**: does
  not catch `Transfer-Encoding: chunked`; nginx
  `client_max_body_size` is recommended in front.
- `RateLimitMiddleware` — sliding window, in-memory,
  `trust_proxy_count: int` for safe `X-Forwarded-For` parsing,
  LRU eviction on `max_clients=10000` to bound memory.
- `AuthenticationMiddleware` — opt-in wrapper around an
  `AuthenticationBackend` Protocol; the institution plugs in its
  SSO/LDAP/JWT scheme.

## Audit trail

Sensitive job mutations (`POST /api/jobs`, `DELETE /api/jobs/{id}`)
emit a structured `[audit]` log line at INFO level with the source
IP, ready to be ingested by the institution's SIEM.

## Reproducibility and integrity

`RunManifest` is byte-deterministic (`model_dump_json` with ordered
fields).  The SHA-256 hash of a manifest can be cited in a scientific
publication to anchor the run.  Cryptographic signing of manifests
(Sigstore) is on the roadmap.

## Cloud API key management

Cloud keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`,
`GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_DOC_INTEL_*`) are read from
environment variables only.  Adapters never log keys.  For
institutional deployments, source the env from a secrets vault
(HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, etc.) at
process startup.

See also [`docs/operations/runbook.md`](docs/operations/runbook.md)
for incident response and [`docs/legal/dpa-template.md`](docs/legal/dpa-template.md)
for the data processing agreement template covering cloud
sub-processors.

## Last revised

2026-05.  This document is reviewed at every major release.
