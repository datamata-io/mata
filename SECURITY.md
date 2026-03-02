# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.9.x   | ✅        |
| < 1.9   | ❌        |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

To report a vulnerability, use one of the following channels:

- **Preferred**: [GitHub Security Advisories](https://github.com/datamata-io/mata/security/advisories/new) — report privately via GitHub's built-in mechanism
- **Alternative**: Email the maintainers at the address listed in `pyproject.toml`

### What to include

- A clear description of the vulnerability
- Steps to reproduce (proof-of-concept if possible)
- Potential impact assessment
- Suggested fix (optional but appreciated)

### Response timeline

- **Acknowledgement**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix / disclosure timeline**: coordinated with the reporter

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We ask that you give us reasonable time to address the issue before any public disclosure.

## Scope

The following are **in scope** for security reports:

- MATA framework source code (`src/mata/`)
- CI/CD workflows (`.github/workflows/`)
- Package configuration (`pyproject.toml`)

The following are **out of scope**:

- Third-party model weights or datasets used with MATA
- Vulnerabilities in upstream dependencies (report those to the respective projects)
- Issues that require physical access to the machine running MATA
- Social engineering attacks

## Security Best Practices for Users

- Always install MATA from the official repository (`datamata-io/mata`) or PyPI
- Pin dependency versions in production environments
- Do not load untrusted model weights (`.pt`, `.onnx`, `.pth`) — deserialization of arbitrary files can execute code
- Review any custom adapter code before running it in sensitive environments
