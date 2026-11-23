"""A small skills taxonomy used for keyword extraction from postings and resumes.

Each entry: canonical name -> (aliases, kind, suggestion). ``kind`` says whether a
gap is best closed by learning ("learn") or by shipping something ("build").
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    aliases: tuple[str, ...]
    kind: str = "learn"
    suggestion: str = ""


_S = Skill
SKILLS: tuple[Skill, ...] = (
    # languages
    _S("Python", ("python",), "build"),
    _S("JavaScript", ("javascript", "js", "es6"), "build"),
    _S("TypeScript", ("typescript",), "build"),
    _S("SQL", ("sql", "postgres", "postgresql", "mysql", "sql server", "t-sql"), "build"),
    _S("PowerShell", ("powershell",), "build"),
    _S("Bash", ("bash", "shell scripting"), "learn"),
    _S("Java", ("java",), "learn"),
    _S("C#/.NET", ("c#", ".net", "dotnet", "asp.net"), "learn"),
    _S("Go", ("golang",), "learn"),
    # web / integration
    _S("REST APIs", ("rest", "restful", "rest api", "rest apis", "apis", "api"), "build",
       "Build a small service that consumes and exposes a REST API, with tests and docs."),
    _S("GraphQL", ("graphql",), "learn"),
    _S("JSON", ("json",), "learn"),
    _S("XML", ("xml", "soap"), "learn"),
    _S("Webhooks", ("webhook", "webhooks"), "build",
       "Wire a webhook receiver (e.g. GitHub -> your service) with signature verification."),
    _S("OAuth/SSO", ("oauth", "oauth2", "oauth 2.0", "saml", "sso", "single sign-on", "openid connect", "oidc"),
       "learn", "Implement an OAuth 2.0 client-credentials and auth-code flow against a sandbox API."),
    _S("Postman", ("postman",), "learn"),
    _S("iPaaS", ("mulesoft", "boomi", "workato", "zapier", "ipaas", "make.com", "power automate",
                 "logic apps"), "learn",
       "Rebuild one integration in a free iPaaS tier (Workato/Zapier/Power Automate) and document it."),
    _S("ETL/Data pipelines", ("etl", "elt", "data pipeline", "data pipelines", "airflow"), "build"),
    _S("Message queues", ("kafka", "rabbitmq", "sqs", "pub/sub", "service bus", "message queue"), "learn"),
    _S("EDI/SFTP", ("edi", "sftp", "ftp"), "learn"),
    _S("HTTP", ("http", "https"), "learn"),
    _S("Node.js", ("node.js", "nodejs", "node"), "build"),
    _S("React", ("react", "react.js"), "build"),
    _S("FastAPI/Flask", ("fastapi", "flask", "django"), "build"),
    # cloud / infra
    _S("Azure", ("azure", "entra id", "azure ad"), "learn"),
    _S("AWS", ("aws", "amazon web services", "lambda", "s3", "ec2"), "learn"),
    _S("GCP", ("gcp", "google cloud"), "learn"),
    _S("Docker", ("docker", "containers", "containerization"), "build"),
    _S("Kubernetes", ("kubernetes", "k8s"), "learn"),
    _S("Terraform", ("terraform", "infrastructure as code", "iac"), "learn"),
    _S("CI/CD", ("ci/cd", "github actions", "gitlab ci", "jenkins", "azure devops"), "build"),
    _S("Git", ("git", "github", "gitlab"), "learn"),
    _S("Linux", ("linux", "unix", "ubuntu"), "learn"),
    _S("Windows Server", ("windows server",), "learn"),
    _S("Active Directory", ("active directory",), "learn"),
    _S("Networking", ("tcp/ip", "dns", "dhcp", "networking", "vpn", "firewall", "firewalls"), "learn"),
    _S("Microsoft 365", ("microsoft 365", "office 365", "m365", "o365", "exchange", "exchange online",
                         "sharepoint", "teams admin"), "learn"),
    _S("Intune/MDM", ("intune", "mdm", "airwatch", "workspace one", "jamf"), "learn"),
    _S("Monitoring/Logging", ("datadog", "splunk", "grafana", "prometheus", "new relic", "log analysis",
                              "observability", "elk"), "learn"),
    # support tooling / practice
    _S("Ticketing systems", ("servicenow", "zendesk", "jira service management", "freshdesk",
                             "ticketing", "jira"), "learn"),
    _S("Salesforce", ("salesforce",), "learn"),
    _S("Troubleshooting", ("troubleshooting", "troubleshoot", "root cause", "debugging", "debug"), "learn"),
    _S("Customer-facing communication", ("customer-facing", "customer facing", "client-facing",
                                         "communication skills", "written communication"), "learn"),
    _S("ITIL", ("itil",), "learn"),
    # AI / ML
    _S("LLMs", ("llm", "llms", "large language model", "large language models", "generative ai", "genai",
                "gpt"), "build"),
    _S("RAG", ("rag", "retrieval-augmented", "retrieval augmented"), "build",
       "Ship a RAG app with an evaluation set and a written retrieval-quality report."),
    _S("Prompt engineering", ("prompt engineering", "prompting", "prompt design"), "learn"),
    _S("LLM APIs", ("openai api", "openai", "anthropic", "claude api", "azure openai", "bedrock"), "build"),
    _S("Agents/tool calling", ("agentic", "ai agents", "llm agents", "tool calling", "function calling",
                               "mcp", "model context protocol"), "build"),
    _S("LLM evaluation", ("evals", "llm evaluation", "model evaluation", "eval harness"), "build",
       "Add an eval harness (labeled set + precision/recall report) to one of your AI projects."),
    _S("LangChain/LlamaIndex", ("langchain", "llamaindex", "langgraph"), "learn"),
    _S("Vector databases", ("vector database", "vector databases", "vector db", "pinecone", "weaviate",
                            "chroma", "pgvector", "faiss", "qdrant", "embeddings"), "build"),
    _S("Machine learning", ("machine learning", "ml", "scikit-learn", "sklearn"), "learn"),
    _S("Deep learning", ("pytorch", "tensorflow", "keras", "deep learning"), "learn"),
    _S("NLP", ("nlp", "natural language processing"), "learn"),
    _S("Hugging Face", ("hugging face", "huggingface", "transformers"), "learn"),
    _S("Fine-tuning", ("fine-tuning", "fine tuning", "lora"), "build"),
    _S("MLOps", ("mlops", "mlflow", "model deployment", "model serving"), "learn"),
)

_SPECIAL = {"c#": r"(?<![\w])c#(?![\w])", ".net": r"(?<![\w])\.net(?![\w])"}


def _alias_pattern(alias: str) -> str:
    if alias in _SPECIAL:
        return _SPECIAL[alias]
    escaped = re.escape(alias).replace(r"\ ", r"[\s-]+")
    return rf"(?<![\w./-]){escaped}(?![\w-])"


_COMPILED: tuple[tuple[Skill, re.Pattern[str]], ...] = tuple(
    (s, re.compile("|".join(_alias_pattern(a) for a in s.aliases), re.I)) for s in SKILLS
)
SKILL_BY_NAME = {s.name: s for s in SKILLS}


def extract_skills(text: str) -> list[str]:
    """Canonical skill names mentioned in ``text``, in taxonomy order."""
    return [skill.name for skill, pattern in _COMPILED if pattern.search(text or "")]


def suggestion_for(name: str) -> tuple[str, str]:
    skill = SKILL_BY_NAME.get(name)
    if skill is None:
        return "learn", f"Learn the basics of {name}."
    if skill.suggestion:
        return skill.kind, skill.suggestion
    if skill.kind == "build":
        return "build", f"Build a small public project that uses {name} and mention it on your resume."
    return "learn", f"Learn {name} fundamentals (docs + a short hands-on lab) so you can speak to it."
