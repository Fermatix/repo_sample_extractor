import os
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, TypedDict

from dotenv import load_dotenv

load_dotenv()

DEFAULT_AGENT_ROLE = "support agent"
DEFAULT_CUSTOMER_ROLE = "customer"
DEFAULT_PROJECT_DESCRIPTION = "No project description provided."
TIMELINESAI_CONTEXT_LOOKBACK_MAX_DAYS = 365
TIMELINESAI_CONTEXT_LOOKBACK_MAX_MINUTES = TIMELINESAI_CONTEXT_LOOKBACK_MAX_DAYS * 24 * 60
# Temporary rollout boundary for Telegram no-agent fallback evaluation.
TELEGRAM_NO_AGENT_RESPONSE_FALLBACK_ENABLED_FROM = datetime(2026, 3, 24, 19, 30, 18)
TELEGRAM_ENTITY_RESOLUTION_LEGACY_SAFE_ERROR = (
    "Telegram could not access this chat after several attempts. Retry this chat from the one-time evaluation run."
)
TELEGRAM_ENTITY_RESOLUTION_SAFE_ERROR = (
    "Telegram could not access this chat after several attempts. Use Retry failed to try this chat again, "
    "or start a new one-time evaluation run for this chat."
)


def is_telegram_no_agent_response_fallback_enabled(created_at: datetime | None) -> bool:
    if created_at is None:
        return False
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    return created_at >= TELEGRAM_NO_AGENT_RESPONSE_FALLBACK_ENABLED_FROM


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name}: {raw}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw}")


MANAGER_DASHBOARD_FACTS_ENABLED = _bool_env("MANAGER_DASHBOARD_FACTS_ENABLED", False)
MANAGER_DASHBOARD_FACTS_WRITE_ENABLED = _bool_env(
    "MANAGER_DASHBOARD_FACTS_WRITE_ENABLED",
    False,
)
if MANAGER_DASHBOARD_FACTS_ENABLED and not MANAGER_DASHBOARD_FACTS_WRITE_ENABLED:
    raise ValueError(
        "MANAGER_DASHBOARD_FACTS_ENABLED=true requires MANAGER_DASHBOARD_FACTS_WRITE_ENABLED=true "
        "so dashboard reads cannot use stale fact rows."
    )

CONVERSATION_MANAGER_ATTRIBUTIONS_VERSION = _int_env("CONVERSATION_MANAGER_ATTRIBUTIONS_VERSION", 1)
CONVERSATION_MANAGER_ATTRIBUTIONS_WRITE_ENABLED = _bool_env(
    "CONVERSATION_MANAGER_ATTRIBUTIONS_WRITE_ENABLED",
    True,
)
CONVERSATION_MANAGER_ATTRIBUTIONS_READ_ENABLED = _bool_env(
    "CONVERSATION_MANAGER_ATTRIBUTIONS_READ_ENABLED",
    False,
)
if CONVERSATION_MANAGER_ATTRIBUTIONS_READ_ENABLED and not CONVERSATION_MANAGER_ATTRIBUTIONS_WRITE_ENABLED:
    raise ValueError(
        "CONVERSATION_MANAGER_ATTRIBUTIONS_READ_ENABLED=true requires "
        "CONVERSATION_MANAGER_ATTRIBUTIONS_WRITE_ENABLED=true so conversation reads cannot use stale attribution rows."
    )


class ConversationManagerAttributionSource(str, Enum):
    REVIEW_OVERRIDE = "review_override"
    IN_WINDOW_MESSAGE = "in_window_message"
    TELEGRAM_FALLBACK = "telegram_fallback"
    TIMELINESAI_FALLBACK = "timelinesai_fallback"


class EvaluationMethod(str, Enum):
    ONE_BY_ONE = "one_by_one"
    ALL_AT_ONCE = "all_at_once"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid evaluation method."""
        return value in cls.list()


class EvaluationResultStatus(str, Enum):
    EVALUATED = "evaluated"
    NO_MESSAGES = "no_messages"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CriteriaScoringMode(str, Enum):
    ADDITION = "addition"
    SUBTRACTION = "subtraction"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class AdditionScoreDirection(str, Enum):
    """Display-only direction for addition-mode scores: whether a higher score is good or bad."""
    MORE_IS_GOOD = "more_is_good"
    MORE_IS_BAD = "more_is_bad"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CriterionContextLookbackMode(str, Enum):
    DEFAULT = "default"
    CUSTOM = "custom"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()

class BenchmarkExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()

class UserRole(str, Enum):
    ADMIN = "admin"
    QC_MANAGER = "qc_manager"
    AGENT = "agent"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all user roles."""
        return {
            cls.ADMIN.value: "Admin",
            cls.QC_MANAGER.value: "QC Manager",
            cls.AGENT.value: "Agent",
        }
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of roles with their display names."""
        display_names = cls.display_names()
        return [{"id": role_id, "name": display_names[role_id]} for role_id in cls.list()]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid user role."""
        return value in cls.list()


class AgentConversationScope(str, Enum):
    OWN_ONLY = "own_only"
    ALL_PROJECT_CONVERSATIONS = "all_project_conversations"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class QcManagerManagerScope(str, Enum):
    ALL_PROJECT_MANAGERS = "all_project_managers"
    SELECTED_PROJECT_MANAGERS = "selected_project_managers"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class ProjectCapability(str, Enum):
    DASHBOARD_READ = "dashboard.read"
    DASHBOARD_WRITE = "dashboard.write"
    # Custom dashboards are a multi-manager configuration surface; agents only ever see
    # their own row, so they intentionally do not get this capability even though they
    # have DASHBOARD_READ for the standard dashboard.
    CUSTOM_DASHBOARDS_READ = "custom_dashboards.read"
    CUSTOM_DASHBOARDS_WRITE = "custom_dashboards.write"
    ALERTS_READ = "alerts.read"
    ALERTS_WRITE = "alerts.write"
    ATTRIBUTES_READ = "attributes.read"
    ATTRIBUTES_WRITE = "attributes.write"
    EVALUATION_TAGS_READ = "evaluation_tags.read"
    EVALUATION_TAGS_WRITE = "evaluation_tags.write"
    PROJECT_MEMBERS_READ = "project_members.read"
    APPEALS_READ = "appeals.read"
    APPEALS_ANALYTICS_READ = "appeals.analytics.read"
    APPEALS_COMMENT = "appeals.comment"
    APPEALS_OPEN = "appeals.open"
    APPEALS_MANAGE = "appeals.manage"
    PROJECT_SETTINGS_READ = "project_settings.read"
    PROJECT_SETTINGS_WRITE = "project_settings.write"
    CRITERIA_READ = "criteria.read"
    CRITERIA_WRITE = "criteria.write"
    CRITERIA_METADATA_READ = "criteria.metadata.read"
    KNOWLEDGE_BASE_READ = "knowledge_base.read"
    KNOWLEDGE_BASE_WRITE = "knowledge_base.write"
    FLOWS_READ = "flows.read"
    FLOWS_WRITE = "flows.write"
    PLAYGROUND_READ = "playground.read"
    PLAYGROUND_WRITE = "playground.write"
    BENCHMARK_READ = "benchmark.read"
    BENCHMARK_WRITE = "benchmark.write"
    ONE_OFF_EVALUATIONS_READ = "one_off_evaluations.read"
    ONE_OFF_EVALUATIONS_WRITE = "one_off_evaluations.write"
    TELEGRAM_METADATA_READ = "telegram.metadata.read"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


def normalize_project_capability_key(value: str | ProjectCapability) -> str:
    if isinstance(value, ProjectCapability):
        return value.value
    return str(value).strip()


class CompanyCapability(str, Enum):
    COMPANY_MEMBERS_READ = "company.members.read"
    COMPANY_MEMBERS_WRITE = "company.members.write"
    COMPANY_INVITATIONS_READ = "company.invitations.read"
    COMPANY_INVITATIONS_WRITE = "company.invitations.write"
    COMPANY_SETTINGS_READ = "company.settings.read"
    COMPANY_SETTINGS_WRITE = "company.settings.write"
    COMPANY_BILLING_READ = "company.billing.read"
    PROJECTS_CREATE = "projects.create"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


def normalize_company_capability_key(value: str | CompanyCapability) -> str:
    if isinstance(value, CompanyCapability):
        return value.value
    return str(value).strip()


class EvaluationAppealStatus(str, Enum):
    ONGOING = "ongoing"
    CLOSED_BY_AGENT = "closed_by_agent"
    CLOSED_BY_QC = "closed_by_qc"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]


class EvaluationAppealDecision(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]


class EvaluationAppealMistakeAttribution(str, Enum):
    QC = "qc"
    APPEAL_OPENER = "appeal_opener"
    NOT_COUNTED = "not_counted"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]


class EvaluationScoreEventType(str, Enum):
    REVIEW_SCORE_CHANGED = "review_score_changed"
    AGENT_PROPOSED_SCORE_CHANGED = "agent_proposed_score_changed"
    APPEAL_FINAL_SCORE_CHANGED = "appeal_final_score_changed"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

class InvitationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all invitation statuses."""
        return {
            cls.PENDING.value: "Pending",
            cls.ACCEPTED.value: "Accepted",
            cls.EXPIRED.value: "Expired"
        }
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of statuses with their display names."""
        display_names = cls.display_names()
        return [{"id": status_id, "name": display_names[status_id]} for status_id in cls.list()]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid invitation status."""
        return value in cls.list()

class CompanyCreationStep(str, Enum):
    CREATED = "created"
    INTEGRATION_CONNECTED = "integration_connected"
    TEMPLATE_SELECTED = "template_selected"
    COMPLETE = "complete"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all company creation steps."""
        return {
            cls.CREATED.value: "Created",
            cls.INTEGRATION_CONNECTED.value: "Integration Connected",
            cls.TEMPLATE_SELECTED.value: "Template Selected",
            cls.COMPLETE.value: "Complete"
        }
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of steps with their display names."""
        display_names = cls.display_names()
        return [{"id": step_id, "name": display_names[step_id]} for step_id in cls.list()]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid company creation step."""
        return value in cls.list()

class SignupStep(str, Enum):
    EMAIL = "email"
    PASSWORD = "password"
    COMPANY = "company"
    INTEGRATION = "integration"
    TEMPLATE = "template"
    COMPLETE = "complete"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all signup steps."""
        return {
            cls.EMAIL.value: "Email Setup",
            cls.PASSWORD.value: "Password Setup",
            cls.COMPANY.value: "Company Setup",
            cls.INTEGRATION.value: "Integration Setup",
            cls.TEMPLATE.value: "Template Selection",
            cls.COMPLETE.value: "Complete"
        }
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of steps with their display names."""
        display_names = cls.display_names()
        return [{"id": step_id, "name": display_names[step_id]} for step_id in cls.list()]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid signup step."""
        return value in cls.list()


class GoogleAuthMode(str, Enum):
    SIGNIN = "signin"
    SIGNUP = "signup"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()

class LlmMetadata(TypedDict):
    display: str
    active: bool


class LlmType(str, Enum):
    GPT_5_CHAT = "gpt-5-chat-latest"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_2 = "gpt-5.2"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GEMINI_3_PRO = "gemini-3-pro-preview"
    
    @classmethod
    def _metadata(cls) -> Dict["LlmType", LlmMetadata]:
        """Centralize LLM metadata including display name and active flag."""
        return {
            # Active model order controls the default selection in shared UI pickers.
            cls.GPT_5_2: {"display": "GPT-5.2", "active": True},
            cls.GPT_5_CHAT: {"display": "GPT-5 Instant", "active": True},
            cls.GPT_5_MINI: {"display": "GPT-5 Mini", "active": True},
            cls.GEMINI_2_5_FLASH: {"display": "Gemini 2.5 Flash", "active": False},
            cls.GEMINI_2_5_PRO: {"display": "Gemini 2.5 Pro", "active": False},
            cls.GEMINI_3_FLASH: {"display": "Gemini 3 Flash", "active": True},
            cls.GEMINI_3_PRO: {"display": "Gemini 3 Pro", "active": True},
        }
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible values."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all LLM models (active and inactive)."""
        return {llm.value: meta["display"] for llm, meta in cls._metadata().items()}
    
    @classmethod
    def active_display_names(cls) -> Dict[str, str]:
        """Get display names for active LLM models."""
        return {
            llm.value: meta["display"]
            for llm, meta in cls._metadata().items()
            if meta["active"]
        }
    
    @classmethod
    def active_list(cls) -> List[str]:
        """Get list of active LLM values."""
        return list(cls.active_display_names().keys())
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of models with their display names."""
        display_names = cls.display_names()
        return [{"id": model_id, "name": display_names[model_id]} for model_id in cls.list()]
    
    @classmethod
    def active_list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of active models with their display names."""
        display_names = cls.active_display_names()
        return [{"id": model_id, "name": display_names[model_id]} for model_id in display_names]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid model type."""
        return value in cls.list() if value else True  # Allow None values


class ReasoningEffort(str, Enum):
    """
    Controls the depth of reasoning for LLM calls.
    
    OpenAI: Maps to reasoning.effort parameter
    Gemini 2.5: Maps to thinking_config.thinking_budget (0=minimal, higher=more thinking)
    Gemini 3: Maps to thinking_config.thinking_level (1=minimal, 2=medium, 3=high)
    """
    MINIMAL = "minimal"  # Fastest, least reasoning
    LOW = "low"
    MEDIUM = "medium"    # Default
    HIGH = "high"        # Most thorough reasoning
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()
    
    def to_gemini_thinking_budget(self) -> int:
        """Convert to Gemini 2.5 thinking_budget value."""
        mapping = {
            ReasoningEffort.MINIMAL: 128,
            ReasoningEffort.LOW: 1024,
            ReasoningEffort.MEDIUM: 8192,
            ReasoningEffort.HIGH: 24576,
        }
        return mapping[self]
    
    def to_gemini_thinking_level(self) -> str:
        """Convert to Gemini 3 ThinkingLevel enum value (LOW/HIGH)."""
        # Gemini 3 only has LOW and HIGH levels
        mapping = {
            ReasoningEffort.MINIMAL: "LOW",
            ReasoningEffort.LOW: "LOW",
            ReasoningEffort.MEDIUM: "HIGH",  # Map medium to high since there's no medium
            ReasoningEffort.HIGH: "HIGH",
        }
        return mapping[self] 

class Language(str, Enum):
    AM = "Amharic"
    AR = "Arabic"
    AZ = "Azerbaijani"
    BN = "Bengali"
    BG = "Bulgarian"
    ZH = "Chinese"
    HR = "Croatian"
    CS = "Czech"
    DA = "Danish"
    NL = "Dutch"
    EN = "English"
    ET = "Estonian"
    FIL = "Filipino"
    FI = "Finnish"
    FR = "French"
    KA = "Georgian"
    DE = "German"
    EL = "Greek"
    HE = "Hebrew"
    HI = "Hindi"
    HU = "Hungarian"
    ID = "Indonesian"
    IT = "Italian"
    JA = "Japanese"
    KK = "Kazakh"
    KO = "Korean"
    LV = "Latvian"
    LT = "Lithuanian"
    MS = "Malay"
    NO = "Norwegian"
    FA = "Persian"
    PL = "Polish"
    PT = "Portuguese"
    RO = "Romanian"
    RU = "Russian"
    SR = "Serbian"
    SK = "Slovak"
    SL = "Slovenian"
    ES = "Spanish"
    SW = "Swahili"
    SV = "Swedish"
    TA = "Tamil"
    TH = "Thai"
    TR = "Turkish"
    UK = "Ukrainian"
    UR = "Urdu"
    VI = "Vietnamese"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def display_names(cls) -> Dict[str, str]:
        # Values are already full names; keep identity mapping
        return {lang: lang for lang in cls.list()}

    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        names = cls.display_names()
        return [{"id": code, "name": names.get(code, code)} for code in cls.list()]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()

class FlowStepType(str, Enum):
    START = "start"
    CLASSIFICATION = "classification"
    ACTION_GUIDANCE = "action_guidance"
    VALIDATION = "validation"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible step types."""
        return [e.value for e in cls]
    
    @classmethod
    def display_names(cls) -> Dict[str, str]:
        """Get display names for all step types."""
        return {
            cls.START.value: "Start",
            cls.CLASSIFICATION.value: "Classification",
            cls.ACTION_GUIDANCE.value: "Action Guidance", 
            cls.VALIDATION.value: "Validation"
        }
    
    @classmethod
    def list_with_display_names(cls) -> List[Dict[str, str]]:
        """Get list of step types with their display names."""
        display_names = cls.display_names()
        return [{"id": step_type, "name": display_names[step_type]} for step_type in cls.list()]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid step type."""
        return value in cls.list()

class JobTaskType(str, Enum):
    EVALUATE_CONVERSATION = "evaluate_conversation"
    SYNC_INTERCOM = "sync_intercom"
    SYNC_ZENDESK = "sync_zendesk"
    SYNC_TELEGRAM = "sync_telegram"
    SYNC_TIMELINESAI = "sync_timelinesai"
    SYNC_HELPSHIFT = "sync_helpshift"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all possible job task types."""
        return [e.value for e in cls]
    
    @classmethod
    def evaluation_task_types(cls) -> List[str]:
        """Get list of evaluation-related task types."""
        return [cls.EVALUATE_CONVERSATION.value]
    
    @classmethod
    def sync_types(cls) -> List[str]:
        """Get list of sync-related task types."""
        return [
            cls.SYNC_INTERCOM.value,
            cls.SYNC_ZENDESK.value,
            cls.SYNC_TELEGRAM.value,
            cls.SYNC_TIMELINESAI.value,
            cls.SYNC_HELPSHIFT.value,
        ]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        """Check if value is a valid job task type."""
        return value in cls.list() 

class EvaluationComponent(str, Enum):
    RULES = "rules"
    CRITERIA = "criteria"
    ATTRIBUTES = "attributes"
    TOPIC_COMMENT = "topic_comment"
    RULES_CRITERIA_TOPIC = "rules_criteria_topic"
    FLOW = "flow"
    CRITERION_AUTO_IMPROVE = "criterion_auto_improve"
    GENERAL_COMMENT_REGENERATION = "general_comment_regeneration"
    ATTRIBUTE_REAPPLY = "attribute_reapply"
    MANAGER_PERIOD_NOTE_GENERATION = "manager_period_note_generation"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class EvaluationBillingReason(str, Enum):
    CONVERSATION = "conversation"
    PLAYGROUND = "playground"
    BENCHMARK = "benchmark"
    CRITERION_AUTO_IMPROVE = "criterion_auto_improve"
    CRITERION_TEST_PREVIEW = "criterion_test_preview"
    GENERAL_COMMENT_REGENERATION = "general_comment_regeneration"
    ATTRIBUTE_REAPPLY = "attribute_reapply"
    MANAGER_PERIOD_NOTE_GENERATION = "manager_period_note_generation"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class AttributeType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TIME = "time"
    DAY_TIME = "day_time"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class AttributeCategoryStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class AttributeResultSelectionKind(str, Enum):
    CURRENT = "current"
    BACKUP = "backup"
    LLM = "llm"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class AlertSeverity(str, Enum):
    URGENT = "urgent"
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class ProjectAlertTargetKind(str, Enum):
    ATTRIBUTE_CATEGORICAL = "attribute_categorical"
    ATTRIBUTE_NUMERICAL = "attribute_numerical"
    ATTRIBUTE_TIME = "attribute_time"
    ATTRIBUTE_DAY_TIME = "attribute_day_time"
    CRITERION = "criterion"
    TOTAL_SCORE = "total_score"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class ProjectAlertOperator(str, Enum):
    EQUALS = "equals"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    GREATER_THAN_OR_EQUAL = "gte"
    GREATER_THAN = "gt"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class ProjectAlertActivationBehavior(str, Enum):
    IMMEDIATE = "immediate"
    ON_APPROVE = "on_approve"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class ConversationAlertStatus(str, Enum):
    ACTIVE = "active"
    PENDING_APPROVAL = "pending_approval"
    READ = "read"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CustomDashboardColumnKind(str, Enum):
    PREDEFINED = "predefined"
    CRITERION = "criterion"
    ATTRIBUTE_NUMERICAL = "attribute_numerical"
    ATTRIBUTE_TIME = "attribute_time"
    ATTRIBUTE_DAY_TIME = "attribute_day_time"
    ATTRIBUTE_CATEGORICAL = "attribute_categorical"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CustomDashboardPredefinedMetric(str, Enum):
    AVG_START_TIME = "avg_start_time"
    AVG_FINISH_TIME = "avg_finish_time"
    CHATS_COUNT = "chats_count"
    MESSAGES_COUNT = "messages_count"
    AVG_TOTAL_SCORE = "avg_total_score"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CustomDashboardNumericalAggregation(str, Enum):
    SUM = "sum"
    AVG = "avg"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


class CustomDashboardCategoricalAggregation(str, Enum):
    COUNT = "count"
    PERCENTAGE = "percentage"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.list()


# Keep criterion-target billing reasons centralized so billing aggregation and
# presentation stay aligned as new billed criterion actions are added.
CRITERION_BILLING_REASONS = (
    EvaluationBillingReason.CRITERION_AUTO_IMPROVE.value,
    EvaluationBillingReason.CRITERION_TEST_PREVIEW.value,
)

# Billing reasons that should remain visible as separate rows instead of being
# collapsed into the underlying conversation/playground/benchmark target.
RUN_ROW_BILLING_REASONS = (
    *CRITERION_BILLING_REASONS,
    EvaluationBillingReason.GENERAL_COMMENT_REGENERATION.value,
    EvaluationBillingReason.ATTRIBUTE_REAPPLY.value,
    EvaluationBillingReason.MANAGER_PERIOD_NOTE_GENERATION.value,
)

# Maximum number of historical tickets/conversations to fetch per project (before project creation)
MAX_HISTORICAL_TICKETS_PER_PROJECT = 50

# ---------------------------------------------------------------------------
# Helpshift integration configuration
# ---------------------------------------------------------------------------
# Helpshift REST API base; the account domain is appended as /v1/{domain}/...
HELPSHIFT_API_BASE_URL = "https://api.helpshift.com/v1"
# Issue states considered "resolved/closed" and therefore eligible for evaluation.
# Helpshift surfaces the state under issue["state_data"]["state"].
HELPSHIFT_RESOLVED_STATES = frozenset({"resolved", "rejected"})
# Message ``type`` values that identify an agent's private/internal note (agent-to-agent, not
# visible to the end user). Normalized to lowercase with underscores -> spaces before matching.
# Kept as a set because the exact Helpshift wording is best-effort without dev access.
HELPSHIFT_INTERNAL_NOTE_TYPES = frozenset({"private note", "internal note", "note"})
# Namespace prefix for Helpshift agent/author external ids. Helpshift agents share manager
# source_type="admin" with Intercom; prefixing their source_id keeps the two from colliding on
# the unique (source_id, project_id) key while Helpshift's own discovery/refresh paths reconcile.
HELPSHIFT_AGENT_SOURCE_ID_PREFIX = "helpshift:"

# ---------------------------------------------------------------------------
# Billing / Quota configuration
# ---------------------------------------------------------------------------
# Initial number of free evaluations granted to a new company
FREE_EVALUATIONS_INITIAL = 200
# Paying companies can continue evaluating only while their current balance
# stays strictly above this floor.
PAYING_COMPANY_MIN_BALANCE_USD = -200.0

# ---------------------------------------------------------------------------
# Background job configuration
# ---------------------------------------------------------------------------

# Queue adder intervals
EVALUATION_QUEUE_ADDER_POLL_INTERVAL_SECONDS = _int_env("EVALUATION_QUEUE_ADDER_POLL_INTERVAL_SECONDS", 30)
DATA_SYNC_QUEUE_ADDER_POLL_INTERVAL_SECONDS = _int_env("DATA_SYNC_QUEUE_ADDER_POLL_INTERVAL_SECONDS", 30)
# Minimum age before a project is considered due for CRM sync (in seconds)
DATA_SYNC_CRM_CALL_INTERVAL_SECONDS = _int_env("DATA_SYNC_CRM_CALL_INTERVAL_SECONDS", 30)

# Processor intervals
EVAL_POLL_INTERVAL_SECONDS = _int_env("EVAL_POLL_INTERVAL_SECONDS", 10)
DATA_SYNC_POLL_INTERVAL_SECONDS = _int_env("DATA_SYNC_POLL_INTERVAL_SECONDS", 10)

# Maximum number of projects that will be synced in parallel
MAX_PARALLEL_PROJECT_SYNCS = _int_env("MAX_PARALLEL_PROJECT_SYNCS", 2)

# Maximum number of concurrent evaluation coroutines
MAX_PARALLEL_EVALUATIONS = _int_env("MAX_PARALLEL_EVALUATIONS", 6)

# Job claim lease duration to protect from duplicate processing if worker session commits mid-processing
JOB_CLAIM_LEASE_SECONDS = _int_env("JOB_CLAIM_LEASE_SECONDS", 900)

# AnyIO worker thread pool size for synchronous `def` route handlers.
# Keep aligned with API_DB_POOL_SIZE + API_DB_MAX_OVERFLOW plus headroom for non-DB routes.
THREAD_POOL_SIZE = _int_env("THREAD_POOL_SIZE", 20)
if THREAD_POOL_SIZE < 1:
    raise ValueError(f"THREAD_POOL_SIZE must be at least 1; got {THREAD_POOL_SIZE}") 

# Default executor size for the standalone background worker process. This controls
# asyncio.to_thread calls used by scheduler DB polling and other sync helpers.
WORKER_THREAD_POOL_SIZE = _int_env("WORKER_THREAD_POOL_SIZE", 20)
if WORKER_THREAD_POOL_SIZE < 1:
    raise ValueError(f"WORKER_THREAD_POOL_SIZE must be at least 1; got {WORKER_THREAD_POOL_SIZE}")
