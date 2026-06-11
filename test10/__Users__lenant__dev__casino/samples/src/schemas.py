import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

import pytz
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, EmailStr, Field, ValidationError, field_serializer, field_validator, validator

from src.constants import (
    DEFAULT_AGENT_ROLE,
    DEFAULT_CUSTOMER_ROLE,
    HELPSHIFT_AGENT_SOURCE_ID_PREFIX,
    HELPSHIFT_INTERNAL_NOTE_TYPES,
    TIMELINESAI_CONTEXT_LOOKBACK_MAX_DAYS,
    TIMELINESAI_CONTEXT_LOOKBACK_MAX_MINUTES,
    AdditionScoreDirection,
    AgentConversationScope,
    AlertSeverity,
    AttributeType,
    CriteriaScoringMode,
    CriterionContextLookbackMode,
    CustomDashboardCategoricalAggregation,
    CustomDashboardColumnKind,
    CustomDashboardNumericalAggregation,
    CustomDashboardPredefinedMetric,
    EvaluationAppealDecision,
    EvaluationAppealMistakeAttribution,
    EvaluationMethod,
    Language,
    LlmType,
    ProjectAlertOperator,
    ProjectAlertActivationBehavior,
    ProjectAlertTargetKind,
    QcManagerManagerScope,
    UserRole,
)
from src.entities.entities import (
    EvaluationGeneralResult,
    FlowEvaluationReport,
    PlaygroundEvaluationUsageSummary,
    PlaygroundMessage,
)
from src.mappers.benchmark_mapper import format_benchmark_execution_for_api
from src.services.token_service import TokenService

T = TypeVar('T')


def _normalize_company_role_input(value: str) -> str:
    normalized_value = str(value or "").strip()
    if normalized_value == "member":
        return UserRole.QC_MANAGER.value
    return normalized_value

# Auth models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    created_at: datetime
    company_id: Optional[str] = None
    role: Optional[str] = None
    signup_step: Optional[str] = None
    agent_project_accesses: List["AgentProjectAccessResponse"] = Field(default_factory=list)
    qc_manager_project_accesses: List["QcManagerProjectAccessResponse"] = Field(default_factory=list)

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# Signup models
class SignupEmailRequest(BaseModel):
    email: EmailStr

class SignupWithInvitationRequest(BaseModel):
    invitation_token: str

class SignupTokenRequest(BaseModel):
    signup_token: str

class SignupPasswordRequest(BaseModel):
    signup_token: str
    password: str
    full_name: str

class SignupCompleteWithInvitationRequest(BaseModel):
    signup_token: str
    invitation_token: str

class SignupStatusResponse(BaseModel):
    email: str
    full_name: str
    signup_step: int
    email_verified: bool
    company_id: Optional[str] = None
    has_company: bool


class GoogleAuthUrlResponse(BaseModel):
    auth_url: str


class GoogleAuthCallbackRequest(BaseModel):
    code: str
    state: str


class GoogleAuthCallbackResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    signup_step: str
    signup_token: Optional[str] = None
    invitation_token: Optional[str] = None
    redirect_path: Optional[str] = None

# Company models
class CompanyCreate(BaseModel):
    name: str
    expected_agents_count: Optional[str] = None
    
    @validator('expected_agents_count', pre=True)
    def convert_expected_agents_count(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            # Handle string values like "6-10", "100+"
            return v
        return str(v)

class CompanySignupCreate(BaseModel):
    signup_token: str
    name: str
    expected_agents_count: Optional[str] = None
    
    @validator('expected_agents_count', pre=True)
    def convert_expected_agents_count(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v
        return str(v)

class CompanyResponse(BaseModel):
    id: str
    name: str
    expected_agents_count: Optional[str]
    creation_step: str
    free_evaluations_remaining: int
    is_paying: bool
    created_at: datetime
    updated_at: datetime
    current_user_company_access: Optional["CompanyAccessResponse"] = None

    class Config:
        from_attributes = True

class CompanyInvitationCreate(BaseModel):
    email: EmailStr
    role: str
    agent_project_accesses: List["AgentProjectAccessRequest"] = Field(default_factory=list)
    qc_manager_project_accesses: List["QcManagerProjectAccessRequest"] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = _normalize_company_role_input(value)
        if not UserRole.contains(value):
            raise ValueError(f'Invalid role. Must be one of: {", ".join(UserRole.list())}')
        return value


class AgentProjectAccessRequest(BaseModel):
    project_id: str
    mapped_manager_id: str
    conversation_scope: str

    @field_validator("conversation_scope")
    @classmethod
    def validate_conversation_scope(cls, value: str) -> str:
        if not AgentConversationScope.contains(value):
            raise ValueError(
                f'Invalid conversation_scope. Must be one of: {", ".join(AgentConversationScope.list())}'
            )
        return value


class AgentProjectAccessResponse(BaseModel):
    id: str
    project_id: str
    mapped_manager_id: str
    conversation_scope: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QcManagerProjectAccessRequest(BaseModel):
    project_id: str
    manager_scope: str
    manager_accesses: List["QcManagerProjectManagerAccessRequest"] = Field(default_factory=list)
    can_access_appeals_read: bool = False
    can_access_appeals_comment: bool = False
    can_access_appeals_manage: bool = False
    can_access_project_settings: bool = False
    can_access_criteria: bool = False
    can_access_knowledge_base: bool = False
    can_access_flows: bool = False
    can_access_playground: bool = False
    can_access_benchmark: bool = False
    can_access_attributes: bool = False
    can_access_evaluation_tags: bool = False
    can_access_alerts: bool = False
    can_access_one_off_evaluations: bool = False

    @field_validator("manager_scope")
    @classmethod
    def validate_manager_scope(cls, value: str) -> str:
        if not QcManagerManagerScope.contains(value):
            raise ValueError(
                f'Invalid manager_scope. Must be one of: {", ".join(QcManagerManagerScope.list())}'
            )
        return value


class QcManagerProjectManagerAccessRequest(BaseModel):
    manager_id: str


class QcManagerProjectManagerAccessResponse(BaseModel):
    id: str
    manager_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QcManagerProjectAccessResponse(BaseModel):
    id: str
    project_id: str
    manager_scope: str
    manager_accesses: List[QcManagerProjectManagerAccessResponse] = Field(default_factory=list)
    can_access_appeals_read: bool = False
    can_access_appeals_comment: bool = False
    can_access_appeals_manage: bool = False
    can_access_project_settings: bool = False
    can_access_criteria: bool = False
    can_access_knowledge_base: bool = False
    can_access_flows: bool = False
    can_access_playground: bool = False
    can_access_benchmark: bool = False
    can_access_attributes: bool = False
    can_access_evaluation_tags: bool = False
    can_access_alerts: bool = False
    can_access_one_off_evaluations: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EvaluationTagCreate(BaseModel):
    name: str
    is_active: bool = True
    assignable_by_agents: bool = False
    assignable_by_qc_managers: bool = True


class EvaluationTagUpdate(BaseModel):
    name: Optional[str] = None
    is_active: Optional[bool] = None
    assignable_by_agents: Optional[bool] = None
    assignable_by_qc_managers: Optional[bool] = None


class EvaluationTagResponse(BaseModel):
    id: str
    project_id: str
    name: str
    is_active: bool
    assignable_by_agents: bool
    assignable_by_qc_managers: bool
    order: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EvaluationTagAvailableResponse(BaseModel):
    # Active tags surfaced to the evaluation modal / filters. `can_assign` reflects whether the
    # requesting user's role may add/remove this tag on a conversation.
    id: str
    name: str
    can_assign: bool


class ConversationEvaluationTagAssignmentResponse(BaseModel):
    id: str
    evaluation_tag_id: str
    name: str
    assigned_by_user_id: Optional[str] = None
    assigned_by_name: Optional[str] = None
    assigned_at: datetime
    # Whether the requesting user's role may remove this assignment. Derived from the tag
    # definition, so a tag deactivated after assignment stays removable by permitted roles.
    can_assign: bool = True


class ConversationEvaluationTagsUpdateRequest(BaseModel):
    # Full desired set of assigned tag ids for the conversation (idempotent set semantics).
    evaluation_tag_ids: List[str] = Field(default_factory=list)


class ProjectAccessResponse(BaseModel):
    capabilities: Dict[str, bool]


class CompanyAccessResponse(BaseModel):
    capabilities: Dict[str, bool]


QcManagerProjectAccessRequest.model_rebuild()
UserResponse.model_rebuild()

class CompanyInvitationResponse(BaseModel):
    id: str
    company_id: str
    email: str
    invited_by_user_id: str
    status: str
    role: str
    created_at: datetime
    expires_at: datetime
    agent_project_accesses: List[AgentProjectAccessResponse] = Field(default_factory=list)
    qc_manager_project_accesses: List[QcManagerProjectAccessResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True

class InvitationDetailsResponse(BaseModel):
    company_name: str
    role: str
    email: str
    expires_at: datetime
    status: str
    can_create_account_from_invite: bool
    block_reason: Optional[str] = None

class AcceptInvitationRequest(BaseModel):
    token: str

class OnboardingStatusResponse(BaseModel):
    company_id: str
    current_step: str
    steps: Dict[str, Any]
    company_name: str
    expected_agents_count: Optional[str]

class UserRoleUpdate(BaseModel):
    role: str
    agent_project_accesses: List[AgentProjectAccessRequest] = Field(default_factory=list)
    qc_manager_project_accesses: List[QcManagerProjectAccessRequest] = Field(default_factory=list)
    
    @validator('role')
    def validate_role(cls, v):
        v = _normalize_company_role_input(v)
        if not UserRole.contains(v):
            raise ValueError(f'Invalid role. Must be one of: {", ".join(UserRole.list())}')
        return v

class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    expected_agents_count: Optional[str] = None
    
    @validator('expected_agents_count', pre=True)
    def convert_expected_agents_count(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            # Handle string values like "6-10", "100+"
            return v
        return str(v)


CompanyInvitationCreate.model_rebuild()
UserRoleUpdate.model_rebuild()

# Project Template models
class ProjectTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    icon: Optional[str]
    category: Optional[str]
    is_active: bool
    created_at: datetime
    criteria_names: List[str]
    knowledge_base_files_count: Optional[int] = 0

    class Config:
        from_attributes = True

class ProjectFromTemplateRequest(BaseModel):
    signup_token: Optional[str] = None
    template_id: str
    project_name: Optional[str] = None
    existing_project_id: Optional[str] = None  # For updating existing projects with template data

# Intercom integration models
class IntercomConnectRequest(BaseModel):
    signup_token: str
    token: str

class IntercomPreviewResponse(BaseModel):
    project_id: str
    detected_agents: List[str]
    agent_count: int
    sample_conversations: int
    success: bool
    message: str

# Zendesk integration models
class ZendeskConnectRequest(BaseModel):
    signup_token: str
    subdomain: str
    email: str
    api_token: str

class ZendeskPreviewResponse(BaseModel):
    project_id: str
    detected_agents: List[str]
    agent_count: int
    sample_tickets: int
    success: bool
    message: str


# Telegram integration models
class TelegramValidateRequest(BaseModel):
    """Request to validate Telegram session credentials."""
    session_string: str


class TelegramAuthStartRequest(BaseModel):
    """Request to start Telegram auth by phone."""
    name: Optional[str] = None
    phone_number: str
    config_id: Optional[str] = None


class TelegramAuthVerifyCodeRequest(BaseModel):
    """Request to verify Telegram login code."""
    auth_token: str
    code: str


class TelegramAuthVerifyPasswordRequest(BaseModel):
    """Request to verify Telegram 2FA password."""
    auth_token: str
    password: str


class TelegramConfigCreate(BaseModel):
    """Request to create a new Telegram integration config."""
    name: str  # Name to identify this Telegram user
    session_string: str


class TelegramConfigUpdate(BaseModel):
    """Request to update Telegram integration config."""
    name: Optional[str] = None
    session_string: Optional[str] = None


class TelegramConfigResponse(BaseModel):
    """Response for Telegram integration config."""
    id: str
    name: str
    session_string_masked: Optional[str] = None  # Masked session string for display
    is_connected: bool = False

    class Config:
        from_attributes = True


class TelegramAuthStartResponse(BaseModel):
    """Response after Telegram auth start."""
    auth_token: str
    status: str = "code_sent"


class TelegramAuthVerifyCodeResponse(BaseModel):
    """Response after Telegram code verification."""
    status: str  # connected | password_required
    auth_token: Optional[str] = None
    config: Optional[TelegramConfigResponse] = None


class TelegramAuthVerifyPasswordResponse(BaseModel):
    """Response after Telegram password verification."""
    status: str = "connected"
    config: TelegramConfigResponse


class TelegramChatResponse(BaseModel):
    """Response for a Telegram chat."""
    id: str
    project_id: str
    telegram_integration_id: str
    chat_id: str
    telegram_peer_id: str
    forum_topic_id: Optional[int] = None
    forum_topic_title: Optional[str] = None
    chat_title: str
    chat_type: str
    tags: List[str] = []
    agent_telegram_user_ids: List[str] = []
    reviewed_agent_telegram_user_ids: Optional[List[str]] = None
    is_enabled: bool
    sync_interval_minutes: Optional[int] = None
    sync_anchor_at: Optional[datetime] = None
    next_sync_at: Optional[datetime] = None
    fetch_window_minutes: Optional[int] = None
    last_sync_at: Optional[datetime] = None
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TelegramChatUpdate(BaseModel):
    """Request to update a Telegram chat settings."""
    is_enabled: Optional[bool] = None
    sync_interval_minutes: Optional[int] = None
    sync_anchor_at: Optional[datetime] = None
    fetch_window_minutes: Optional[int] = None
    tags: Optional[List[str]] = None
    reviewed_agent_telegram_user_ids: Optional[List[str]] = None


class TelegramChatsBulkUpdateByTagsRequest(BaseModel):
    """Request to update Telegram chats matched by tags."""
    tags: Optional[List[str]] = None
    agent_telegram_user_ids: Optional[List[str]] = None
    is_enabled: Optional[bool] = None
    sync_interval_minutes: Optional[int] = None
    sync_anchor_at: Optional[datetime] = None
    fetch_window_minutes: Optional[int] = None
    sync_now: Optional[bool] = None


class TelegramChatBulkExactUpdateItem(BaseModel):
    """Request item to update one Telegram chat by database ID."""
    chat_id: str
    is_enabled: Optional[bool] = None
    sync_interval_minutes: Optional[int] = None
    sync_anchor_at: Optional[datetime] = None
    fetch_window_minutes: Optional[int] = None
    tags: Optional[List[str]] = None
    reviewed_agent_telegram_user_ids: Optional[List[str]] = None


class TelegramChatsBulkUpdateRequest(BaseModel):
    """Request to update multiple exact Telegram chats."""
    updates: List[TelegramChatBulkExactUpdateItem]


class TelegramChatAgentFilterStatusResponse(BaseModel):
    """Response describing whether cached chat participant agent data is stale."""
    is_stale: bool
    participants_refreshed_at: Optional[datetime] = None


class TelegramChatRefreshStatusResponse(BaseModel):
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    discovered_chat_count: int = 0
    eligible_chat_count: int = 0
    processed_eligible_chat_count: int = 0


class TelegramChatRefreshStartResponse(TelegramChatRefreshStatusResponse):
    already_running: bool = False


class TelegramAgentCreate(BaseModel):
    """Request to create a Telegram agent."""
    telegram_user_id: str
    telegram_username: Optional[str] = None
    display_name: str
    tags: Optional[List[str]] = None
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None

    @validator("evaluation_language")
    def validate_evaluation_language(cls, v):
        if v is not None and not Language.contains(v):
            raise ValueError(f'Invalid language value. Must be one of: {", ".join(Language.list())}')
        return v


class TelegramAgentUpdate(BaseModel):
    """Request to update a Telegram agent."""
    telegram_user_id: Optional[str] = None
    telegram_username: Optional[str] = None
    display_name: Optional[str] = None
    tags: Optional[List[str]] = None
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None

    @validator("evaluation_language")
    def validate_evaluation_language(cls, v):
        if v is not None and not Language.contains(v):
            raise ValueError(f'Invalid language value. Must be one of: {", ".join(Language.list())}')
        return v


class TelegramAgentResponse(BaseModel):
    """Response for a Telegram agent."""
    id: str
    project_id: str
    telegram_user_id: str
    telegram_username: Optional[str] = None
    display_name: str
    tags: List[str] = []
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TimelinesAIConnectRequest(BaseModel):
    token: str


class TimelinesAISettingsUpdate(BaseModel):
    auto_import_resolved_chats: bool
    context_lookback_minutes: Optional[int] = None

    @validator("context_lookback_minutes")
    def validate_context_lookback_minutes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("context_lookback_minutes must be > 0")
        if v is not None and v > TIMELINESAI_CONTEXT_LOOKBACK_MAX_MINUTES:
            raise ValueError(
                f"context_lookback_minutes cannot exceed {TIMELINESAI_CONTEXT_LOOKBACK_MAX_DAYS} days"
            )
        return v


class TimelinesAIConnectionResponse(BaseModel):
    integration_status: str
    token_masked: Optional[str] = None
    auto_import_resolved_chats: bool = False
    context_lookback_minutes: Optional[int] = None
    last_sync_at: Optional[datetime] = None
    metadata_refreshed_at: Optional[datetime] = None
    is_connected: bool = False


class TimelinesAIWhatsAppAccountResponse(BaseModel):
    id: str
    external_account_id: str
    account_name: str
    phone: Optional[str] = None
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    status: Optional[str] = None
    connected_on: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class TimelinesAIAgentResponse(BaseModel):
    id: str
    project_id: str
    source_id: str
    display_name: str
    email: Optional[str] = None
    tags: List[str] = []
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TimelinesAIAgentUpdate(BaseModel):
    tags: Optional[List[str]] = None
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None


class TimelinesAIChatResponse(BaseModel):
    id: str
    project_id: str
    external_chat_id: str
    external_whatsapp_account_id: Optional[str] = None
    name: str
    phone: Optional[str] = None
    jid: Optional[str] = None
    chat_url: Optional[str] = None
    closed: bool
    is_group: bool
    is_allowed_to_message: Optional[bool] = None
    read: Optional[bool] = None
    unattended: Optional[bool] = None
    responsible_name: Optional[str] = None
    responsible_email: Optional[str] = None
    last_message_uid: Optional[str] = None
    last_message_timestamp: Optional[datetime] = None
    created_timestamp: Optional[datetime] = None
    platform_labels: List[str] = []
    local_tags: List[str] = []
    excluded_platform_labels: List[str] = []
    reviewed_manager_ids: Optional[List[str]] = None
    effective_tags: List[str] = []
    last_imported_message_uid: Optional[str] = None
    last_imported_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class TimelinesAIChatUpdate(BaseModel):
    local_tags: Optional[List[str]] = None
    excluded_platform_labels: Optional[List[str]] = None
    reviewed_manager_ids: Optional[List[str]] = None


class TimelinesAIChatBulkUpdateItem(BaseModel):
    chat_id: str
    local_tags: Optional[List[str]] = None
    excluded_platform_labels: Optional[List[str]] = None
    reviewed_manager_ids: Optional[List[str]] = None


class TimelinesAIChatBulkUpdateRequest(BaseModel):
    updates: List[TimelinesAIChatBulkUpdateItem]


class TimelinesAIStatusResponse(BaseModel):
    connection: TimelinesAIConnectionResponse
    accounts: List[TimelinesAIWhatsAppAccountResponse] = []
    agents: List[TimelinesAIAgentResponse] = []
    chats: List[TimelinesAIChatResponse] = []


class TimelinesAIMetadataAgentResponse(BaseModel):
    id: str
    project_id: str
    display_name: str
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime


class TimelinesAIMetadataChatResponse(BaseModel):
    id: str
    project_id: str
    external_chat_id: str
    name: str
    closed: bool
    effective_tags: List[str] = []
    reviewed_manager_ids: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime


class TimelinesAIMetadataResponse(BaseModel):
    agents: List[TimelinesAIMetadataAgentResponse] = []
    chats: List[TimelinesAIMetadataChatResponse] = []


# Project models
class GoogleSheetsIntegrationResponse(BaseModel):
    is_enabled: bool = False
    share_email: Optional[str] = None
    spreadsheet_id: Optional[str] = None

    class Config:
        from_attributes = True


class GoogleSheetsIntegrationUpdate(BaseModel):
    is_enabled: Optional[bool] = None
    share_email: Optional[str] = None
    spreadsheet_id: Optional[str] = None


def _normalize_optional_non_empty_text(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    criteria_rules_eval_type: str = EvaluationMethod.ONE_BY_ONE.value
    criteria_scoring_mode: str = CriteriaScoringMode.ADDITION.value
    addition_score_direction: str = AdditionScoreDirection.MORE_IS_GOOD.value
    subtraction_max_score: int = 100
    subtraction_min_score: Optional[int] = None
    llm_id: str
    language: str
    agent_role: str = DEFAULT_AGENT_ROLE
    customer_role: str = DEFAULT_CUSTOMER_ROLE
    pii_redaction_basic_enabled: bool = True
    pii_redaction_author_names_enabled: bool = True
    pii_redaction_names_enabled: bool = False
    default_alert_activation_behavior: str = ProjectAlertActivationBehavior.IMMEDIATE.value
    telegram_context_lookback_minutes: Optional[int] = None
    intercom_access_token: Optional[str] = None
    intercom_channel_types: Optional[List[str]] = None
    intercom_email_domains: Optional[List[str]] = None
    zendesk_subdomain: Optional[str] = None
    zendesk_email: Optional[str] = None
    zendesk_api_token: Optional[str] = None
    zendesk_channel_types: Optional[List[str]] = None
    zendesk_email_domains: Optional[List[str]] = None

    @validator('criteria_rules_eval_type')
    def validate_criteria_rules_eval_type(cls, v):
        if not EvaluationMethod.contains(v):
            raise ValueError(f'Invalid criteria_rules_eval_type value. Must be one of: {", ".join(EvaluationMethod.list())}')
        return v

    @validator("criteria_scoring_mode")
    def validate_criteria_scoring_mode(cls, v):
        if not CriteriaScoringMode.contains(v):
            raise ValueError(
                f'Invalid criteria_scoring_mode value. Must be one of: {", ".join(CriteriaScoringMode.list())}'
            )
        return v

    @validator("addition_score_direction")
    def validate_addition_score_direction(cls, v):
        if not AdditionScoreDirection.contains(v):
            raise ValueError(
                f'Invalid addition_score_direction value. Must be one of: {", ".join(AdditionScoreDirection.list())}'
            )
        return v

    @validator('llm_id')
    def validate_llm_id(cls, v):
        if not LlmType.contains(v):
            raise ValueError(f'Invalid llm_id value. Must be one of: {", ".join(LlmType.list())}')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        if not Language.contains(v):
            raise ValueError(f'Invalid language value. Must be one of: {", ".join(Language.list())}')
        return v

    @validator("agent_role")
    def validate_agent_role(cls, v):
        normalized = _normalize_optional_non_empty_text(v, "agent_role")
        if normalized is None:
            raise ValueError("agent_role must not be empty")
        return normalized

    @validator("customer_role")
    def validate_customer_role(cls, v):
        normalized = _normalize_optional_non_empty_text(v, "customer_role")
        if normalized is None:
            raise ValueError("customer_role must not be empty")
        return normalized

    @validator("telegram_context_lookback_minutes")
    def validate_telegram_context_lookback_minutes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("telegram_context_lookback_minutes must be > 0")
        return v

    @validator("default_alert_activation_behavior")
    def validate_default_alert_activation_behavior(cls, v):
        if not ProjectAlertActivationBehavior.contains(v):
            raise ValueError(
                "Invalid default_alert_activation_behavior value. Must be one of: "
                f"{', '.join(ProjectAlertActivationBehavior.list())}"
            )
        return v

    @validator("subtraction_max_score")
    def validate_subtraction_max_score(cls, v):
        if v <= 0:
            raise ValueError("subtraction_max_score must be > 0")
        return v

    @validator("subtraction_min_score")
    def validate_subtraction_min_score(cls, v, values):
        max_score = values.get("subtraction_max_score", 100)
        if v is not None and v > max_score:
            raise ValueError("subtraction_min_score must be <= subtraction_max_score")
        return v

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    agent_role: str = DEFAULT_AGENT_ROLE
    customer_role: str = DEFAULT_CUSTOMER_ROLE
    owner_id: str
    criteria_rules_eval_type: str
    criteria_scoring_mode: str = CriteriaScoringMode.ADDITION.value
    addition_score_direction: str = AdditionScoreDirection.MORE_IS_GOOD.value
    subtraction_max_score: int = 100
    subtraction_min_score: Optional[int] = None
    llm_id: str
    language: str
    pii_redaction_basic_enabled: bool = True
    pii_redaction_author_names_enabled: bool = True
    pii_redaction_names_enabled: bool = True
    default_alert_activation_behavior: str = ProjectAlertActivationBehavior.IMMEDIATE.value
    telegram_context_lookback_minutes: Optional[int] = None
    timelinesai_context_lookback_minutes: Optional[int] = None
    intercom_access_token: Optional[str] = None
    intercom_channel_types: Optional[List[str]] = None
    intercom_email_domains: Optional[List[str]] = None
    zendesk_subdomain: Optional[str] = None
    zendesk_email: Optional[str] = None
    zendesk_api_token: Optional[str] = None
    zendesk_channel_types: Optional[List[str]] = None
    zendesk_email_domains: Optional[List[str]] = None
    telegram_configs: List[TelegramConfigResponse] = []  # Multiple Telegram users per project
    timelinesai_integration_status: Optional[str] = None
    helpshift_integration_status: Optional[str] = None
    google_sheets_integration: Optional[GoogleSheetsIntegrationResponse] = None
    current_user_access: Optional[ProjectAccessResponse] = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle database object and entity conversion with custom logic."""
        # Check if it's a ProjectEntity (has evaluation_config attribute)
        if hasattr(obj, 'evaluation_config'):
            # It's a ProjectEntity - extract data from nested config objects
            token_service = TokenService()
            
            # Extract evaluation config
            eval_config = obj.evaluation_config
            
            # Extract Intercom config
            intercom_config = obj.intercom_config
            intercom_access_token = None
            intercom_channel_types = None
            intercom_email_domains = None
            if intercom_config:
                # Mask the token
                if intercom_config.access_token:
                    try:
                        decrypted_token = token_service.decrypt_token(intercom_config.access_token)
                        if decrypted_token:
                            intercom_access_token = token_service.mask_token(decrypted_token)
                    except ValueError:
                        intercom_access_token = None
                intercom_channel_types = intercom_config.channel_types
                intercom_email_domains = intercom_config.email_domains
            
            # Extract Zendesk config
            zendesk_config = obj.zendesk_config
            zendesk_subdomain = None
            zendesk_email = None
            zendesk_api_token = None
            zendesk_channel_types = None
            zendesk_email_domains = None
            if zendesk_config:
                zendesk_subdomain = zendesk_config.subdomain
                zendesk_email = zendesk_config.email
                # Mask the token
                if zendesk_config.api_token:
                    try:
                        decrypted_token = token_service.decrypt_token(zendesk_config.api_token)
                        if decrypted_token:
                            zendesk_api_token = token_service.mask_token(decrypted_token)
                    except ValueError:
                        zendesk_api_token = None
                zendesk_channel_types = zendesk_config.channel_types
                zendesk_email_domains = zendesk_config.email_domains
            
            # Extract Telegram configs (multiple users)
            telegram_configs_response = []
            for telegram_config in (obj.telegram_configs or []):
                session_string_masked = None
                if telegram_config.session_string:
                    try:
                        decrypted_session = token_service.decrypt_token(telegram_config.session_string)
                        if decrypted_session:
                            session_string_masked = token_service.mask_token(decrypted_session)
                    except ValueError:
                        session_string_masked = None
                telegram_configs_response.append(TelegramConfigResponse(
                    id=telegram_config.id,
                    name=telegram_config.name,
                    session_string_masked=session_string_masked,
                    is_connected=bool(
                        telegram_config.session_string
                        and telegram_config.api_id
                        and telegram_config.api_hash
                    ),
                ))

            google_sheets_integration_response: Optional[GoogleSheetsIntegrationResponse] = None
            if obj.google_sheets_integration is not None:
                google_sheets_integration_response = GoogleSheetsIntegrationResponse(
                    is_enabled=obj.google_sheets_integration.is_enabled,
                    share_email=obj.google_sheets_integration.share_email,
                    spreadsheet_id=obj.google_sheets_integration.spreadsheet_id,
                )
            
            data = {
                'id': obj.id,
                'name': obj.name,
                'description': obj.description,
                'agent_role': getattr(obj, "agent_role", DEFAULT_AGENT_ROLE),
                'customer_role': getattr(obj, "customer_role", DEFAULT_CUSTOMER_ROLE),
                'owner_id': obj.owner_id,
                'criteria_rules_eval_type': eval_config.criteria_rules_eval_type.value,
                'criteria_scoring_mode': eval_config.criteria_scoring_mode.value,
                'addition_score_direction': eval_config.addition_score_direction.value,
                'subtraction_max_score': eval_config.subtraction_max_score,
                'subtraction_min_score': eval_config.subtraction_min_score,
                'llm_id': eval_config.llm_id,
                'language': eval_config.language,
                'pii_redaction_basic_enabled': obj.pii_redaction_basic_enabled,
                'pii_redaction_author_names_enabled': obj.pii_redaction_author_names_enabled,
                'pii_redaction_names_enabled': obj.pii_redaction_names_enabled,
                'default_alert_activation_behavior': obj.default_alert_activation_behavior,
                'telegram_context_lookback_minutes': obj.telegram_context_lookback_minutes,
                'timelinesai_context_lookback_minutes': getattr(obj, "timelinesai_context_lookback_minutes", None),
                'intercom_access_token': intercom_access_token,
                'intercom_channel_types': intercom_channel_types,
                'intercom_email_domains': intercom_email_domains,
                'zendesk_subdomain': zendesk_subdomain,
                'zendesk_email': zendesk_email,
                'zendesk_api_token': zendesk_api_token,
                'zendesk_channel_types': zendesk_channel_types,
                'zendesk_email_domains': zendesk_email_domains,
                'telegram_configs': telegram_configs_response,
                'timelinesai_integration_status': getattr(obj, "timelinesai_integration_status", None),
                'helpshift_integration_status': getattr(obj, "helpshift_integration_status", None),
                'google_sheets_integration': google_sheets_integration_response,
                'created_at': obj.created_at,
                'updated_at': obj.updated_at,
            }
            return super().model_validate(data, *args, **kwargs)
        
        if hasattr(obj, '__table__'):  # It's an SQLAlchemy model
            # Create a dict from the model
            data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
            
            # Process the Intercom access token - mask it if present
            if data.get('intercom_access_token'):
                token_service = TokenService()
                try:
                    decrypted_token = token_service.decrypt_token(data['intercom_access_token'])
                    if decrypted_token:
                        # Mask the token using the token service
                        data['intercom_access_token'] = token_service.mask_token(decrypted_token)
                except ValueError:
                    # If decryption fails, set to None to avoid exposing the encrypted token
                    data['intercom_access_token'] = None
            
            # Process the Zendesk API token - mask it if present
            if data.get('zendesk_api_token'):
                token_service = TokenService()
                try:
                    decrypted_token = token_service.decrypt_token(data['zendesk_api_token'])
                    if decrypted_token:
                        # Mask the token using the token service
                        data['zendesk_api_token'] = token_service.mask_token(decrypted_token)
                except ValueError:
                    # If decryption fails, set to None to avoid exposing the encrypted token
                    data['zendesk_api_token'] = None
            
            # Parse JSON strings to Python lists before validation
            if data.get('intercom_channel_types'):
                try:
                    if isinstance(data['intercom_channel_types'], str):
                        data['intercom_channel_types'] = json.loads(data['intercom_channel_types'])
                except (json.JSONDecodeError, TypeError):
                    data['intercom_channel_types'] = []
            
            if data.get('intercom_email_domains'):
                try:
                    if isinstance(data['intercom_email_domains'], str):
                        data['intercom_email_domains'] = json.loads(data['intercom_email_domains'])
                except (json.JSONDecodeError, TypeError):
                    data['intercom_email_domains'] = []
            
            # Parse Zendesk JSON strings to Python lists before validation
            if data.get('zendesk_channel_types'):
                try:
                    if isinstance(data['zendesk_channel_types'], str):
                        data['zendesk_channel_types'] = json.loads(data['zendesk_channel_types'])
                except (json.JSONDecodeError, TypeError):
                    data['zendesk_channel_types'] = []
            
            if data.get('zendesk_email_domains'):
                try:
                    if isinstance(data['zendesk_email_domains'], str):
                        data['zendesk_email_domains'] = json.loads(data['zendesk_email_domains'])
                except (json.JSONDecodeError, TypeError):
                    data['zendesk_email_domains'] = []

            google_sheets_integration = getattr(obj, "google_sheets_integration", None)
            if google_sheets_integration is not None:
                data["google_sheets_integration"] = GoogleSheetsIntegrationResponse(
                    is_enabled=google_sheets_integration.is_enabled,
                    share_email=google_sheets_integration.share_email,
                    spreadsheet_id=google_sheets_integration.spreadsheet_id,
                )
            else:
                data["google_sheets_integration"] = None

            timelinesai_integration = getattr(obj, "timelinesai_integration", None)
            data["timelinesai_integration_status"] = (
                timelinesai_integration.integration_status
                if timelinesai_integration is not None
                else None
            )

            helpshift_integration = getattr(obj, "helpshift_integration", None)
            data["helpshift_integration_status"] = (
                helpshift_integration.integration_status
                if helpshift_integration is not None
                else None
            )

            # Use the processed data for validation
            return super().model_validate(data, *args, **kwargs)
        
        # If it's not an SQLAlchemy model or entity, use standard validation
        return super().model_validate(obj, *args, **kwargs)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agent_role: Optional[str] = None
    customer_role: Optional[str] = None
    criteria_rules_eval_type: Optional[str] = None
    criteria_scoring_mode: Optional[str] = None
    addition_score_direction: Optional[str] = None
    subtraction_max_score: Optional[int] = None
    subtraction_min_score: Optional[int] = None
    llm_id: Optional[str] = None
    language: Optional[str] = None
    pii_redaction_basic_enabled: Optional[bool] = None
    pii_redaction_author_names_enabled: Optional[bool] = None
    pii_redaction_names_enabled: Optional[bool] = None
    default_alert_activation_behavior: Optional[str] = None
    telegram_context_lookback_minutes: Optional[int] = None
    intercom_access_token: Optional[str] = None
    intercom_channel_types: Optional[List[str]] = None
    intercom_email_domains: Optional[List[str]] = None
    zendesk_subdomain: Optional[str] = None
    zendesk_email: Optional[str] = None
    zendesk_api_token: Optional[str] = None
    zendesk_channel_types: Optional[List[str]] = None
    zendesk_email_domains: Optional[List[str]] = None
    google_sheets_integration: Optional[GoogleSheetsIntegrationUpdate] = None
    
    @validator('criteria_rules_eval_type')
    def validate_criteria_rules_eval_type(cls, v):
        if v is not None and not EvaluationMethod.contains(v):
            raise ValueError(f'Invalid criteria_rules_eval_type value. Must be one of: {", ".join(EvaluationMethod.list())}')
        return v

    @validator("criteria_scoring_mode")
    def validate_criteria_scoring_mode(cls, v):
        if v is not None and not CriteriaScoringMode.contains(v):
            raise ValueError(
                f'Invalid criteria_scoring_mode value. Must be one of: {", ".join(CriteriaScoringMode.list())}'
            )
        return v

    @validator("addition_score_direction")
    def validate_addition_score_direction(cls, v):
        if v is not None and not AdditionScoreDirection.contains(v):
            raise ValueError(
                f'Invalid addition_score_direction value. Must be one of: {", ".join(AdditionScoreDirection.list())}'
            )
        return v

    @validator('llm_id')
    def validate_llm_id(cls, v):
        if v is not None and not LlmType.contains(v):
            raise ValueError(f'Invalid llm_id value. Must be one of: {", ".join(LlmType.list())}')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        if v is not None and not Language.contains(v):
            raise ValueError(f'Invalid language value. Must be one of: {", ".join(Language.list())}')
        return v

    @validator("agent_role")
    def validate_agent_role(cls, v):
        return _normalize_optional_non_empty_text(v, "agent_role")

    @validator("customer_role")
    def validate_customer_role(cls, v):
        return _normalize_optional_non_empty_text(v, "customer_role")

    @validator("telegram_context_lookback_minutes")
    def validate_telegram_context_lookback_minutes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("telegram_context_lookback_minutes must be > 0")
        return v

    @validator("default_alert_activation_behavior")
    def validate_update_default_alert_activation_behavior(cls, v):
        if v is not None and not ProjectAlertActivationBehavior.contains(v):
            raise ValueError(
                "Invalid default_alert_activation_behavior value. Must be one of: "
                f"{', '.join(ProjectAlertActivationBehavior.list())}"
            )
        return v

    @validator("subtraction_max_score")
    def validate_subtraction_max_score(cls, v):
        if v is not None and v <= 0:
            raise ValueError("subtraction_max_score must be > 0")
        return v

# Conversation models
class ConversationMessage(BaseModel):
    message_id: str
    created_at: str
    content: str
    author_type: Optional[str] = None
    author_name: Optional[str] = None
    is_internal_note: bool = False


class ConversationProjectResponse(BaseModel):
    project_id: str


class ConversationListAlert(BaseModel):
    id: str
    alert_name: str
    severity: str


class ConversationGroup(BaseModel):
    conversation_id: str
    manager_name: str
    manager_tags: List[str] = Field(default_factory=list)
    is_finished: bool
    created_at: str
    chat_title: Optional[str] = None
    chat_tags: List[str] = Field(default_factory=list)
    evaluation_tags: List[str] = Field(default_factory=list)
    evaluation_result: Optional[Dict[str, Any]]
    evaluation_approved: bool = False
    reviewed_at: Optional[datetime] = None
    appeal_status: Optional[str] = None
    active_alert_count: int = 0
    active_alert_highest_severity: Optional[str] = None
    active_alerts: List[ConversationListAlert] = Field(default_factory=list)
    active_alert_hidden_names: List[str] = Field(default_factory=list)
    active_alert_hidden_highest_severity: Optional[str] = None
    total_score: float
    max_score: float
    message_count: int


class ConversationExportFormat(str, Enum):
    CSV = "csv"
    GOOGLE_SHEETS = "google_sheets"


class ConversationExportRequest(BaseModel):
    export_format: ConversationExportFormat
    manager: Optional[str] = None
    manager_id: Optional[str] = None
    evaluation_status: str = "evaluated"
    approval_status: str = "all"
    appeal_status: str = "all"
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    criterion_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    chat_id: Optional[str] = None
    chat_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None
    evaluation_tags: Optional[List[str]] = None
    attribute_id: Optional[str] = None
    attribute_min_value: Optional[float] = None
    attribute_max_value: Optional[float] = None
    attribute_category_id: Optional[str] = None
    one_off_evaluation_run_batch_id: Optional[str] = None
    share_email: Optional[EmailStr] = None

    @validator("evaluation_status")
    def validate_evaluation_status(cls, value):
        if value not in {"evaluated", "not_evaluated", "all"}:
            raise ValueError("Invalid evaluation_status. Use evaluated | not_evaluated | all")
        return value

    @validator("approval_status")
    def validate_approval_status(cls, value):
        if value not in {"all", "approved", "reviewed_only", "not_reviewed", "not_approved"}:
            raise ValueError("Invalid approval_status. Use all | approved | reviewed_only | not_reviewed | not_approved")
        return value

    @validator("appeal_status")
    def validate_appeal_status(cls, value):
        if value not in {"all", "ongoing", "finished", "none"}:
            raise ValueError("Invalid appeal_status. Use all | ongoing | finished | none")
        return value


class ConversationExportGoogleSheetsResponse(BaseModel):
    spreadsheet_id: str
    spreadsheet_url: str
    rows_exported: int


class UploadResponse(BaseModel):
    message: Optional[str] = None
    error: Optional[str] = None

class EvaluationCriteriaCreate(BaseModel):
    name: str
    instruction: str
    max_value: Optional[int] = None
    minimum_value: Optional[int] = None
    is_active: bool = True
    context_lookback_mode: str = CriterionContextLookbackMode.DEFAULT.value
    context_lookback_minutes: Optional[int] = None
    order: Optional[int] = None

    @validator("context_lookback_mode")
    def validate_context_lookback_mode(cls, v):
        if not CriterionContextLookbackMode.contains(v):
            raise ValueError(
                f'Invalid context_lookback_mode value. Must be one of: {", ".join(CriterionContextLookbackMode.list())}'
            )
        return v

class EvaluationCriteriaUpdate(BaseModel):
    name: Optional[str] = None
    instruction: Optional[str] = None
    max_value: Optional[int] = None
    minimum_value: Optional[int] = None
    is_active: Optional[bool] = None
    context_lookback_mode: Optional[str] = None
    context_lookback_minutes: Optional[int] = None
    order: Optional[int] = None

    @validator("context_lookback_mode")
    def validate_context_lookback_mode(cls, v):
        if v is not None and not CriterionContextLookbackMode.contains(v):
            raise ValueError(
                f'Invalid context_lookback_mode value. Must be one of: {", ".join(CriterionContextLookbackMode.list())}'
            )
        return v

class EvaluationCriteriaResponse(BaseModel):
    id: str
    project_id: str
    name: str
    instruction: str
    max_value: Optional[int] = None
    minimum_value: Optional[int] = None
    is_active: bool = True
    context_lookback_mode: str
    context_lookback_minutes: Optional[int] = None
    order: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectAttributeCategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectAttributeCategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectAttributeCreate(BaseModel):
    name: str
    question: Optional[str] = None
    attribute_type: str
    is_active: bool = True
    allow_new_categories: bool = False
    allow_multiple_categories: bool = False
    max_selected_categories: Optional[int] = None
    show_seconds: bool = False
    run_after_evaluation: bool = False
    context_lookback_mode: str = CriterionContextLookbackMode.DEFAULT.value
    context_lookback_minutes: Optional[int] = None
    order: Optional[int] = None
    categories: List[ProjectAttributeCategoryCreate] = Field(default_factory=list)

    @field_validator("attribute_type")
    @classmethod
    def validate_attribute_type(cls, value: str) -> str:
        if not AttributeType.contains(value):
            raise ValueError(f'Invalid attribute_type. Must be one of: {", ".join(AttributeType.list())}')
        return value

    @validator("context_lookback_mode")
    def validate_context_lookback_mode(cls, value):
        if not CriterionContextLookbackMode.contains(value):
            raise ValueError(
                f'Invalid context_lookback_mode value. Must be one of: {", ".join(CriterionContextLookbackMode.list())}'
            )
        return value


class ProjectAttributeUpdate(BaseModel):
    name: Optional[str] = None
    question: Optional[str] = None
    is_active: Optional[bool] = None
    allow_new_categories: Optional[bool] = None
    allow_multiple_categories: Optional[bool] = None
    max_selected_categories: Optional[int] = None
    show_seconds: Optional[bool] = None
    run_after_evaluation: Optional[bool] = None
    context_lookback_mode: Optional[str] = None
    context_lookback_minutes: Optional[int] = None
    order: Optional[int] = None

    @validator("context_lookback_mode")
    def validate_context_lookback_mode(cls, value):
        if value is not None and not CriterionContextLookbackMode.contains(value):
            raise ValueError(
                f'Invalid context_lookback_mode value. Must be one of: {", ".join(CriterionContextLookbackMode.list())}'
            )
        return value


class ProjectAttributeCategoryResponse(BaseModel):
    id: str
    attribute_id: str
    name: str
    description: Optional[str] = None
    status: str
    is_active: bool
    proposal_conversation_id: Optional[str] = None
    deleted_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectAttributeCategoryTaxonomyDraftRequest(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    is_active: bool = True
    is_deleted: bool = False
    proposal_decision: Literal["keep", "accept", "reject", "merge"] = "keep"
    merge_target_category_id: Optional[str] = None


class ProjectAttributeTaxonomySaveRequest(BaseModel):
    name: str
    question: Optional[str] = None
    is_active: bool
    allow_new_categories: bool
    allow_multiple_categories: Optional[bool] = None
    max_selected_categories: Optional[int] = None
    run_after_evaluation: bool = False
    context_lookback_mode: str = CriterionContextLookbackMode.DEFAULT.value
    context_lookback_minutes: Optional[int] = None
    categories: List[ProjectAttributeCategoryTaxonomyDraftRequest] = Field(default_factory=list)

    @validator("context_lookback_mode")
    def validate_context_lookback_mode(cls, value):
        if not CriterionContextLookbackMode.contains(value):
            raise ValueError(
                f'Invalid context_lookback_mode value. Must be one of: {", ".join(CriterionContextLookbackMode.list())}'
            )
        return value


class ProjectAttributeResponse(BaseModel):
    id: str
    project_id: str
    name: str
    question: Optional[str] = None
    attribute_type: str
    is_active: bool
    allow_new_categories: bool
    allow_multiple_categories: bool = False
    max_selected_categories: Optional[int] = None
    show_seconds: bool
    run_after_evaluation: bool = False
    context_lookback_mode: str
    context_lookback_minutes: Optional[int] = None
    deleted_at: Optional[datetime] = None
    order: int
    created_at: datetime
    updated_at: datetime
    categories: List[ProjectAttributeCategoryResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class AttributeAnalyticsFiltersResponse(BaseModel):
    manager_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    chat_tags: List[str] = Field(default_factory=list)
    agent_tags: List[str] = Field(default_factory=list)
    approval_status: str

    class Config:
        from_attributes = True


class AttributeAnalyticsCategoryResponse(BaseModel):
    id: str
    name: str
    status: str
    is_active: bool
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AttributeAnalyticsCategoricalDistributionPointResponse(BaseModel):
    category: AttributeAnalyticsCategoryResponse
    count: int
    percentage: Optional[float] = None

    class Config:
        from_attributes = True


class AttributeAnalyticsCategoricalTrendPointResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    label: str
    count: int
    percentage: Optional[float] = None

    class Config:
        from_attributes = True


class AttributeAnalyticsCategoricalTrendSeriesResponse(BaseModel):
    category: AttributeAnalyticsCategoryResponse
    total_count: int
    points: List[AttributeAnalyticsCategoricalTrendPointResponse]

    class Config:
        from_attributes = True


class AttributeAnalyticsCategoricalDataResponse(BaseModel):
    total_count: int
    distribution: List[AttributeAnalyticsCategoricalDistributionPointResponse]
    trend: List[AttributeAnalyticsCategoricalTrendSeriesResponse]
    default_visible_category_ids: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


class AttributeAnalyticsNumericalHistogramBucketResponse(BaseModel):
    bucket_index: int
    label: str
    min_value: float
    max_value: float
    count: int
    filter_min_value: float
    filter_max_value: float

    class Config:
        from_attributes = True


class AttributeAnalyticsNumericalTrendPointResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    label: str
    average_value: Optional[float] = None
    conversation_count: int

    class Config:
        from_attributes = True


class AttributeAnalyticsNumericalDataResponse(BaseModel):
    conversation_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    average_value: Optional[float] = None
    histogram: List[AttributeAnalyticsNumericalHistogramBucketResponse]
    trend: List[AttributeAnalyticsNumericalTrendPointResponse]

    class Config:
        from_attributes = True


class AttributeAnalyticsResponse(BaseModel):
    attribute: ProjectAttributeResponse
    filters: AttributeAnalyticsFiltersResponse
    trend_granularity: str
    categorical: Optional[AttributeAnalyticsCategoricalDataResponse] = None
    numerical: Optional[AttributeAnalyticsNumericalDataResponse] = None
    time: Optional[AttributeAnalyticsNumericalDataResponse] = None
    day_time: Optional[AttributeAnalyticsNumericalDataResponse] = None

    class Config:
        from_attributes = True


class ProjectAttributeProposalDecisionRequest(BaseModel):
    target_category_id: Optional[str] = None


class ProjectAttributeCategoryMergeRequest(BaseModel):
    target_category_id: str


class EvaluationCriteriaVersionResponse(BaseModel):
    id: str
    criteria_id: str
    project_id: str
    version_number: int
    change_type: str
    reverted_from_version_number: Optional[int] = None
    actor_user_id: Optional[str] = None
    actor_name: str
    name: str
    instruction: str
    max_value: Optional[int] = None
    minimum_value: Optional[int] = None
    is_active: bool
    context_lookback_mode: str
    context_lookback_minutes: Optional[int] = None
    order: int
    created_at: datetime


class EvaluationCriteriaRevertRequest(BaseModel):
    version_number: int


class CriterionEvaluationHistoryResponse(BaseModel):
    conversation_id: str
    conversation_timestamp: datetime
    topic: Optional[str] = None
    manager_name: str
    chat_name: Optional[str] = None
    evaluation_approved: bool = False
    criterion_score_changed: bool = False
    score: int


class CriterionEvaluationHistoryPagination(BaseModel):
    page: int
    page_size: int
    has_next: bool


class CriterionEvaluationHistoryPageResponse(BaseModel):
    data: List[CriterionEvaluationHistoryResponse]
    pagination: CriterionEvaluationHistoryPagination


class CriterionEvaluationPreviewRequest(BaseModel):
    conversation_id: str
    instruction: str


class CriterionEvaluationPreviewResponse(BaseModel):
    criterion_id: str
    criterion_name: str
    score: int
    min_score: int
    max_score: int
    explanation: str
    prompt_loaded_from: Optional[datetime] = None
    prompt_loaded_to: Optional[datetime] = None
    message_references: Optional[Dict[str, str]] = None


class CriterionAutoImproveRequest(BaseModel):
    conversation_id: str
    instruction: str
    comment: str


class CriterionAutoImproveResponse(BaseModel):
    suggested_instruction: str

class KnowledgeBaseFileResponse(BaseModel):
    id: str
    project_id: str
    name: str
    content: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class KnowledgeBaseFileListResponse(BaseModel):
    id: str
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KnowledgeBaseFileReferenceResponse(BaseModel):
    source_type: Literal["criterion", "attribute_question", "attribute_category"]
    source_id: str
    source_name: str
    parent_id: Optional[str] = None
    parent_name: Optional[str] = None
    is_active: bool = True

    class Config:
        from_attributes = True

class PlaygroundMessagePayload(BaseModel):
    name: str
    datetime: str
    author_type: str  # "user" or "manager"
    body: str

class PlaygroundEvaluationRequest(BaseModel):
    project_id: str
    messages: List[PlaygroundMessagePayload]
    evaluation_method: str
    model: str
    conversation_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None


class PlaygroundRedactRequest(BaseModel):
    project_id: str
    messages: List[PlaygroundMessagePayload]
    pii_redaction_basic_enabled: bool = True
    pii_redaction_author_names_enabled: bool = True
    pii_redaction_names_enabled: bool = True


class PlaygroundRedactResponse(BaseModel):
    messages: List[PlaygroundMessage]

class PlaygroundCombinedEvaluationResponse(BaseModel):
    id: Optional[str] = None
    evaluation_result: Optional[EvaluationGeneralResult] = None
    flow_results: List[FlowEvaluationReport] = Field(default_factory=list)
    messages: List[PlaygroundMessage] = Field(default_factory=list)
    conversation_tags: List[str] = Field(default_factory=list)
    agent_tags: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
    }

    @field_serializer("evaluation_result")
    def _serialize_evaluation_result(self, value: EvaluationGeneralResult, _info):
        return jsonable_encoder(value)

    @field_serializer("flow_results")
    def _serialize_flow_results(self, value: List[FlowEvaluationReport], _info):
        items = []
        for fr in value or []:
            items.append({
                "flow_id": getattr(fr, "flow_id", None),
                "flow_name": getattr(fr, "flow_name", None),
                "evaluation": {
                    "step_evaluations": [
                        {
                            "step_id": se.step_id,
                            "step_name": se.step_name,
                            "step_type": getattr(se.step_type, "value", se.step_type),
                            "thoughts": se.thoughts,
                            "result": se.result,
                        }
                        for se in fr.step_evaluations
                    ],
                    "final_result": {
                        "completed": fr.final_result.completed,
                        "overall_assessment": fr.final_result.overall_assessment,
                        "final_step": fr.final_result.final_step,
                    },
                },
                "usage": {
                    "input_tokens": fr.usage.input_tokens,
                    "cached_input_tokens": fr.usage.cached_input_tokens,
                    "output_tokens": fr.usage.output_tokens,
                },
            })
        return items

    @field_serializer("messages")
    def _serialize_messages(self, value: List[PlaygroundMessage], _info):
        items = []
        for m in value or []:
            items.append({
                "name": getattr(m, "name", None),
                "datetime": getattr(m, "datetime", None),
                "author_type": getattr(m, "author_type", None),
                "body": getattr(m, "body", None),
            })
        return items

# Intercom models
class ComponentUsageAggregateResponse(BaseModel):
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    price_usd: Optional[float] = None
    count: int

class PlaygroundEvaluationUsageSummaryResponse(BaseModel):
    total_input_tokens: int
    total_cached_input_tokens: int
    total_output_tokens: int
    total_price_usd: Optional[float] = None
    run_count: int
    component_breakdown: Dict[str, ComponentUsageAggregateResponse]
    show_tokens: bool

    @classmethod
    def from_entity(cls, e: PlaygroundEvaluationUsageSummary, include_tokens: bool = True) -> "PlaygroundEvaluationUsageSummaryResponse":
        return cls(
            total_input_tokens=e.total_input_tokens if include_tokens else 0,
            total_cached_input_tokens=e.total_cached_input_tokens if include_tokens else 0,
            total_output_tokens=e.total_output_tokens if include_tokens else 0,
            total_price_usd=e.total_price_usd,
            run_count=e.run_count,
            component_breakdown={
                k: ComponentUsageAggregateResponse(
                    input_tokens=v.input_tokens if include_tokens else 0,
                    cached_input_tokens=v.cached_input_tokens if include_tokens else 0,
                    output_tokens=v.output_tokens if include_tokens else 0,
                    price_usd=v.price_usd,
                    count=v.count,
                ) for k, v in (e.component_breakdown or {}).items()
            },
            show_tokens=include_tokens,
        )

def _parse_helpshift_datetime(value: Any) -> datetime:
    """Parse a Helpshift timestamp into a naive UTC datetime.

    Helpshift may express timestamps as unix milliseconds, unix seconds, or ISO-8601
    strings depending on the endpoint/version, so handle all three. Falls back to
    ``datetime.utcfromtimestamp(0)`` epoch when the value is missing/unparseable.
    """
    if value is None or value == "":
        return datetime.utcfromtimestamp(0)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(pytz.UTC).replace(tzinfo=None)
        return value
    if isinstance(value, (int, float)):
        timestamp = float(value)
        # Heuristic: values above ~year 2286 in seconds are actually milliseconds.
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.utcfromtimestamp(timestamp)
    normalized = str(value).strip()
    if not normalized:
        return datetime.utcfromtimestamp(0)
    if normalized.isdigit():
        return _parse_helpshift_datetime(int(normalized))
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone(pytz.UTC).replace(tzinfo=None)
        return parsed
    except ValueError:
        return datetime.utcfromtimestamp(0)


_HELPSHIFT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".svg")


def _format_helpshift_attachment_markers(message: dict) -> str:
    """Build attachment markers for a Helpshift message.

    Mirrors Telegram's media markers: returns ``[Image: name]`` for images and
    ``[Attachment: name]`` for other files (one line per attachment). Helpshift may express
    attachments either as a single ``attachment`` dict or an ``attachments`` list.
    """
    raw_attachments: list[dict] = []
    single = message.get("attachment")
    if isinstance(single, dict):
        raw_attachments.append(single)
    many = message.get("attachments")
    if isinstance(many, list):
        raw_attachments.extend(item for item in many if isinstance(item, dict))

    markers: list[str] = []
    for attachment in raw_attachments:
        file_name = attachment.get("file_name") or attachment.get("name")
        content_type = str(attachment.get("content_type") or attachment.get("content-type") or "").lower()
        lowered_name = str(file_name or "").lower()
        is_image = content_type.startswith("image/") or lowered_name.endswith(_HELPSHIFT_IMAGE_EXTENSIONS)
        label = "Image" if is_image else "Attachment"
        markers.append(f"[{label}: {file_name}]" if file_name else f"[{label}]")
    return "\n".join(markers)


class ImportedConversationAuthor(BaseModel):
    """Represents an author in normalized imported conversation data."""
    source_id: str = Field(..., alias="id")
    type: str
    name: str
    email: Optional[str] = None
    manager_id: Optional[str] = None  # ID of the manager in our database, if this author is a manager

    @field_validator("name", mode="before")
    @classmethod
    def normalize_missing_name(cls, value):
        # Intercom can return null author names for some sources; keep import paths resilient.
        if value is None:
            return "Unknown"
        normalized = str(value).strip()
        return normalized or "Unknown"

    class Config:
        allow_population_by_field_name = True

class ImportedConversationSource(BaseModel):
    """Represents the first/source message in normalized imported conversation data."""
    source_id: str = Field(..., alias="id")
    type: Optional[str] = None  # Type of the source (e.g., "email")
    subject: Optional[str] = None
    body: Optional[str] = None
    author: Optional[ImportedConversationAuthor] = None
    is_internal_note: bool = False  # Private agent note not visible to the customer

    class Config:
        allow_population_by_field_name = True

class ImportedConversationMessage(BaseModel):
    """Represents a non-source message in normalized imported conversation data."""
    source_id: str = Field(..., alias="id")
    part_type: str
    body: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    author: ImportedConversationAuthor
    is_internal_note: bool = False  # Private agent note not visible to the customer

    class Config:
        allow_population_by_field_name = True

class ImportedConversationData(BaseModel):
    """Represents a normalized imported conversation before persistence."""
    source_id: str = Field(..., alias="id")
    project_id: str
    state: str
    created_at: datetime
    updated_at: datetime
    loaded_from: Optional[datetime] = None
    loaded_to: Optional[datetime] = None
    source: Optional[ImportedConversationSource] = None
    conversation_parts: List[ImportedConversationMessage]
    
    class Config:
        allow_population_by_field_name = True
        
    @classmethod
    def from_intercom_api_response(cls, data: dict, project_id: str) -> "ImportedConversationData":
        """Create normalized imported conversation data from an Intercom API response."""
        return cls(
            id=data["id"],  # Use id since aliasing works both ways
            project_id=project_id,
            state=data.get("state", ""),
            created_at=datetime.fromtimestamp(data.get("created_at", 0)),
            updated_at=datetime.fromtimestamp(data.get("updated_at", 0)),
            source=ImportedConversationSource(**data["source"]) if data.get("source") else None,
            conversation_parts=[
                # Intercom internal notes are conversation parts with part_type "note".
                ImportedConversationMessage(
                    **{**part, "is_internal_note": str(part.get("part_type") or "").strip().lower() == "note"}
                )
                for part in data.get("conversation_parts", {}).get("conversation_parts", [])
            ]
        )

    @classmethod
    def from_helpshift_api_response(cls, issue: dict, project_id: str) -> "ImportedConversationData":
        """Create normalized imported conversation data from a Helpshift issue.

        Helpshift returns messages inline on the issue. The first message is treated as the
        conversation source; the rest become conversation parts. Author type follows the
        message ``origin``: ``end-user`` is the customer (``user``), everything else is an
        agent/system reply (``admin``) so it is attributed to a manager.

        The conversation ``source_id`` uses the stable scheme ``hs_{project_id}_{issue_id}``
        so it can be matched by the chat-tag resolution/filter code.

        Private/internal agent notes (returned inline in ``messages``) are kept and flagged
        structurally via ``is_internal_note`` (the body text is left untouched) so the evaluator
        treats them as internal context rather than customer-facing replies.
        """
        issue_id = str(issue["id"])
        state_data = issue.get("state_data") or {}
        messages = [msg for msg in (issue.get("messages") or []) if isinstance(msg, dict)]

        parsed_messages = [cls._helpshift_message_to_dict(issue_id, idx, msg) for idx, msg in enumerate(messages)]

        source: Optional[ImportedConversationSource] = None
        parts_payload = parsed_messages
        if parsed_messages:
            first = parsed_messages[0]
            source = ImportedConversationSource(
                id=first["id"],
                type="helpshift",
                subject=issue.get("title"),
                body=first["body"],
                author=ImportedConversationAuthor(**first["author"]),
                is_internal_note=first["is_internal_note"],
            )
            parts_payload = parsed_messages[1:]

        return cls(
            id=f"hs_{project_id}_{issue_id}",
            project_id=project_id,
            state=state_data.get("state", ""),
            created_at=_parse_helpshift_datetime(issue.get("created_at")),
            updated_at=_parse_helpshift_datetime(issue.get("updated_at")),
            source=source,
            conversation_parts=[ImportedConversationMessage(**part) for part in parts_payload],
        )

    @staticmethod
    def _helpshift_message_to_dict(issue_id: str, index: int, message: dict) -> dict:
        """Normalize a single Helpshift message into ImportedConversationMessage kwargs."""
        origin = str(message.get("origin") or "").strip().lower()
        author = message.get("author") or {}
        emails = author.get("emails") or []
        author_email = emails[0] if isinstance(emails, list) and emails else None
        # end-user => customer; any other origin (helpshift agent, bot, system) => admin/agent side.
        author_type = "user" if origin == "end-user" else "admin"
        created_at = _parse_helpshift_datetime(message.get("created_at"))

        body = message.get("body") or ""
        # Surface attachments the way Telegram marks media: always append a marker, even when the
        # message also has text, so images/files are not silently dropped from the transcript.
        marker_text = _format_helpshift_attachment_markers(message)
        if body and marker_text:
            body = f"{body}\n{marker_text}"
        elif marker_text:
            body = marker_text

        # Private/internal agent notes are returned inline. Keep them but flag them structurally
        # (not in the body text) so the evaluator treats them as internal context.
        message_type = str(message.get("type") or "").strip().lower().replace("_", " ")
        is_internal_note = message_type in HELPSHIFT_INTERNAL_NOTE_TYPES

        message_source_id = str(message.get("id") or f"{issue_id}_{index}")
        # Namespace the author id so Helpshift agents never collide with Intercom admins on the
        # manager (source_id, project_id) key (both use source_type="admin").
        raw_author_id = author.get("id")
        author_source_id = (
            f"{HELPSHIFT_AGENT_SOURCE_ID_PREFIX}{raw_author_id}"
            if raw_author_id not in (None, "")
            else f"{HELPSHIFT_AGENT_SOURCE_ID_PREFIX}{origin or 'unknown'}:{issue_id}"
        )
        return {
            "id": message_source_id,
            "part_type": "comment",
            "body": body,
            "created_at": created_at,
            "updated_at": created_at,
            "is_internal_note": is_internal_note,
            "author": {
                "id": author_source_id,
                "type": author_type,
                "name": author.get("name"),
                "email": author_email,
            },
        }

# Zendesk models
class ZendeskAuthor(BaseModel):
    """Represents an author of a ticket comment from Zendesk."""
    source_id: str = Field(..., alias="id")
    name: str
    email: Optional[str] = None
    role: Optional[str] = None
    manager_id: Optional[str] = None  # ID of the manager in our database, if this author is a manager

    @property
    def type(self) -> Optional[str]:
        """Map Zendesk role to type for compatibility."""
        return self.role
    
    @field_validator('source_id', mode='before')
    @classmethod
    def convert_id_to_string(cls, v):
        return str(v)

    class Config:
        allow_population_by_field_name = True

class ZendeskComment(BaseModel):
    """Represents a comment in a ticket from Zendesk."""
    source_id: str = Field(..., alias="id")
    type: str
    body: str
    html_body: Optional[str] = None
    plain_body: Optional[str] = None
    public: bool
    created_at: datetime
    author_id: str
    author: Optional[ZendeskAuthor] = None
    
    @field_validator('source_id', 'author_id', mode='before')
    @classmethod
    def convert_ids_to_string(cls, v):
        return str(v)
    
    @field_validator('created_at', mode='before')
    @classmethod
    def normalize_datetime(cls, v):
        """Convert timezone-aware datetime to timezone-naive UTC."""
        if isinstance(v, datetime):
            if v.tzinfo is not None:
                # Convert to UTC and make timezone-naive
                return v.astimezone(pytz.UTC).replace(tzinfo=None)
        return v
    
    class Config:
        allow_population_by_field_name = True

class ZendeskTicketData(BaseModel):
    """Represents a ticket with its comments from Zendesk."""
    source_id: str = Field(..., alias="id")
    project_id: str
    subject: Optional[str] = None
    description: Optional[str] = None
    status: str
    priority: Optional[str] = None
    type: Optional[str] = None
    channel: Optional[str] = None
    requester_email: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    comments: List[ZendeskComment]
    
    @property
    def state(self) -> str:
        """Map Zendesk status to conversation state for compatibility."""
        return self.status
    
    @property
    def source(self) -> Optional[object]:
        """Map Zendesk ticket description to source for compatibility."""
        if self.description:
            # Create a simple object with the required attributes
            class SourceObject:
                def __init__(self, source_id, body, author):
                    self.source_id = source_id
                    self.body = body
                    self.author = author
            
            return SourceObject(
                source_id=f"{self.source_id}_description",
                body=self.description,
                author=None  # Will be set during creation if requester data is available
            )
        return None
    
    @property
    def conversation_parts(self) -> List[object]:
        """Map Zendesk comments to conversation parts for compatibility."""
        parts = []
        for comment in self.comments:
            # Create a simple object with the required attributes
            class ConversationPart:
                def __init__(self, source_id, body, author, created_at, is_internal_note):
                    self.source_id = source_id
                    self.body = body
                    self.author = author
                    self.created_at = created_at
                    self.is_internal_note = is_internal_note

            parts.append(ConversationPart(
                source_id=comment.source_id,
                body=comment.body,
                author=comment.author,
                created_at=comment.created_at,
                # Zendesk non-public comments are internal notes (not visible to the requester).
                is_internal_note=comment.public is False,
            ))
        return parts
    
    @field_validator('source_id', mode='before')
    @classmethod
    def convert_id_to_string(cls, v):
        return str(v)
    
    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def normalize_datetime(cls, v):
        """Convert timezone-aware datetime to timezone-naive UTC."""
        if isinstance(v, datetime):
            if v.tzinfo is not None:
                # Convert to UTC and make timezone-naive
                return v.astimezone(pytz.UTC).replace(tzinfo=None)
        return v
    
    class Config:
        allow_population_by_field_name = True
        
    @classmethod
    def from_zendesk_data(cls, ticket_data: dict, comments_data: List[dict], project_id: str) -> "ZendeskTicketData":
        """Create ZendeskTicketData from Zendesk API response."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        logger.debug(f"Ticket data keys: {list(ticket_data.keys())}")
        logger.debug(f"Ticket requester info: requester_id={ticket_data.get('requester_id')}, submitter_id={ticket_data.get('submitter_id')}")
        logger.debug(f"Converting {len(comments_data)} comments to ZendeskComment objects")
        converted_comments = []
        for i, comment in enumerate(comments_data):
            logger.debug(f"Converting comment {i}: id={comment.get('id')}, has_author={bool(comment.get('author'))}")
            if comment.get('author'):
                logger.debug(f"  Author data: {comment['author']}")
            try:
                zendesk_comment = ZendeskComment(**comment)
                logger.debug(f"  Converted successfully, author: {zendesk_comment.author}")
                converted_comments.append(zendesk_comment)
            except Exception as e:
                logger.error(f"  Failed to convert comment {comment.get('id')}: {e}")
                raise
        
        # Create the instance
        instance = cls(
            id=ticket_data["id"],
            project_id=project_id,
            subject=ticket_data.get("subject", ""),
            description=ticket_data.get("description", ""),
            status=ticket_data.get("status", ""),
            priority=ticket_data.get("priority"),
            type=ticket_data.get("type"),
            channel=ticket_data.get("via", {}).get("channel"),
            requester_email=ticket_data.get("requester_email"),
            created_at=datetime.fromisoformat(ticket_data.get("created_at", "").replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(ticket_data.get("updated_at", "").replace("Z", "+00:00")),
            comments=converted_comments
        )
        
        logger.debug(f"Created ZendeskTicketData with requester_author: {bool(ticket_data.get('requester_author'))}")
        
        return instance

class ManagerResponse(BaseModel):
    """Response model for manager data."""
    id: str
    source_id: str
    source_type: Optional[str]
    name: str
    email: Optional[str]
    tags: List[str] = Field(default_factory=list)
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None
    project_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ManagerUpdate(BaseModel):
    """Source-agnostic update payload for editing an agent's tags and evaluation hints.

    Used by the generic manager PATCH endpoint so every integration (Intercom, Telegram,
    TimelinesAI, Zendesk, ...) edits agent tags through one path without source-type branching.
    """
    tags: Optional[List[str]] = None
    evaluation_instruction: Optional[str] = None
    evaluation_language: Optional[str] = None

class ManagerEvaluationResponse(BaseModel):
    """Response model for manager evaluation results."""
    manager_id: str
    criteria_averages: Dict[str, float]
    total_average: float


class ManagerDashboardPeriod(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL_TIME = "all_time"


class ManagerDashboardApprovalStatus(str, Enum):
    ALL = "all"
    APPROVED = "approved"


class ManagerDashboardRowResponse(BaseModel):
    period_start: date
    period_end: date
    period_label: str
    manager_id: str
    manager_name: str
    manager_email: Optional[str]
    chats_count: int
    total_average: float
    criteria_averages: Dict[str, float]
    notes_count: int = 0
    latest_note_text: Optional[str] = None
    latest_note_author_name: Optional[str] = None
    latest_note_updated_at: Optional[datetime] = None


class ManagerDashboardCriterionResponse(BaseModel):
    name: str
    is_active: bool


class ManagerPerformanceOverviewAgentResponse(BaseModel):
    id: str
    name: str
    email: Optional[str] = None


class ManagerPerformanceOverviewCriterionResponse(BaseModel):
    id: str
    name: str
    is_active: bool


class ManagerPerformanceOverviewSeriesResponse(BaseModel):
    id: str
    manager_id: str
    manager_name: str
    score_key: str
    label: str


class ManagerPerformanceOverviewPeriodResponse(BaseModel):
    period_start: date
    period_end: date
    period_label: str
    values: Dict[str, Optional[float]]


class ManagerPerformanceOverviewResponse(BaseModel):
    agents: List[ManagerPerformanceOverviewAgentResponse]
    criteria: List[ManagerPerformanceOverviewCriterionResponse]
    series: List[ManagerPerformanceOverviewSeriesResponse]
    periods: List[ManagerPerformanceOverviewPeriodResponse]


class CustomDashboardColumnKindResponse(str, Enum):
    PREDEFINED = CustomDashboardColumnKind.PREDEFINED.value
    CRITERION = CustomDashboardColumnKind.CRITERION.value
    ATTRIBUTE_NUMERICAL = CustomDashboardColumnKind.ATTRIBUTE_NUMERICAL.value
    ATTRIBUTE_TIME = CustomDashboardColumnKind.ATTRIBUTE_TIME.value
    ATTRIBUTE_DAY_TIME = CustomDashboardColumnKind.ATTRIBUTE_DAY_TIME.value
    ATTRIBUTE_CATEGORICAL = CustomDashboardColumnKind.ATTRIBUTE_CATEGORICAL.value


class CustomDashboardPredefinedMetricResponse(str, Enum):
    AVG_START_TIME = CustomDashboardPredefinedMetric.AVG_START_TIME.value
    AVG_FINISH_TIME = CustomDashboardPredefinedMetric.AVG_FINISH_TIME.value
    CHATS_COUNT = CustomDashboardPredefinedMetric.CHATS_COUNT.value
    MESSAGES_COUNT = CustomDashboardPredefinedMetric.MESSAGES_COUNT.value
    AVG_TOTAL_SCORE = CustomDashboardPredefinedMetric.AVG_TOTAL_SCORE.value


class CustomDashboardAggregationResponse(str, Enum):
    SUM = CustomDashboardNumericalAggregation.SUM.value
    AVG = CustomDashboardNumericalAggregation.AVG.value
    MEDIAN = CustomDashboardNumericalAggregation.MEDIAN.value
    MIN = CustomDashboardNumericalAggregation.MIN.value
    MAX = CustomDashboardNumericalAggregation.MAX.value
    COUNT = CustomDashboardCategoricalAggregation.COUNT.value
    PERCENTAGE = CustomDashboardCategoricalAggregation.PERCENTAGE.value


class ProjectCustomDashboardColumnWriteRequest(BaseModel):
    column_kind: CustomDashboardColumnKindResponse
    predefined_metric: Optional[CustomDashboardPredefinedMetricResponse] = None
    attribute_id: Optional[str] = None
    criterion_id: Optional[str] = None
    category_id: Optional[str] = None
    aggregation: Optional[CustomDashboardAggregationResponse] = None
    custom_unit: Optional[str] = None
    include_empty_conversations: bool = True


class ProjectCustomDashboardCreateRequest(BaseModel):
    name: str
    columns: List[ProjectCustomDashboardColumnWriteRequest]


class ProjectCustomDashboardUpdateRequest(BaseModel):
    name: str
    columns: List[ProjectCustomDashboardColumnWriteRequest]


class ProjectCustomDashboardColumnResponse(BaseModel):
    id: str
    order: int
    column_kind: CustomDashboardColumnKindResponse
    predefined_metric: Optional[CustomDashboardPredefinedMetricResponse] = None
    attribute_id: Optional[str] = None
    criterion_id: Optional[str] = None
    category_id: Optional[str] = None
    aggregation: Optional[CustomDashboardAggregationResponse] = None
    custom_unit: Optional[str] = None
    include_empty_conversations: bool = True
    criterion_name: Optional[str] = None
    criterion_is_active: Optional[bool] = None
    criterion_missing: bool = False
    attribute_name: Optional[str] = None
    attribute_type: Optional[str] = None
    attribute_show_seconds: Optional[bool] = None
    attribute_deleted_at: Optional[datetime] = None
    category_name: Optional[str] = None
    category_status: Optional[str] = None
    category_is_active: Optional[bool] = None
    category_deleted_at: Optional[datetime] = None


class ProjectCustomDashboardResponse(BaseModel):
    id: str
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    columns: List[ProjectCustomDashboardColumnResponse]


class ProjectCustomDashboardCatalogAttributeCategoryResponse(BaseModel):
    id: str
    name: str
    status: str
    is_active: bool
    deleted_at: Optional[datetime] = None


class ProjectCustomDashboardCatalogAttributeResponse(BaseModel):
    id: str
    name: str
    attribute_type: str
    show_seconds: bool = False
    is_active: bool
    deleted_at: Optional[datetime] = None
    categories: List[ProjectCustomDashboardCatalogAttributeCategoryResponse]


class ProjectCustomDashboardCatalogCriterionResponse(BaseModel):
    id: str
    name: str
    is_active: bool


class ProjectCustomDashboardCatalogResponse(BaseModel):
    attributes: List[ProjectCustomDashboardCatalogAttributeResponse]
    criteria: List[ProjectCustomDashboardCatalogCriterionResponse]


class ProjectCustomDashboardRowsColumnResponse(ProjectCustomDashboardColumnResponse):
    display_name: str


class ProjectCustomDashboardRowResponse(BaseModel):
    manager_id: str
    manager_name: str
    manager_email: Optional[str]
    values: Dict[str, Optional[float]]


class ProjectCustomDashboardRowsResponse(BaseModel):
    dashboard: ProjectCustomDashboardResponse
    columns: List[ProjectCustomDashboardRowsColumnResponse]
    rows: List[ProjectCustomDashboardRowResponse]


class ManagerPeriodNoteResponse(BaseModel):
    id: str
    author_user_id: str
    author_name: str
    note: str
    created_at: datetime
    updated_at: datetime
    can_edit: bool


class ManagerPeriodNotesListResponse(BaseModel):
    notes: List[ManagerPeriodNoteResponse]


class ManagerPeriodNoteUpsertRequest(BaseModel):
    period: ManagerDashboardPeriod
    period_start: Optional[date] = None
    note: str = Field(..., max_length=5000)


class ManagerPeriodNoteGenerationRequest(BaseModel):
    period: ManagerDashboardPeriod
    period_start: Optional[date] = None
    approval_status: ManagerDashboardApprovalStatus = ManagerDashboardApprovalStatus.ALL
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chat_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None
    evaluation_tags: Optional[List[str]] = None
    one_off_evaluation_run_batch_id: Optional[str] = None
    include_conversation_links: bool = True


class ManagerPeriodNoteGenerationInputSummaryResponse(BaseModel):
    conversation_count: int
    accepted_appeal_count: int
    criteria_count: int
    agent_id: str
    agent_name: str
    period: str
    period_label: str
    filters: Dict[str, Any] = Field(default_factory=dict)


class ManagerPeriodNoteGenerationReferenceResponse(BaseModel):
    conversation_id: str
    url: str


class ManagerPeriodNoteGenerationResponse(BaseModel):
    status: Literal["generated", "no_data"]
    note: str
    input_summary: ManagerPeriodNoteGenerationInputSummaryResponse
    references: List[ManagerPeriodNoteGenerationReferenceResponse] = Field(default_factory=list)


class ManagerDashboardTableResponse(BaseModel):
    data: List[ManagerDashboardRowResponse]
    pagination: 'PaginationMetadata'
    criteria_names: List[str]
    criteria: List[ManagerDashboardCriterionResponse] = Field(default_factory=list)
    supports_performance_overview_filters: bool = False


class ManagerDashboardExportFormat(str, Enum):
    CSV = "csv"
    GOOGLE_SHEETS = "google_sheets"


class ManagerDashboardExportRequest(BaseModel):
    export_format: ManagerDashboardExportFormat
    period: ManagerDashboardPeriod = ManagerDashboardPeriod.MONTH
    approval_status: ManagerDashboardApprovalStatus = ManagerDashboardApprovalStatus.ALL
    include_inactive_criteria: bool = False
    manager_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chat_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None
    evaluation_tags: Optional[List[str]] = None
    one_off_evaluation_run_batch_id: Optional[str] = None
    share_email: Optional[EmailStr] = None


class ManagerDashboardExportGoogleSheetsResponse(BaseModel):
    spreadsheet_id: str
    spreadsheet_url: str
    rows_exported: int


class PaginationMetadata(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int

class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    pagination: PaginationMetadata 

class EvaluationCriterionReviewInput(BaseModel):
    score_override: Optional[int] = None
    comment: Optional[str] = None


class EvaluationAttributeReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numeric_value: Optional[float] = None
    category_ids: Optional[List[str]] = None
    timezone_name: Optional[str] = None
    timezone_offset: Optional[str] = None


class EvaluationReviewUpsertRequest(BaseModel):
    approved: Optional[bool] = None
    general_comment: str = ""
    criteria_reviews: Dict[str, EvaluationCriterionReviewInput] = Field(default_factory=dict)
    attribute_reviews: Dict[str, EvaluationAttributeReviewInput] = Field(default_factory=dict)
    attribute_reapply_draft_ids: List[str] = Field(default_factory=list)
    alerts_marked_read_ids: List[str] = Field(default_factory=list)
    alerts_marked_unread_ids: List[str] = Field(default_factory=list)
    alerts_triggered_ids: List[str] = Field(default_factory=list)
    included_manager_ids: Optional[List[str]] = None


class ConversationAttributeReapplyRequest(BaseModel):
    general_comment: str = ""
    criteria_reviews: Dict[str, EvaluationCriterionReviewInput] = Field(default_factory=dict)
    attribute_reviews: Dict[str, EvaluationAttributeReviewInput] = Field(default_factory=dict)
    attribute_reapply_draft_ids: List[str] = Field(default_factory=list)
    included_manager_ids: Optional[List[str]] = None


class ConversationAttributeReapplyDraftCleanupRequest(BaseModel):
    draft_ids: List[str] = Field(default_factory=list)


class GeneralCommentRegenerationRequest(BaseModel):
    general_comment: str = ""
    criteria_reviews: Dict[str, EvaluationCriterionReviewInput] = Field(default_factory=dict)
    additional_instruction: str = ""


class GeneralCommentRegenerationResponse(BaseModel):
    text_comment: str


class GeneralCommentReplacementRequest(BaseModel):
    text_comment: str


class EvaluationCriterionReviewResponse(BaseModel):
    score_override: Optional[int] = None
    comment: Optional[str] = None


class EvaluationAttributeReviewResponse(BaseModel):
    numeric_value: Optional[float] = None
    category_ids: Optional[List[str]] = None
    timezone_name: Optional[str] = None
    timezone_offset: Optional[str] = None


class EvaluationReviewResponse(BaseModel):
    general_comment: str = ""
    criteria_reviews: Dict[str, EvaluationCriterionReviewResponse] = Field(default_factory=dict)
    attribute_reviews: Dict[str, EvaluationAttributeReviewResponse] = Field(default_factory=dict)
    included_manager_ids: Optional[List[str]] = None


class EvaluationAppealCommentCreateRequest(BaseModel):
    appeal_item_id: str
    content: str
    version: int


class EvaluationAppealCommentUpdateRequest(BaseModel):
    content: str
    version: int


class EvaluationAppealCommentResponse(BaseModel):
    id: str
    appeal_item_id: str
    author_user_id: str
    author_name: Optional[str] = None
    content: str
    created_at: datetime
    updated_at: datetime


class EvaluationScoreEventResponse(BaseModel):
    id: str
    target_type: str
    target_id: str
    criterion_id: Optional[str] = None
    criterion_name: Optional[str] = None
    appeal_item_id: Optional[str] = None
    actor_user_id: str
    actor_name: Optional[str] = None
    event_type: str
    previous_score: Optional[int] = None
    next_score: Optional[int] = None
    created_at: datetime


class EvaluationAppealItemSaveRequest(BaseModel):
    appeal_item_id: Optional[str] = None
    criteria_id: Optional[str] = None
    criterion_name: Optional[str] = None
    is_general: bool = False
    agent_proposed_score: Optional[int] = None


class EvaluationAppealDecisionSaveRequest(BaseModel):
    appeal_item_id: str
    qc_decision: str
    mistake_attribution: Optional[str] = None
    final_score: Optional[int] = None

    @field_validator("qc_decision")
    @classmethod
    def validate_qc_decision(cls, value: str) -> str:
        if value not in EvaluationAppealDecision.list():
            raise ValueError(
                f'Invalid qc_decision. Must be one of: {", ".join(EvaluationAppealDecision.list())}'
            )
        return value

    @field_validator("mistake_attribution")
    @classmethod
    def validate_mistake_attribution(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in EvaluationAppealMistakeAttribution.list():
            raise ValueError(
                f'Invalid mistake_attribution. Must be one of: {", ".join(EvaluationAppealMistakeAttribution.list())}'
            )
        return value


class EvaluationAppealSaveRequest(BaseModel):
    version: Optional[int] = None
    initialize_for_comment: bool = False
    items: List[EvaluationAppealItemSaveRequest] = Field(default_factory=list)


class EvaluationAppealDecisionRequest(BaseModel):
    version: int
    items: List[EvaluationAppealDecisionSaveRequest] = Field(default_factory=list)


class EvaluationAppealCloseRequest(BaseModel):
    version: int


class EvaluationAppealReopenRequest(BaseModel):
    version: int


class EvaluationAppealItemResponse(BaseModel):
    id: str
    criteria_id: Optional[str] = None
    criterion_name: Optional[str] = None
    is_general: bool = False
    agent_proposed_score: Optional[int] = None
    accepted_score: Optional[int] = None
    qc_decision: str
    mistake_attribution: str = "not_counted"
    decided_by_user_id: Optional[str] = None
    decided_by_name: Optional[str] = None
    decided_at: Optional[datetime] = None
    comments: List[EvaluationAppealCommentResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class EvaluationAppealResponse(BaseModel):
    id: str
    conversation_id: str
    project_id: str
    status: str
    started_by_agent_user_id: str
    started_by_agent_name: Optional[str] = None
    linked_reviewer_user_id: Optional[str] = None
    linked_reviewer_name: Optional[str] = None
    version: int
    closed_at: Optional[datetime] = None
    reopened_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    items: List[EvaluationAppealItemResponse] = Field(default_factory=list)


class EvaluationAppealListItemResponse(BaseModel):
    id: str
    conversation_id: str
    project_id: str
    status: str
    started_by_agent_user_id: str
    started_by_agent_name: Optional[str] = None
    linked_reviewer_user_id: Optional[str] = None
    linked_reviewer_name: Optional[str] = None
    reviewed_by_user_id: Optional[str] = None
    reviewed_by_name: Optional[str] = None
    topic: Optional[str] = None
    chat_name: Optional[str] = None
    chat_tags: List[str] = Field(default_factory=list)
    manager_name: Optional[str] = None
    manager_tags: List[str] = Field(default_factory=list)
    appealed_criteria_count: int
    total_score: int
    max_score: int
    updated_at: datetime


class EvaluationAppealCountResponse(BaseModel):
    count: int


class EvaluationAppealCriterionSummaryRow(BaseModel):
    qc_decision: str
    mistake_attribution: str
    count: int


class EvaluationAppealCriterionSummaryResponse(BaseModel):
    rows: List[EvaluationAppealCriterionSummaryRow] = Field(default_factory=list)


class EvaluationAppealStatsSummaryResponse(BaseModel):
    appealed_criteria_count: int
    conversations_count: int
    accepted_count: int
    rejected_count: int
    pending_count: int
    accepted_qc_mistake_count: int
    accepted_no_mistake_count: int


class EvaluationAppealStatsOriginalQcRowResponse(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    appealed_criteria_count: int
    accepted_count: int
    accepted_qc_mistake_count: int
    rejected_count: int
    accepted_no_mistake_count: int
    pending_count: int
    accepted_rate: int


class EvaluationAppealStatsCriterionRowResponse(BaseModel):
    criteria_id: Optional[str] = None
    criterion_name: str
    appealed_criteria_count: int
    accepted_count: int
    rejected_count: int
    accepted_no_mistake_count: int
    pending_count: int


class EvaluationAppealStatsReviewerRowResponse(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    criteria_count: int
    accepted_count: int
    rejected_count: int
    pending_count: int


class EvaluationAppealStatsAgentRowResponse(BaseModel):
    user_id: str
    name: Optional[str] = None
    opened_count: int
    accepted_count: int
    mistaken_count: int
    success_rate: int


class EvaluationAppealStatsResponse(BaseModel):
    summary: EvaluationAppealStatsSummaryResponse
    original_qc_rows: List[EvaluationAppealStatsOriginalQcRowResponse] = Field(default_factory=list)
    criterion_rows: List[EvaluationAppealStatsCriterionRowResponse] = Field(default_factory=list)
    appeal_reviewer_rows: List[EvaluationAppealStatsReviewerRowResponse] = Field(default_factory=list)
    agent_rows: List[EvaluationAppealStatsAgentRowResponse] = Field(default_factory=list)


class ReviewableManagerResponse(BaseModel):
    id: str
    name: str


class ConversationAttributeCategoryResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    is_active: bool = True


class ConversationAttributeDefinitionResponse(BaseModel):
    id: str
    name: str
    question: Optional[str] = None
    attribute_type: str
    show_seconds: bool = False
    allow_multiple_categories: bool = False
    max_selected_categories: Optional[int] = None
    run_after_evaluation: bool = False


class ConversationAttributeResultResponse(BaseModel):
    id: str
    attribute: ConversationAttributeDefinitionResponse
    category: Optional[ConversationAttributeCategoryResponse] = None
    backup_category: Optional[ConversationAttributeCategoryResponse] = None
    llm_category: Optional[ConversationAttributeCategoryResponse] = None
    categories: List[ConversationAttributeCategoryResponse] = Field(default_factory=list)
    backup_categories: List[ConversationAttributeCategoryResponse] = Field(default_factory=list)
    llm_categories: List[ConversationAttributeCategoryResponse] = Field(default_factory=list)
    available_categories: List[ConversationAttributeCategoryResponse] = Field(default_factory=list)
    numeric_value: Optional[float] = None
    timezone_name: Optional[str] = None
    timezone_offset: Optional[str] = None
    explanation: str
    message_references: Optional[Dict[str, str]] = None
    extracted_at: datetime


class ConversationAttributeReapplyResponse(BaseModel):
    draft_id: str
    attribute_result: ConversationAttributeResultResponse


class ConversationAlertResponse(BaseModel):
    id: str
    alert_id: Optional[str] = None
    activation_behavior: Optional[str] = None
    status: str
    alert_name: str
    severity: str
    target_kind: str
    condition_summary: str
    matched_value_summary: str
    triggered_at: datetime
    updated_at: datetime
    read_at: Optional[datetime] = None
    read_by_user_id: Optional[str] = None


class ProjectAlertBase(BaseModel):
    name: str
    severity: str
    is_active: bool = True
    target_kind: str
    operator: str
    attribute_id: Optional[str] = None
    category_id: Optional[str] = None
    criterion_id: Optional[str] = None
    threshold_numeric: Optional[float] = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        if not AlertSeverity.contains(value):
            raise ValueError(f'Invalid severity. Must be one of: {", ".join(AlertSeverity.list())}')
        return value

    @field_validator("target_kind")
    @classmethod
    def validate_target_kind(cls, value: str) -> str:
        if not ProjectAlertTargetKind.contains(value):
            raise ValueError(
                f'Invalid target_kind. Must be one of: {", ".join(ProjectAlertTargetKind.list())}'
            )
        return value

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        if not ProjectAlertOperator.contains(value):
            raise ValueError(f'Invalid operator. Must be one of: {", ".join(ProjectAlertOperator.list())}')
        return value


class ProjectAlertCreate(ProjectAlertBase):
    activation_behavior: Optional[str] = None

    @field_validator("activation_behavior")
    @classmethod
    def validate_activation_behavior(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not ProjectAlertActivationBehavior.contains(value):
            raise ValueError(
                "Invalid activation_behavior. Must be one of: "
                f"{', '.join(ProjectAlertActivationBehavior.list())}"
            )
        return value


class ProjectAlertUpdate(ProjectAlertBase):
    activation_behavior: Optional[str] = None

    @field_validator("activation_behavior")
    @classmethod
    def validate_activation_behavior(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not ProjectAlertActivationBehavior.contains(value):
            raise ValueError(
                "Invalid activation_behavior. Must be one of: "
                f"{', '.join(ProjectAlertActivationBehavior.list())}"
            )
        return value


class ProjectAlertResponse(ProjectAlertBase):
    id: str
    activation_behavior: str
    attribute_show_seconds: Optional[bool] = None
    created_at: datetime
    updated_at: datetime


class ProjectAlertsCountResponse(BaseModel):
    count: int
    urgent_count: int
    critical_count: int
    warning_count: int
    info_count: int


class ProjectAlertBackfillResponse(BaseModel):
    scanned_conversations: int
    matched_conversations: int
    created_active_hits: int
    updated_active_hits: int
    removed_active_hits: int
    skipped_read_conversations: int


class TriggeredAlertConversationResponse(BaseModel):
    conversation_id: str
    manager_name: Optional[str] = None
    manager_tags: List[str] = Field(default_factory=list)
    chat_name: Optional[str] = None
    chat_tags: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    total_score: int
    max_score: int
    highest_severity: str
    alert_status: str
    alerts: List[ConversationAlertResponse] = Field(default_factory=list)
    updated_at: datetime


class ConversationAlertsMarkReadRequest(BaseModel):
    conversation_alert_ids: List[str] = Field(default_factory=list)


class AlertStatusQueryResponse(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in {"active", "read", "all"}:
            raise ValueError("Invalid status. Must be active | read | all")
        return value


class ConversationEvaluationDetailResponse(BaseModel):
    conversation_id: str
    manager_name: str
    manager_tags: List[str] = Field(default_factory=list)
    chat_name: Optional[str] = None
    chat_tags: List[str] = Field(default_factory=list)
    loaded_from: Optional[datetime] = None
    loaded_to: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    evaluation_result: Optional[Dict[str, Any]]
    model_evaluation_result: Optional[Dict[str, Any]] = None
    alerts: List[ConversationAlertResponse] = Field(default_factory=list)
    attribute_results: List[ConversationAttributeResultResponse] = Field(default_factory=list)
    model_attribute_results: List[ConversationAttributeResultResponse] = Field(default_factory=list)
    evaluation_review: Optional[EvaluationReviewResponse] = None
    score_events: List[EvaluationScoreEventResponse] = Field(default_factory=list)
    reviewable_managers: List[ReviewableManagerResponse] = Field(default_factory=list)
    evaluation_approved: bool = False
    reviewed_at: Optional[datetime] = None
    reviewed_by_user_id: Optional[str] = None
    reviewed_by_name: Optional[str] = None
    approved_at: Optional[datetime] = None
    approved_by_user_id: Optional[str] = None
    approved_by_name: Optional[str] = None
    appeal: Optional[EvaluationAppealResponse] = None
    messages: List[ConversationMessage]

# Filter models
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page") 

# Flow models
class FlowStepCreate(BaseModel):
    step_type: str
    name: str
    instruction: str
    order: int
    node_data: Optional[Dict[str, Any]] = None
    classes: Optional[List[str]] = None  # For classification steps
    parent_node: Optional[str] = None  # Parent node ID for branching

class FlowStepUpdate(BaseModel):
    step_type: Optional[str] = None
    name: Optional[str] = None
    instruction: Optional[str] = None
    order: Optional[int] = None
    node_data: Optional[Dict[str, Any]] = None
    classes: Optional[List[str]] = None  # For classification steps
    parent_node: Optional[str] = None  # Parent node ID for branching

class FlowStepResponse(BaseModel):
    step_type: str
    name: str
    instruction: str
    order: int
    node_data: Optional[Dict[str, Any]] = None
    classes: Optional[List[str]] = None  # For classification steps
    parent_node: Optional[str] = None  # Parent node ID for branching

class FlowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_active: bool = True
    flow_data: Optional[Dict[str, Any]] = None
    steps: List[FlowStepCreate] = []

class FlowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    flow_data: Optional[Dict[str, Any]] = None
    steps: Optional[List[FlowStepCreate]] = None

class FlowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    is_active: bool
    flow_data: Optional[Dict[str, Any]] = None
    steps: List[FlowStepResponse] = []
    created_at: str
    updated_at: str

class FlowListResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str 

# Benchmark models
class BenchmarkGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None

class BenchmarkGroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class BenchmarkGroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    project_id: str
    conversation_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BenchmarkConversationCreate(BaseModel):
    name: str
    source_type: str  # "existing" or "custom"
    # For existing conversations
    existing_conversation_id: Optional[str] = None
    # For custom conversations  
    messages: Optional[List[Dict[str, Any]]] = None
    conversation_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None

class BenchmarkConversationUpdate(BaseModel):
    name: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    conversation_tags: Optional[List[str]] = None
    agent_tags: Optional[List[str]] = None

class BenchmarkConversationResponse(BaseModel):
    id: str
    benchmark_group_id: str
    name: str
    messages: List[Dict[str, Any]]
    conversation_tags: List[str] = Field(default_factory=list)
    agent_tags: List[str] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle database object and entity conversion."""
        if hasattr(obj, '__table__'):  # It's an SQLAlchemy model
            data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
            # Parse JSON string to Python list
            if data.get('messages') and isinstance(data['messages'], str):
                try:
                    data['messages'] = json.loads(data['messages'])
                except (json.JSONDecodeError, TypeError):
                    data['messages'] = []
            for tag_field in ('conversation_tags', 'agent_tags'):
                if data.get(tag_field) and isinstance(data[tag_field], str):
                    try:
                        parsed_tags = json.loads(data[tag_field])
                        data[tag_field] = parsed_tags if isinstance(parsed_tags, list) else []
                    except (json.JSONDecodeError, TypeError):
                        data[tag_field] = []
                elif not data.get(tag_field):
                    data[tag_field] = []
            return super().model_validate(data, *args, **kwargs)
        elif hasattr(obj, 'messages') and hasattr(obj, 'benchmark_group_id'):
            # It's a domain entity with BenchmarkMessage objects
            from dataclasses import asdict
            data = {
                'id': obj.id,
                'benchmark_group_id': obj.benchmark_group_id,
                'name': obj.name,
                'messages': [asdict(msg) for msg in obj.messages] if obj.messages else [],
                'conversation_tags': list(obj.conversation_tags or []),
                'agent_tags': list(obj.agent_tags or []),
                'created_at': obj.created_at
            }
            return super().model_validate(data, *args, **kwargs)
        return super().model_validate(obj, *args, **kwargs)

class BenchmarkCriteriaExpectationCreate(BaseModel):
    criteria_id: str
    expected_score: int
    comment: Optional[str] = None

class BenchmarkCriteriaExpectationUpdate(BaseModel):
    expected_score: Optional[int] = None
    comment: Optional[str] = None

class BenchmarkCriteriaExpectationResponse(BaseModel):
    id: str
    benchmark_conversation_id: str
    criteria_id: str
    expected_score: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class BenchmarkFlowStepExpectation(BaseModel):
    step_id: str
    step_name: str
    step_type: str  # "action_guidance" | "validation" | "classification"
    execution_expectation: str  # "not_expected" | "correct" | "incorrect" | "classify"
    expected_classes: Optional[List[str]] = None
    comment: Optional[str] = None

class BenchmarkFlowExpectationCreate(BaseModel):
    flow_id: str
    applies: bool
    step_expectations: Optional[List[BenchmarkFlowStepExpectation]] = None
    comment: Optional[str] = None

class BenchmarkFlowExpectationUpdate(BaseModel):
    applies: Optional[bool] = None
    step_expectations: Optional[List[BenchmarkFlowStepExpectation]] = None
    comment: Optional[str] = None

class BenchmarkFlowExpectationResponse(BaseModel):
    id: str
    benchmark_conversation_id: str
    flow_id: str
    applies: bool
    step_expectations: Optional[List[BenchmarkFlowStepExpectation]]
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle database object conversion."""
        if hasattr(obj, '__table__'):  # It's an SQLAlchemy model
            data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
            # Parse JSON string to Python list of dicts, then convert to BenchmarkFlowStepExpectation objects
            if data.get('step_expectations') and isinstance(data['step_expectations'], str):
                try:
                    step_expectations_data = json.loads(data['step_expectations'])
                    data['step_expectations'] = [BenchmarkFlowStepExpectation(**step) for step in step_expectations_data]
                except (json.JSONDecodeError, TypeError, ValidationError):
                    data['step_expectations'] = None
            return super().model_validate(data, *args, **kwargs)
        return super().model_validate(obj, *args, **kwargs)

class BenchmarkExpectationsResponse(BaseModel):
    criteria: List[BenchmarkCriteriaExpectationResponse]
    flows: List[BenchmarkFlowExpectationResponse]

class BenchmarkExecuteRequest(BaseModel):
    model: str
    evaluation_method: str
    
    @validator('evaluation_method')
    def validate_evaluation_method(cls, v):
        if not EvaluationMethod.contains(v):
            raise ValueError(f'Invalid evaluation_method value. Must be one of: {", ".join(EvaluationMethod.list())}')
        return v

class BenchmarkExecutionStatus(BaseModel):
    execution_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class BenchmarkAsyncExecutionResponse(BaseModel):
    execution_id: str
    status: str
    message: str

class BenchmarkGroupExecutionResponse(BaseModel):
    id: str
    benchmark_group_id: str
    results: Dict[str, Any]
    executed_at: datetime
    execution_duration_seconds: Optional[float]
    model: str
    evaluation_method: str

    class Config:
        from_attributes = True
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle entity conversion."""
        # It's a domain entity (BenchmarkGroupExecution dataclass)
        data = {
            'id': obj.id,
            'benchmark_group_id': obj.benchmark_group_id,
            'results': json.loads(obj.results) if isinstance(obj.results, str) else obj.results,
            'executed_at': obj.executed_at,
            'execution_duration_seconds': obj.execution_duration_seconds,
            'model': obj.model,
            'evaluation_method': obj.evaluation_method,
        }
        return super().model_validate(data, *args, **kwargs)

class BenchmarkExecutionResponse(BaseModel):
    id: str
    group_execution_id: str
    benchmark_conversation_id: str
    criteria_diff: Optional[Dict[str, Any]]  # {accuracy, differences, stats}
    flows_diff: Optional[Dict[str, Any]]  # {accuracy, differences}
    criteria_results: Optional[List[Dict[str, Any]]]
    execution_status: str
    error_message: Optional[str]
    executed_at: datetime
    model: str
    evaluation_method: str

    class Config:
        from_attributes = True
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle entity conversion using mapper."""
        data = format_benchmark_execution_for_api(obj)
        return super().model_validate(data, *args, **kwargs)

class BenchmarkGroupExecutionDetailResponse(BaseModel):
    execution: BenchmarkGroupExecutionResponse
    conversation_executions: List[BenchmarkExecutionResponse]


# =============================================================================
# Intercom OAuth Schemas
# =============================================================================

class IntercomAuthUrlResponse(BaseModel):
    """Response containing the Intercom OAuth authorization URL."""
    auth_url: str


class IntercomCallbackRequest(BaseModel):
    """Request for processing OAuth callback."""
    code: str
    state: str
    error: Optional[str] = None
    error_description: Optional[str] = None


class IntercomCallbackResponse(BaseModel):
    """Response after processing OAuth callback."""
    project_id: str
    success: bool
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    message: Optional[str] = None


class IntercomStatusResponse(BaseModel):
    """Response containing Intercom integration status."""
    is_connected: bool
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    status: str  # "connected", "disconnected", "reauth_required"


class IntercomDisconnectResponse(BaseModel):
    """Response after disconnecting Intercom."""
    success: bool
    message: str


# Helpshift integration models (API-key based)
class HelpshiftConnectRequest(BaseModel):
    """Request to connect Helpshift with an API key + account domain."""
    domain: str
    api_key: str


class HelpshiftStatusResponse(BaseModel):
    """Response containing Helpshift integration status."""
    is_connected: bool
    domain: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    status: str  # "connected", "disconnected", "reauth_required"


class HelpshiftValidateResponse(BaseModel):
    """Response after validating Helpshift credentials."""
    valid: bool
    message: str


class HelpshiftDisconnectResponse(BaseModel):
    """Response after disconnecting Helpshift."""
    success: bool
    message: str


class HelpshiftChatFilterOption(BaseModel):
    """A Helpshift issue exposed as a chat-tag filter option."""
    external_issue_id: str
    title: Optional[str] = None
    tags: List[str]


# Billing models
class BalanceDepositCreate(BaseModel):
    amount_usd: float = Field(gt=0)
    description: Optional[str] = None

class BalanceDepositResponse(BaseModel):
    id: str
    company_id: str
    amount_usd: float
    description: Optional[str]
    created_by_user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class DailyBalancePoint(BaseModel):
    date: str
    balance: float
    deposits: float
    charges: float


class PaginatedBalanceDepositsResponse(BaseModel):
    items: List[BalanceDepositResponse]
    total: int
    offset: int
    limit: int
    has_next: bool


class AggregatedChargeResponse(BaseModel):
    row_id: str
    type: Literal[
        "conversation",
        "playground",
        "benchmark",
        "message",
        "criterion_auto_improve",
        "criterion_test_preview",
        "general_comment_regeneration",
        "attribute_reapply",
        "manager_period_note_generation",
    ]
    target_id: str
    target_label: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    models: str
    price_usd: float
    is_free_evaluation: bool
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    last_evaluated_at: datetime


class PaginatedChargesResponse(BaseModel):
    items: List[AggregatedChargeResponse]
    total: int
    offset: int
    limit: int
    has_next: bool


class BillingOverviewResponse(BaseModel):
    balance_usd: float
    total_deposits_usd: float
    total_charges_usd: float
    deposits: PaginatedBalanceDepositsResponse
    charges: PaginatedChargesResponse
    daily_balance: List[DailyBalancePoint]


# ---------------------------------------------------------------------------
# One-off evaluation run batch schemas
# ---------------------------------------------------------------------------

class OneOffEvaluationRunBatchCreateRequest(BaseModel):
    chat_ids: List[str]
    source_type: Literal["telegram", "timelinesai"] = "telegram"
    window_from: datetime
    window_to: datetime
    scheduled_at: Optional[datetime] = None
    draft_id: Optional[str] = None


class OneOffEvaluationRunBatchUpdateRequest(BaseModel):
    chat_ids: List[str]
    source_type: Literal["telegram", "timelinesai"] = "telegram"
    window_from: datetime
    window_to: datetime
    scheduled_at: Optional[datetime] = None

class OneOffEvaluationRunBatchResponse(BaseModel):
    id: str
    project_id: str
    created_by_user_id: Optional[str] = None
    created_by_name: Optional[str] = None
    window_from: datetime
    window_to: datetime
    scheduled_at: Optional[datetime] = None
    status: str
    total_chat_count: int
    completed_chat_count: int
    failed_chat_count: int
    total_conversation_count: int = 0
    evaluated_conversation_count: int = 0
    created_at: datetime
    updated_at: datetime

class OneOffEvaluationRunBatchChatResponse(BaseModel):
    id: str
    batch_id: str
    source_type: Literal["telegram", "timelinesai"] = "telegram"
    telegram_chat_id: Optional[str] = None
    timelinesai_chat_id: Optional[str] = None
    chat_title: Optional[str] = None
    external_chat_id: Optional[str] = None
    source_account_name: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class OneOffEvaluationDraftUpsertRequest(BaseModel):
    chat_ids: List[str] = Field(default_factory=list)
    source_type: Literal["telegram", "timelinesai"] = "telegram"
    window_start_date: Optional[date] = None
    window_end_date: Optional[date] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    run_mode: Literal["now", "later"] = "now"
    scheduled_at: Optional[datetime] = None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def normalize_optional_time(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class OneOffEvaluationDraftResponse(BaseModel):
    id: str
    project_id: str
    created_by_user_id: Optional[str] = None
    created_by_name: Optional[str] = None
    chat_ids: List[str]
    source_type: Literal["telegram", "timelinesai"] = "telegram"
    window_start_date: Optional[date] = None
    window_end_date: Optional[date] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    run_mode: Literal["now", "later"]
    scheduled_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
