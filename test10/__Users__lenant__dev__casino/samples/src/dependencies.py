import logging
import os
from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from src.agents.conversation_evaluator import ConversationEvaluator
from src.agents.flow_evaluator import FlowEvaluator
from src.api.gemini_caller import GeminiCaller
from src.api.llm_caller import LLMCaller
from src.api.openai_caller import OpenAICaller
from src.database.db import SessionLocal, WorkerSessionLocal
from src.database.models import JobRecord
from src.repositories.alert_repository import AlertRepository
from src.repositories.attribute_repository import AttributeRepository
from src.repositories.evaluation_tag_repository import EvaluationTagRepository
from src.repositories.billing_repository import BillingRepository
from src.repositories.company_repository import CompanyRepository
from src.repositories.conversation import (
    ConversationAnalyticsRepository,
    ConversationAttributionRepository,
    ConversationCoreRepository,
    ConversationExportRepository,
    ConversationFilterRepository,
    EvaluationAppealRepository,
    EvaluationReviewRepository,
    EvaluationScoreEventRepository,
    PlaygroundEvaluationRepository,
)
from src.repositories.evaluation_result_repository import EvaluationResultRepository
from src.repositories.evaluation_run_repository import EvaluationRunRepository
from src.repositories.google_sheets_integration_repository import GoogleSheetsIntegrationRepository
from src.repositories.helpshift_integration_repository import HelpshiftIntegrationRepository
from src.repositories.intercom_integration_repository import IntercomIntegrationRepository
from src.repositories.manager_period_note_repository import ManagerPeriodNoteRepository
from src.repositories.one_off_evaluation_repository import OneOffEvaluationRepository
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_template_repository import ProjectTemplateRepository
from src.repositories.telegram_repository import TelegramRepository
from src.repositories.timelinesai_repository import TimelinesAIRepository
from src.repositories.user_repository import UserRepository
from src.services.alert_service import AlertService
from src.services.attribute_extraction_service import AttributeExtractionService
from src.services.attribute_service import AttributeService
from src.services.evaluation_tag_service import EvaluationTagService
from src.services.billing_service import BillingService
from src.services.company_service import CompanyService
from src.services.conversation_import_service import ConversationImportService
from src.services.conversation_service import ConversationService
from src.services.criteria_service import CriteriaService
from src.services.criterion_evaluation_service import CriterionEvaluationService
from src.services.custom_dashboard_service import CustomDashboardService
from src.services.email_service import EmailService, GmailAPIGateway
from src.services.evaluation_appeal_service import EvaluationAppealService
from src.services.evaluation_result_service import EvaluationResultService
from src.services.evaluation_review_service import EvaluationReviewService
from src.services.evaluation_run_service import EvaluationRunService
from src.services.flow_execution_service import FlowExecutionService
from src.services.flow_service import FlowService
from src.services.google_sheets_export_scheduler import GoogleSheetsExportScheduler, GoogleSheetsExportServices
from src.services.google_sheets_service import GoogleSheetsService
from src.services.helpshift_service import HelpshiftService
from src.services.intercom_service import IntercomService
from src.services.knowledge_base_service import KnowledgeBaseService
from src.services.manager_period_note_service import ManagerPeriodNoteGenerationService, ManagerPeriodNoteService
from src.services.manager_service import ManagerService
from src.services.one_off_evaluation_service import OneOffEvaluationService
from src.services.pii_redactor_client import PiiRedactorClient
from src.services.project_service import ProjectService
from src.services.project_template_service import ProjectTemplateService
from src.services.quota_service import QuotaService
from src.services.signup_service import SignupService
from src.services.signup_token_service import SignupTokenService
from src.services.telegram_app_provider import TelegramAppProvider
from src.services.telegram import TelegramService
from src.services.timelinesai_service import TimelinesAIService
from src.services.token_service import TokenService
from src.services.zendesk_service import ZendeskService
from src.usecases.check_if_needs_evaluation_use_case import CheckIfNeedsEvaluationUseCase
from src.usecases.evaluate_conversation_use_case import EvaluateConversationUseCase
from src.usecases.evaluate_playground_use_case import EvaluatePlaygroundUseCase
from src.usecases.execute_benchmark_conversation_use_case import ExecuteBenchmarkConversationUseCase
from src.usecases.execute_benchmark_group_use_case import ExecuteBenchmarkGroupUseCase
from src.usecases.get_managers_evaluations_use_case import GetManagersEvaluationsUseCase
from src.usecases.load_helpshift_conversation_and_store_use_case import LoadHelpshiftConversationAndStoreUseCase
from src.usecases.load_intecom_conversation_and_store_use_case import LoadIntercomConversationAndStoreUseCase
from src.usecases.load_telegram_messages_use_case import LoadTelegramMessagesAndStoreUseCase
from src.usecases.load_timelinesai_chats_use_case import LoadTimelinesAIChatsAndStoreUseCase
from src.usecases.load_zendesk_tickets_and_store_use_case import LoadZendeskTicketsAndStoreUseCase
from src.usecases.reapply_post_evaluation_attribute_use_case import ReapplyPostEvaluationAttributeUseCase
from src.usecases.refresh_helpshift_agents_use_case import RefreshHelpshiftAgentsUseCase
from src.usecases.refresh_intercom_agents_use_case import RefreshIntercomAgentsUseCase

INTERCOM_API_URL = "https://api.intercom.io"
PII_REDACTOR_URL = os.getenv("PII_REDACTOR_URL")
PII_REDACTOR_TIMEOUT_SECONDS = float(os.getenv("PII_REDACTOR_TIMEOUT_SECONDS", "10"))
FLOWS_STORAGE_PATH = os.getenv("FLOWS_STORAGE_PATH", "storage/flows")
if not FLOWS_STORAGE_PATH:
    raise ValueError("FLOWS_STORAGE_PATH environment variable is required")

logger = logging.getLogger(__name__)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_knowledge_base_service(db: Session = Depends(get_db)) -> KnowledgeBaseService:
    return KnowledgeBaseService(db)

def get_criteria_service(db: Session = Depends(get_db)) -> CriteriaService:
    """Get criteria service instance."""
    return CriteriaService(db)

def get_flow_service() -> FlowService:
    """Get flow service instance."""
    return FlowService(storage_path=FLOWS_STORAGE_PATH)

def get_llm_caller() -> LLMCaller:
    """Get LLMCaller instance."""
    openai_caller = OpenAICaller()
    gemini_caller = GeminiCaller()
    return LLMCaller(openai_caller=openai_caller, gemini_caller=gemini_caller)

def get_conversation_evaluator(
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    llm_caller: LLMCaller = Depends(get_llm_caller)
) -> ConversationEvaluator:
    return ConversationEvaluator(
        knowledge_base_service=knowledge_base_service,
        criteria_service=criteria_service,
        llm_caller=llm_caller,
        verbose=True
    )


def get_attribute_repository(db: Session = Depends(get_db)) -> AttributeRepository:
    return AttributeRepository(db)


def get_attribute_service(
    db: Session = Depends(get_db),
    llm_caller: LLMCaller = Depends(get_llm_caller),
    evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
    attribute_repository: AttributeRepository = Depends(get_attribute_repository),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> AttributeService:
    extraction_service = AttributeExtractionService(
        db=db,
        llm_caller=llm_caller,
        evaluator=evaluator,
        knowledge_base_service=knowledge_base_service,
    )
    return AttributeService(
        attribute_repository=attribute_repository,
        attribute_extraction_service=extraction_service,
    )


def get_evaluation_tag_repository(db: Session = Depends(get_db)) -> EvaluationTagRepository:
    return EvaluationTagRepository(db)


def get_evaluation_tag_service(
    evaluation_tag_repository: EvaluationTagRepository = Depends(get_evaluation_tag_repository),
) -> EvaluationTagService:
    return EvaluationTagService(evaluation_tag_repository=evaluation_tag_repository)


def get_custom_dashboard_service(db: Session = Depends(get_db)) -> CustomDashboardService:
    return CustomDashboardService(db=db)

def get_conversation_repository(db: Session = Depends(get_db)) -> ConversationCoreRepository:
    return ConversationCoreRepository(db)


def get_conversation_attribution_repository(db: Session = Depends(get_db)) -> ConversationAttributionRepository:
    return ConversationAttributionRepository(db)


def get_conversation_filter_repository(db: Session = Depends(get_db)) -> ConversationFilterRepository:
    return ConversationFilterRepository(db)


def get_conversation_analytics_repository(db: Session = Depends(get_db)) -> ConversationAnalyticsRepository:
    return ConversationAnalyticsRepository(db)


def get_evaluation_review_repository(db: Session = Depends(get_db)) -> EvaluationReviewRepository:
    return EvaluationReviewRepository(db)


def get_evaluation_appeal_repository(db: Session = Depends(get_db)) -> EvaluationAppealRepository:
    return EvaluationAppealRepository(db)


def get_evaluation_score_event_repository(db: Session = Depends(get_db)) -> EvaluationScoreEventRepository:
    return EvaluationScoreEventRepository(db)


def get_playground_evaluation_repository(db: Session = Depends(get_db)) -> PlaygroundEvaluationRepository:
    return PlaygroundEvaluationRepository(db)


def get_conversation_export_repository(db: Session = Depends(get_db)) -> ConversationExportRepository:
    return ConversationExportRepository(db)


def get_evaluation_result_repository(db: Session = Depends(get_db)) -> EvaluationResultRepository:
    return EvaluationResultRepository(db)


def get_evaluation_result_service(
    evaluation_result_repository: EvaluationResultRepository = Depends(get_evaluation_result_repository),
) -> EvaluationResultService:
    return EvaluationResultService(evaluation_result_repository)


def get_alert_repository(db: Session = Depends(get_db)) -> AlertRepository:
    return AlertRepository(db)


def get_alert_service(
    alert_repository: AlertRepository = Depends(get_alert_repository),
    conversation_repository: ConversationFilterRepository = Depends(get_conversation_filter_repository),
    evaluation_review_repository: EvaluationReviewRepository = Depends(get_evaluation_review_repository),
    evaluation_result_repository: EvaluationResultRepository = Depends(get_evaluation_result_repository),
) -> AlertService:
    return AlertService(
        alert_repository=alert_repository,
        conversation_repository=conversation_repository,
        evaluation_review_repository=evaluation_review_repository,
        evaluation_result_repository=evaluation_result_repository,
    )


def get_pii_redactor_client() -> PiiRedactorClient:
    if not PII_REDACTOR_URL:
        raise ValueError("PII_REDACTOR_URL environment variable is required")
    return PiiRedactorClient(base_url=PII_REDACTOR_URL, timeout_seconds=PII_REDACTOR_TIMEOUT_SECONDS)

def get_token_service() -> TokenService:
    """Get token service instance."""
    return TokenService()

def get_project_repository(db: Session = Depends(get_db)) -> ProjectRepository:
    """Get project repository instance."""
    return ProjectRepository(db)

def get_intercom_integration_repository(db: Session = Depends(get_db)) -> IntercomIntegrationRepository:
    """Get intercom integration repository instance."""
    return IntercomIntegrationRepository(db)


def get_helpshift_integration_repository(db: Session = Depends(get_db)) -> HelpshiftIntegrationRepository:
    """Get Helpshift integration repository instance."""
    return HelpshiftIntegrationRepository(db)


def get_google_sheets_integration_repository(db: Session = Depends(get_db)) -> GoogleSheetsIntegrationRepository:
    """Get google sheets integration repository instance."""
    return GoogleSheetsIntegrationRepository(db)


def get_google_sheets_service() -> GoogleSheetsService:
    """Get google sheets service instance."""
    return GoogleSheetsService()


def get_project_service(
    project_repository: ProjectRepository = Depends(get_project_repository),
    conversation_repository: ConversationAttributionRepository = Depends(get_conversation_attribution_repository),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    token_service: TokenService = Depends(get_token_service),
    intercom_integration_repository: IntercomIntegrationRepository = Depends(get_intercom_integration_repository),
    google_sheets_integration_repository: GoogleSheetsIntegrationRepository = Depends(
        get_google_sheets_integration_repository
    ),
) -> ProjectService:
    """Get project service instance."""
    return ProjectService(
        project_repository,
        conversation_repository=conversation_repository,
        criteria_service=criteria_service,
        token_service=token_service,
        intercom_integration_repository=intercom_integration_repository,
        google_sheets_integration_repository=google_sheets_integration_repository,
    )

def get_flow_evaluator(llm_caller: LLMCaller = Depends(get_llm_caller)) -> FlowEvaluator:
    """Get flow evaluator instance."""
    return FlowEvaluator(llm_caller=llm_caller, verbose=True)

def get_flow_execution_service(
    flow_evaluator: FlowEvaluator = Depends(get_flow_evaluator),
    flow_service: FlowService = Depends(get_flow_service),
) -> FlowExecutionService:
    """Get flow execution service instance."""
    return FlowExecutionService(
        flow_evaluator=flow_evaluator,
        flow_service=flow_service,
        verbose=True
    )

# Quota service dependency (must be defined before it's referenced below)
def get_quota_service(db: Session = Depends(get_db)) -> QuotaService:
    return QuotaService(CompanyRepository(db), BillingRepository(db))

def get_evaluation_run_repository(db: Session = Depends(get_db)) -> EvaluationRunRepository:
    return EvaluationRunRepository(db)

def get_billing_repository(db: Session = Depends(get_db)) -> BillingRepository:
    return BillingRepository(db)

def get_billing_service(billing_repository: BillingRepository = Depends(get_billing_repository)) -> BillingService:
    return BillingService(billing_repository)

def get_evaluation_run_service(evaluation_run_repository: EvaluationRunRepository = Depends(get_evaluation_run_repository)) -> EvaluationRunService:
    """Get evaluation run service instance."""
    return EvaluationRunService(evaluation_run_repository)

def get_evaluation_review_service(
    conversation_repository: ConversationCoreRepository = Depends(get_conversation_repository),
    attribute_repository: AttributeRepository = Depends(get_attribute_repository),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    evaluation_review_repository: EvaluationReviewRepository = Depends(get_evaluation_review_repository),
    evaluation_score_event_repository: EvaluationScoreEventRepository = Depends(get_evaluation_score_event_repository),
    conversation_analytics_repository: ConversationAnalyticsRepository = Depends(get_conversation_analytics_repository),
    evaluation_result_repository: EvaluationResultRepository = Depends(get_evaluation_result_repository),
) -> EvaluationReviewService:
    return EvaluationReviewService(
        conversation_repository=conversation_repository,
        attribute_repository=attribute_repository,
        criteria_service=criteria_service,
        evaluation_review_repository=evaluation_review_repository,
        evaluation_score_event_repository=evaluation_score_event_repository,
        conversation_analytics_repository=conversation_analytics_repository,
        evaluation_result_repository=evaluation_result_repository,
    )

def get_evaluation_appeal_service(
    conversation_repository: ConversationCoreRepository = Depends(get_conversation_repository),
    evaluation_review_service: EvaluationReviewService = Depends(get_evaluation_review_service),
    evaluation_appeal_repository: EvaluationAppealRepository = Depends(get_evaluation_appeal_repository),
    evaluation_review_repository: EvaluationReviewRepository = Depends(get_evaluation_review_repository),
    evaluation_score_event_repository: EvaluationScoreEventRepository = Depends(get_evaluation_score_event_repository),
) -> EvaluationAppealService:
    return EvaluationAppealService(
        conversation_repository=conversation_repository,
        evaluation_review_service=evaluation_review_service,
        evaluation_appeal_repository=evaluation_appeal_repository,
        evaluation_review_repository=evaluation_review_repository,
        evaluation_score_event_repository=evaluation_score_event_repository,
    )

def get_conversation_import_service(
    conversation_repository: ConversationCoreRepository = Depends(get_conversation_repository),
    project_service: ProjectService = Depends(get_project_service),
    pii_redactor_client: PiiRedactorClient = Depends(get_pii_redactor_client),
) -> ConversationImportService:
    return ConversationImportService(
        conversation_repository=conversation_repository,
        project_service=project_service,
        pii_redactor_client=pii_redactor_client,
    )

def get_conversation_service(
    conversation_repository: ConversationCoreRepository = Depends(get_conversation_repository),
    project_service: ProjectService = Depends(get_project_service),
    quota_service: QuotaService = Depends(get_quota_service),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    evaluation_review_service: EvaluationReviewService = Depends(get_evaluation_review_service),
    evaluation_appeal_service: EvaluationAppealService = Depends(get_evaluation_appeal_service),
    conversation_import_service: ConversationImportService = Depends(get_conversation_import_service),
    attribute_service: AttributeService = Depends(get_attribute_service),
    conversation_attribution_repository: ConversationAttributionRepository = Depends(get_conversation_attribution_repository),
    conversation_filter_repository: ConversationFilterRepository = Depends(get_conversation_filter_repository),
    conversation_analytics_repository: ConversationAnalyticsRepository = Depends(get_conversation_analytics_repository),
    evaluation_review_repository: EvaluationReviewRepository = Depends(get_evaluation_review_repository),
    evaluation_score_event_repository: EvaluationScoreEventRepository = Depends(get_evaluation_score_event_repository),
    playground_repository: PlaygroundEvaluationRepository = Depends(get_playground_evaluation_repository),
    conversation_export_repository: ConversationExportRepository = Depends(get_conversation_export_repository),
    evaluation_result_repository: EvaluationResultRepository = Depends(get_evaluation_result_repository),
) -> ConversationService:
    """Get conversation service instance."""
    return ConversationService(
        conversation_repository=conversation_repository,
        project_service=project_service,
        quota_service=quota_service,
        evaluation_run_service=evaluation_run_service,
        evaluation_review_service=evaluation_review_service,
        evaluation_appeal_service=evaluation_appeal_service,
        conversation_import_service=conversation_import_service,
        attribute_service=attribute_service,
        conversation_attribution_repository=conversation_attribution_repository,
        conversation_filter_repository=conversation_filter_repository,
        conversation_analytics_repository=conversation_analytics_repository,
        evaluation_review_repository=evaluation_review_repository,
        evaluation_score_event_repository=evaluation_score_event_repository,
        playground_repository=playground_repository,
        conversation_export_repository=conversation_export_repository,
        evaluation_result_repository=evaluation_result_repository,
    )


def get_criterion_evaluation_service(
    conversation_repository: ConversationCoreRepository = Depends(get_conversation_repository),
    conversation_service: ConversationService = Depends(get_conversation_service),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
    llm_caller: LLMCaller = Depends(get_llm_caller),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    conversation_filter_repository: ConversationFilterRepository = Depends(get_conversation_filter_repository),
    evaluation_review_repository: EvaluationReviewRepository = Depends(get_evaluation_review_repository),
    evaluation_result_repository: EvaluationResultRepository = Depends(get_evaluation_result_repository),
) -> CriterionEvaluationService:
    return CriterionEvaluationService(
        conversation_repository=conversation_repository,
        conversation_service=conversation_service,
        criteria_service=criteria_service,
        knowledge_base_service=knowledge_base_service,
        evaluator=evaluator,
        llm_caller=llm_caller,
        evaluation_run_service=evaluation_run_service,
        conversation_filter_repository=conversation_filter_repository,
        evaluation_review_repository=evaluation_review_repository,
        evaluation_result_repository=evaluation_result_repository,
    )

def get_intercom_service(token_service: TokenService = Depends(get_token_service)) -> IntercomService:
    return IntercomService(api_url=INTERCOM_API_URL, token_service=token_service)

def get_zendesk_service(token_service: TokenService = Depends(get_token_service)) -> ZendeskService:
    return ZendeskService(token_service=token_service)

def get_check_if_needs_evaluation_use_case(
    project_service: ProjectService = Depends(get_project_service)) -> CheckIfNeedsEvaluationUseCase:
    return CheckIfNeedsEvaluationUseCase(project_service)

def get_manager_service(db: Session = Depends(get_db)) -> ManagerService:
    return ManagerService(db)

def get_manager_period_note_repository(db: Session = Depends(get_db)) -> ManagerPeriodNoteRepository:
    return ManagerPeriodNoteRepository(db)

def get_manager_period_note_service(
    manager_period_note_repository: ManagerPeriodNoteRepository = Depends(get_manager_period_note_repository),
    manager_service: ManagerService = Depends(get_manager_service),
) -> ManagerPeriodNoteService:
    return ManagerPeriodNoteService(
        manager_period_note_repository,
        manager_service,
    )

def get_manager_period_note_generation_service(
    manager_period_note_repository: ManagerPeriodNoteRepository = Depends(get_manager_period_note_repository),
    manager_service: ManagerService = Depends(get_manager_service),
    llm_caller: LLMCaller = Depends(get_llm_caller),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
) -> ManagerPeriodNoteGenerationService:
    return ManagerPeriodNoteGenerationService(
        manager_period_note_repository,
        manager_service,
        llm_caller=llm_caller,
        evaluation_run_service=evaluation_run_service,
    )

def get_one_off_evaluation_repository(db: Session = Depends(get_db)) -> OneOffEvaluationRepository:
    return OneOffEvaluationRepository(db)

def get_telegram_repository(db: Session = Depends(get_db)) -> TelegramRepository:
    """Get Telegram repository instance."""
    return TelegramRepository(db)

def get_timelinesai_repository(db: Session = Depends(get_db)) -> TimelinesAIRepository:
    """Get TimelinesAI repository instance."""
    return TimelinesAIRepository(db)

def get_telegram_service(
    telegram_repository: TelegramRepository = Depends(get_telegram_repository),
    token_service: TokenService = Depends(get_token_service)
) -> TelegramService:
    """Get Telegram service instance."""
    telegram_app_provider = TelegramAppProvider()
    return TelegramService(telegram_repository, token_service=token_service, telegram_app_provider=telegram_app_provider)

def get_timelinesai_service(
    timelinesai_repository: TimelinesAIRepository = Depends(get_timelinesai_repository),
    token_service: TokenService = Depends(get_token_service),
    manager_service: ManagerService = Depends(get_manager_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> TimelinesAIService:
    return TimelinesAIService(
        timelinesai_repository,
        token_service=token_service,
        manager_service=manager_service,
        conversation_service=conversation_service,
    )

def get_one_off_evaluation_service(
    one_off_evaluation_repository: OneOffEvaluationRepository = Depends(get_one_off_evaluation_repository),
    telegram_service: TelegramService = Depends(get_telegram_service),
    timelinesai_service: TimelinesAIService = Depends(get_timelinesai_service),
) -> OneOffEvaluationService:
    return OneOffEvaluationService(one_off_evaluation_repository, telegram_service, timelinesai_service)

def get_load_telegram_messages_use_case(
    telegram_service: TelegramService = Depends(get_telegram_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    manager_service: ManagerService = Depends(get_manager_service)
) -> LoadTelegramMessagesAndStoreUseCase:
    """Get load Telegram messages use case instance."""
    return LoadTelegramMessagesAndStoreUseCase(telegram_service, conversation_service, manager_service)

def get_load_timelinesai_chats_use_case(
    timelinesai_service: TimelinesAIService = Depends(get_timelinesai_service),
) -> LoadTimelinesAIChatsAndStoreUseCase:
    return LoadTimelinesAIChatsAndStoreUseCase(timelinesai_service)

def get_load_intercom_conversation_and_store_use_case(
    intercom_service: IntercomService = Depends(get_intercom_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    manager_service: ManagerService = Depends(get_manager_service),
    project_service: ProjectService = Depends(get_project_service)
) -> LoadIntercomConversationAndStoreUseCase:
    return LoadIntercomConversationAndStoreUseCase(intercom_service, conversation_service, manager_service, project_service)

def get_refresh_intercom_agents_use_case(
    intercom_service: IntercomService = Depends(get_intercom_service),
    manager_service: ManagerService = Depends(get_manager_service),
    project_service: ProjectService = Depends(get_project_service)
) -> RefreshIntercomAgentsUseCase:
    return RefreshIntercomAgentsUseCase(intercom_service, manager_service, project_service)

def get_helpshift_service(token_service: TokenService = Depends(get_token_service)) -> HelpshiftService:
    return HelpshiftService(token_service=token_service)

def get_load_helpshift_conversation_and_store_use_case(
    helpshift_service: HelpshiftService = Depends(get_helpshift_service),
    helpshift_repository: HelpshiftIntegrationRepository = Depends(get_helpshift_integration_repository),
    conversation_service: ConversationService = Depends(get_conversation_service),
    manager_service: ManagerService = Depends(get_manager_service),
    project_service: ProjectService = Depends(get_project_service),
) -> LoadHelpshiftConversationAndStoreUseCase:
    return LoadHelpshiftConversationAndStoreUseCase(
        helpshift_service, helpshift_repository, conversation_service, manager_service, project_service
    )

def get_refresh_helpshift_agents_use_case(
    helpshift_service: HelpshiftService = Depends(get_helpshift_service),
    helpshift_repository: HelpshiftIntegrationRepository = Depends(get_helpshift_integration_repository),
    manager_service: ManagerService = Depends(get_manager_service),
) -> RefreshHelpshiftAgentsUseCase:
    return RefreshHelpshiftAgentsUseCase(helpshift_service, helpshift_repository, manager_service)

def get_load_zendesk_tickets_and_store_use_case(
    zendesk_service: ZendeskService = Depends(get_zendesk_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    manager_service: ManagerService = Depends(get_manager_service),
    project_service: ProjectService = Depends(get_project_service)
) -> LoadZendeskTicketsAndStoreUseCase:
    return LoadZendeskTicketsAndStoreUseCase(zendesk_service, conversation_service, project_service, manager_service)

def get_managers_evaluations_use_case(
    manager_service: ManagerService = Depends(get_manager_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    project_service: ProjectService = Depends(get_project_service),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    attribute_service: AttributeService = Depends(get_attribute_service),
) -> GetManagersEvaluationsUseCase:
    """Get managers evaluations use case instance."""
    return GetManagersEvaluationsUseCase(
        manager_service,
        conversation_service,
        project_service,
        criteria_service,
        attribute_service,
    )

def get_benchmark_repository(db: Session = Depends(get_db)):
    """Get benchmark repository instance."""
    from src.repositories.benchmark_repository import BenchmarkRepository
    return BenchmarkRepository(db)

def get_benchmark_service(
    benchmark_repository = Depends(get_benchmark_repository),
    conversation_evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
    flow_execution_service: FlowExecutionService = Depends(get_flow_execution_service),
    criteria_service: CriteriaService = Depends(get_criteria_service),
    flow_service: FlowService = Depends(get_flow_service),
    project_service: ProjectService = Depends(get_project_service),
):
    """Get benchmark service instance."""
    from src.services.benchmark_service import BenchmarkService
    # Configure max concurrent executions based on environment
    max_concurrent_executions = int(os.environ.get('BENCHMARK_MAX_CONCURRENT_EXECUTIONS', '10'))

    return BenchmarkService(
        benchmark_repository=benchmark_repository,
        conversation_evaluator=conversation_evaluator,
        flow_execution_service=flow_execution_service,
        criteria_service=criteria_service,
        flow_service=flow_service,
        project_service=project_service,
        max_concurrent_executions=max_concurrent_executions,
    )

# New repository dependencies
def get_company_repository(db: Session = Depends(get_db)) -> CompanyRepository:
    """Get company repository instance."""
    return CompanyRepository(db)

def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    """Get user repository instance."""
    return UserRepository(db)

def get_project_template_repository(db: Session = Depends(get_db)) -> ProjectTemplateRepository:
    """Get project template repository instance."""
    return ProjectTemplateRepository(db)

# Email service dependency
def get_email_service() -> EmailService:
    """Get email service instance with Gmail API gateway."""
    email_from = os.getenv("EMAIL_FROM")
    base_url = os.getenv("FRONTEND_BASE_URL")
    gmail_service_account_file = os.getenv("GMAIL_SERVICE_ACCOUNT_FILE")
    gmail_subject_email = os.getenv("GMAIL_SUBJECT_EMAIL")

    # Require Gmail API configuration
    if not all([gmail_service_account_file, email_from, base_url, gmail_subject_email]):
        raise ValueError("Gmail API configuration is required. Please set GMAIL_SERVICE_ACCOUNT_FILE, EMAIL_FROM, FRONTEND_BASE_URL, and GMAIL_SUBJECT_EMAIL environment variables.")

    try:
        email_gateway = GmailAPIGateway(
            service_account_file=gmail_service_account_file,  # type: ignore
            from_email=email_from,  # type: ignore
            base_url=base_url,  # type: ignore
            subject_email=gmail_subject_email  # type: ignore
        )
        return EmailService(email_gateway)
    except Exception as e:
        raise ValueError(f"Failed to initialize Gmail API gateway: {e}")

# New service dependencies
def get_company_service(
    company_repo: CompanyRepository = Depends(get_company_repository),
    email_service: EmailService = Depends(get_email_service)
) -> CompanyService:
    """Get company service instance."""
    return CompanyService(company_repo, email_service)

def get_signup_service(
    user_repo: UserRepository = Depends(get_user_repository),
    token_service: TokenService = Depends(get_token_service),
    project_service: ProjectService = Depends(get_project_service),
    intercom_service: IntercomService = Depends(get_intercom_service),
    zendesk_service: ZendeskService = Depends(get_zendesk_service),
    manager_service: ManagerService = Depends(get_manager_service),
    email_service: EmailService = Depends(get_email_service),
    company_service: CompanyService = Depends(get_company_service)
) -> SignupService:
    """Get signup service instance."""
    return SignupService(
        user_repo=user_repo,
        token_service=token_service,
        project_service=project_service,
        intercom_service=intercom_service,
        zendesk_service=zendesk_service,
        manager_service=manager_service,
        email_service=email_service,
        company_service=company_service
    )

def get_signup_token_service() -> SignupTokenService:
    """Get signup token service instance."""
    return SignupTokenService()

def get_project_template_service(template_repo: ProjectTemplateRepository = Depends(get_project_template_repository)) -> ProjectTemplateService:
    """Get project template service instance."""
    return ProjectTemplateService(template_repo)

def get_evaluate_conversation_use_case(
    conversation_service: ConversationService = Depends(get_conversation_service),
    alert_service: AlertService = Depends(get_alert_service),
    attribute_service: AttributeService = Depends(get_attribute_service),
    evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
    flow_execution_service: FlowExecutionService = Depends(get_flow_execution_service),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    quota_service: QuotaService = Depends(get_quota_service)
) -> EvaluateConversationUseCase:
    return EvaluateConversationUseCase(
        conversation_service=conversation_service,
        evaluator=evaluator,
        flow_execution_service=flow_execution_service,
        evaluation_run_service=evaluation_run_service,
        quota_service=quota_service,
        alert_service=alert_service,
        attribute_service=attribute_service,
    )


def get_reapply_post_evaluation_attribute_use_case(
    conversation_service: ConversationService = Depends(get_conversation_service),
    project_service: ProjectService = Depends(get_project_service),
    attribute_service: AttributeService = Depends(get_attribute_service),
    alert_service: AlertService = Depends(get_alert_service),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    quota_service: QuotaService = Depends(get_quota_service),
    evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
) -> ReapplyPostEvaluationAttributeUseCase:
    return ReapplyPostEvaluationAttributeUseCase(
        conversation_service=conversation_service,
        project_service=project_service,
        attribute_service=attribute_service,
        alert_service=alert_service,
        evaluation_run_service=evaluation_run_service,
        quota_service=quota_service,
        evaluator=evaluator,
    )


def get_evaluate_playground_use_case(
    conversation_service: ConversationService = Depends(get_conversation_service),
    evaluator: ConversationEvaluator = Depends(get_conversation_evaluator),
    flow_execution_service: FlowExecutionService = Depends(get_flow_execution_service),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    quota_service: QuotaService = Depends(get_quota_service)
) -> EvaluatePlaygroundUseCase:
    return EvaluatePlaygroundUseCase(conversation_service, evaluator, flow_execution_service, evaluation_run_service, quota_service)

def get_execute_benchmark_group_use_case(
    benchmark_service = Depends(get_benchmark_service),
    benchmark_repository = Depends(get_benchmark_repository),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    quota_service: QuotaService = Depends(get_quota_service)
) -> ExecuteBenchmarkGroupUseCase:
    return ExecuteBenchmarkGroupUseCase(benchmark_service, benchmark_repository, evaluation_run_service, quota_service)

def get_execute_benchmark_conversation_use_case(
    benchmark_service = Depends(get_benchmark_service),
    benchmark_repository = Depends(get_benchmark_repository),
    evaluation_run_service: EvaluationRunService = Depends(get_evaluation_run_service),
    quota_service: QuotaService = Depends(get_quota_service)
) -> ExecuteBenchmarkConversationUseCase:
    return ExecuteBenchmarkConversationUseCase(benchmark_service, benchmark_repository, evaluation_run_service, quota_service)


def _create_benchmark_service_for_session(db: Session):
    """Build benchmark dependencies against an explicit session factory context."""
    from src.repositories.benchmark_repository import BenchmarkRepository
    from src.services.benchmark_service import BenchmarkService

    benchmark_repository = BenchmarkRepository(db)
    criteria_service = CriteriaService(db)
    knowledge_base_service = KnowledgeBaseService(db)
    flow_service = FlowService(storage_path=FLOWS_STORAGE_PATH)
    llm_caller = get_llm_caller()
    flow_evaluator = FlowEvaluator(llm_caller=llm_caller, verbose=True)
    flow_execution_service = FlowExecutionService(
        flow_evaluator=flow_evaluator,
        flow_service=flow_service,
        verbose=True,
    )
    project_service = ProjectService(
        ProjectRepository(db),
        conversation_repository=ConversationAttributionRepository(db),
        criteria_service=criteria_service,
        token_service=get_token_service(),
        intercom_integration_repository=IntercomIntegrationRepository(db),
        google_sheets_integration_repository=GoogleSheetsIntegrationRepository(db),
    )
    conversation_evaluator = ConversationEvaluator(
        knowledge_base_service=knowledge_base_service,
        criteria_service=criteria_service,
        llm_caller=llm_caller,
        verbose=True,
    )
    max_concurrent_executions = int(os.environ.get("BENCHMARK_MAX_CONCURRENT_EXECUTIONS", "10"))
    return BenchmarkService(
        benchmark_repository=benchmark_repository,
        conversation_evaluator=conversation_evaluator,
        flow_execution_service=flow_execution_service,
        criteria_service=criteria_service,
        flow_service=flow_service,
        project_service=project_service,
        max_concurrent_executions=max_concurrent_executions,
    )


def create_execute_benchmark_group_use_case_for_session(db: Session) -> ExecuteBenchmarkGroupUseCase:
    from src.repositories.benchmark_repository import BenchmarkRepository

    return ExecuteBenchmarkGroupUseCase(
        _create_benchmark_service_for_session(db),
        BenchmarkRepository(db),
        EvaluationRunService(EvaluationRunRepository(db)),
        QuotaService(CompanyRepository(db), BillingRepository(db)),
    )


def create_execute_benchmark_conversation_use_case_for_session(db: Session) -> ExecuteBenchmarkConversationUseCase:
    from src.repositories.benchmark_repository import BenchmarkRepository

    return ExecuteBenchmarkConversationUseCase(
        _create_benchmark_service_for_session(db),
        BenchmarkRepository(db),
        EvaluationRunService(EvaluationRunRepository(db)),
        QuotaService(CompanyRepository(db), BillingRepository(db)),
    )

# ---------------------------------------------------------------------------
# Background scheduler dependencies
# ---------------------------------------------------------------------------
from src.services.data_sync_queue_adder_scheduler import DataSyncQueueAdderScheduler
from src.services.evaluation_queue_adder_scheduler import EvaluationQueueAdderScheduler
from src.services.handlers.data_sync_job_handler import DataSyncJobHandler
from src.services.handlers.evaluation_job_handler import EvaluationJobHandler
from src.services.job_queue_processor import JobGroupConfig, JobQueueProcessor


def get_job_queue_processor(
    load_intercom_use_case: LoadIntercomConversationAndStoreUseCase = Depends(get_load_intercom_conversation_and_store_use_case),
    load_zendesk_use_case: LoadZendeskTicketsAndStoreUseCase = Depends(get_load_zendesk_tickets_and_store_use_case),
    load_telegram_use_case: LoadTelegramMessagesAndStoreUseCase = Depends(get_load_telegram_messages_use_case),
    load_timelinesai_use_case: LoadTimelinesAIChatsAndStoreUseCase = Depends(get_load_timelinesai_chats_use_case),
    load_helpshift_use_case: LoadHelpshiftConversationAndStoreUseCase = Depends(get_load_helpshift_conversation_and_store_use_case),
    evaluate_conversation_use_case: EvaluateConversationUseCase = Depends(get_evaluate_conversation_use_case)
) -> JobQueueProcessor:
    from src.constants import (
        DATA_SYNC_POLL_INTERVAL_SECONDS,
        EVAL_POLL_INTERVAL_SECONDS,
        MAX_PARALLEL_EVALUATIONS,
        MAX_PARALLEL_PROJECT_SYNCS,
        JobTaskType,
    )

    eval_handler = EvaluationJobHandler(evaluate_conversation_use_case)
    sync_handler = DataSyncJobHandler(
        load_intercom_use_case=load_intercom_use_case,
        load_zendesk_use_case=load_zendesk_use_case,
        load_telegram_use_case=load_telegram_use_case,
        load_timelinesai_use_case=load_timelinesai_use_case,
        load_helpshift_use_case=load_helpshift_use_case,
        one_off_session_factory=WorkerSessionLocal,
    )

    groups = [
        JobGroupConfig(
            name="evaluation",
            task_types=JobTaskType.evaluation_task_types(),
            max_workers_count=MAX_PARALLEL_EVALUATIONS,
            poll_interval_seconds=EVAL_POLL_INTERVAL_SECONDS,
            handler=eval_handler,
        ),
        JobGroupConfig(
            name="data_sync",
            task_types=JobTaskType.sync_types(),
            max_workers_count=MAX_PARALLEL_PROJECT_SYNCS,
            poll_interval_seconds=DATA_SYNC_POLL_INTERVAL_SECONDS,
            handler=sync_handler,
        ),
    ]

    return JobQueueProcessor(session_factory=WorkerSessionLocal, groups=groups, offload_sync_db_ops=True)

def get_evaluation_queue_adder_scheduler() -> EvaluationQueueAdderScheduler:
    """Provide an EvaluationQueueAdderScheduler instance."""
    return EvaluationQueueAdderScheduler(session_factory=WorkerSessionLocal)

def get_data_sync_queue_adder_scheduler() -> DataSyncQueueAdderScheduler:
    """Provide a DataSyncQueueAdderScheduler instance."""
    return DataSyncQueueAdderScheduler(session_factory=WorkerSessionLocal)

# Startup-specific scheduler factory
def create_all_schedulers_for_startup():
    """Create all background workers for startup events.
    """
    token_service = get_token_service()
    llm_caller = get_llm_caller()
    flow_service = get_flow_service()
    pii_redactor_client = get_pii_redactor_client()
    intercom_service = get_intercom_service(token_service)
    zendesk_service = get_zendesk_service(token_service)
    helpshift_service = get_helpshift_service(token_service)

    def resolve_session(session_or_factory) -> tuple[Session, bool]:
        if callable(session_or_factory):
            return session_or_factory(), True
        return session_or_factory, False

    def build_project_service(db: Session) -> ProjectService:
        project_repository = ProjectRepository(db)
        conversation_repository = ConversationAttributionRepository(db)
        intercom_integration_repository = IntercomIntegrationRepository(db)
        google_sheets_integration_repository = GoogleSheetsIntegrationRepository(db)
        return ProjectService(
            project_repository,
            conversation_repository=conversation_repository,
            criteria_service=CriteriaService(db),
            token_service=token_service,
            intercom_integration_repository=intercom_integration_repository,
            google_sheets_integration_repository=google_sheets_integration_repository,
        )

    def build_evaluation_runtime(db: Session):
        criteria_service = CriteriaService(db)
        knowledge_base_service = KnowledgeBaseService(db)
        flow_evaluator = FlowEvaluator(llm_caller=llm_caller, verbose=True)
        conversation_repository = ConversationCoreRepository(db)
        conversation_attribution_repository = ConversationAttributionRepository(db)
        conversation_filter_repository = ConversationFilterRepository(db)
        conversation_analytics_repository = ConversationAnalyticsRepository(db)
        evaluation_review_repository = EvaluationReviewRepository(db)
        evaluation_appeal_repository = EvaluationAppealRepository(db)
        evaluation_score_event_repository = EvaluationScoreEventRepository(db)
        playground_repository = PlaygroundEvaluationRepository(db)
        conversation_export_repository = ConversationExportRepository(db)
        evaluation_result_repository = EvaluationResultRepository(db)
        alert_repository = AlertRepository(db)
        alert_service = AlertService(
            alert_repository=alert_repository,
            conversation_repository=conversation_filter_repository,
            evaluation_review_repository=evaluation_review_repository,
            evaluation_result_repository=evaluation_result_repository,
        )
        project_service = build_project_service(db)
        flow_execution_service = FlowExecutionService(
            flow_evaluator=flow_evaluator,
            flow_service=flow_service,
            verbose=True,
        )
        quota_service = QuotaService(CompanyRepository(db), BillingRepository(db))
        evaluation_run_service = EvaluationRunService(EvaluationRunRepository(db))
        evaluator = ConversationEvaluator(
            knowledge_base_service=knowledge_base_service,
            criteria_service=criteria_service,
            llm_caller=llm_caller,
            verbose=True,
        )
        attribute_extraction_service = AttributeExtractionService(
            db=db,
            llm_caller=llm_caller,
            evaluator=evaluator,
            knowledge_base_service=knowledge_base_service,
        )
        attribute_service = AttributeService(
            attribute_repository=AttributeRepository(db),
            attribute_extraction_service=attribute_extraction_service,
        )
        evaluation_review_service = EvaluationReviewService(
            conversation_repository=conversation_repository,
            attribute_repository=AttributeRepository(db),
            criteria_service=criteria_service,
            evaluation_review_repository=evaluation_review_repository,
            evaluation_score_event_repository=evaluation_score_event_repository,
            conversation_analytics_repository=conversation_analytics_repository,
            evaluation_result_repository=evaluation_result_repository,
        )
        evaluation_appeal_service = EvaluationAppealService(
            conversation_repository=conversation_repository,
            evaluation_review_service=evaluation_review_service,
            evaluation_appeal_repository=evaluation_appeal_repository,
            evaluation_review_repository=evaluation_review_repository,
            evaluation_score_event_repository=evaluation_score_event_repository,
        )
        conversation_import_service = ConversationImportService(
            conversation_repository=conversation_repository,
            project_service=project_service,
            pii_redactor_client=pii_redactor_client,
        )
        conversation_service = ConversationService(
            conversation_repository=conversation_repository,
            project_service=project_service,
            quota_service=quota_service,
            evaluation_run_service=evaluation_run_service,
            evaluation_review_service=evaluation_review_service,
            evaluation_appeal_service=evaluation_appeal_service,
            conversation_import_service=conversation_import_service,
            attribute_service=attribute_service,
            conversation_attribution_repository=conversation_attribution_repository,
            conversation_filter_repository=conversation_filter_repository,
            conversation_analytics_repository=conversation_analytics_repository,
            evaluation_review_repository=evaluation_review_repository,
            evaluation_score_event_repository=evaluation_score_event_repository,
            playground_repository=playground_repository,
            conversation_export_repository=conversation_export_repository,
            evaluation_result_repository=evaluation_result_repository,
        )
        return (
            conversation_service,
            alert_service,
            attribute_service,
            evaluator,
            flow_execution_service,
            evaluation_run_service,
            quota_service,
        )

    def build_data_sync_runtime(db: Session):
        manager_service = ManagerService(db)
        project_service = build_project_service(db)
        conversation_service, _, _, _, _, _, _ = build_evaluation_runtime(db)
        telegram_repository = TelegramRepository(db)
        telegram_app_provider = TelegramAppProvider()
        telegram_service = TelegramService(
            telegram_repository,
            token_service=token_service,
            telegram_app_provider=telegram_app_provider,
        )
        timelinesai_service = TimelinesAIService(
            TimelinesAIRepository(db),
            token_service=token_service,
            manager_service=manager_service,
            conversation_service=conversation_service,
        )
        helpshift_repository = HelpshiftIntegrationRepository(db)
        return (
            conversation_service,
            manager_service,
            project_service,
            telegram_service,
            timelinesai_service,
            helpshift_repository,
        )

    # Create session-aware handlers bound to fresh per-job work sessions.
    class SessionAwareDataSyncJobHandler(DataSyncJobHandler):
        """Data sync handler using a per-job work session."""

        def __init__(self):
            self.one_off_session_factory = WorkerSessionLocal

        async def _run_with_data_sync_runtime(self, session_or_factory, runner) -> None:
            db, should_close = resolve_session(session_or_factory)
            try:
                runtime = build_data_sync_runtime(db)
                await runner(*runtime)
            finally:
                try:
                    db.rollback()
                except Exception:
                    logger.exception("SessionAwareDataSyncJobHandler: rollback failed")
                if should_close:
                    db.close()

        async def process(self, job: JobRecord, session_or_factory) -> None:
            if job.task_type == JobTaskType.SYNC_INTERCOM.value:
                project_id = self._preflight_intercom_project(job.target_id, session_or_factory)
                if project_id is None:
                    return

                async def run_intercom(
                    conversation_service,
                    manager_service,
                    project_service,
                    telegram_service,
                    timelinesai_service,
                    helpshift_repository,
                ):
                    logger.info("DataSyncJobHandler: Starting Intercom sync for project %s", project_id)
                    load_intercom_use_case = LoadIntercomConversationAndStoreUseCase(
                        intercom_service,
                        conversation_service,
                        manager_service,
                        project_service,
                    )
                    await load_intercom_use_case.execute(project_id)
                    logger.info("DataSyncJobHandler: Completed Intercom sync for project %s", project_id)

                await self._run_with_data_sync_runtime(session_or_factory, run_intercom)
                return

            if job.task_type == JobTaskType.SYNC_ZENDESK.value:
                project_id = self._preflight_zendesk_project(job.target_id, session_or_factory)
                if project_id is None:
                    return

                async def run_zendesk(
                    conversation_service,
                    manager_service,
                    project_service,
                    telegram_service,
                    timelinesai_service,
                    helpshift_repository,
                ):
                    logger.info("DataSyncJobHandler: Starting Zendesk sync for project %s", project_id)
                    load_zendesk_use_case = LoadZendeskTicketsAndStoreUseCase(
                        zendesk_service,
                        conversation_service,
                        project_service,
                        manager_service,
                    )
                    await load_zendesk_use_case.execute(project_id)
                    logger.info("DataSyncJobHandler: Completed Zendesk sync for project %s", project_id)

                await self._run_with_data_sync_runtime(session_or_factory, run_zendesk)
                return

            if job.task_type == JobTaskType.SYNC_TELEGRAM.value:
                async def run_telegram(
                    conversation_service,
                    manager_service,
                    project_service,
                    telegram_service,
                    timelinesai_service,
                    helpshift_repository,
                ):
                    load_telegram_use_case = LoadTelegramMessagesAndStoreUseCase(
                        telegram_service,
                        conversation_service,
                        manager_service,
                    )
                    await self._execute_telegram_sync_job(job, load_telegram_use_case)

                await self._run_with_data_sync_runtime(session_or_factory, run_telegram)
                return

            if job.task_type == JobTaskType.SYNC_TIMELINESAI.value:
                async def run_timelinesai(
                    conversation_service,
                    manager_service,
                    project_service,
                    telegram_service,
                    timelinesai_service,
                    helpshift_repository,
                ):
                    load_timelinesai_use_case = LoadTimelinesAIChatsAndStoreUseCase(timelinesai_service)
                    await self._execute_timelinesai_sync_job(job, load_timelinesai_use_case)

                await self._run_with_data_sync_runtime(session_or_factory, run_timelinesai)
                return

            if job.task_type == JobTaskType.SYNC_HELPSHIFT.value:
                project_id = self._preflight_helpshift_project(job.target_id, session_or_factory)
                if project_id is None:
                    return

                async def run_helpshift(
                    conversation_service,
                    manager_service,
                    project_service,
                    telegram_service,
                    timelinesai_service,
                    helpshift_repository,
                ):
                    logger.info("DataSyncJobHandler: Starting Helpshift sync for project %s", project_id)
                    load_helpshift_use_case = LoadHelpshiftConversationAndStoreUseCase(
                        helpshift_service,
                        helpshift_repository,
                        conversation_service,
                        manager_service,
                        project_service,
                    )
                    await load_helpshift_use_case.execute(project_id)
                    logger.info("DataSyncJobHandler: Completed Helpshift sync for project %s", project_id)

                await self._run_with_data_sync_runtime(session_or_factory, run_helpshift)
                return

            logger.error("DataSyncJobHandler: Unknown task type %s", job.task_type)
            raise ValueError(f"Unknown sync task type: {job.task_type}")

    class SessionAwareEvaluationJobHandler(EvaluationJobHandler):
        """Evaluation handler using a per-job work session."""

        def __init__(self):
            pass

        async def process(self, job: JobRecord, session_or_factory) -> None:
            from src.constants import JobTaskType
            from src.exceptions import NegativeBalanceLimitExceededError, QuotaExceededError

            db, should_close = resolve_session(session_or_factory)
            try:
                conversation_service, alert_service, attribute_service, evaluator, flow_execution_service, evaluation_run_service, quota_service = (
                    build_evaluation_runtime(db)
                )

                if job.task_type == JobTaskType.EVALUATE_CONVERSATION.value:
                    evaluate_conversation_use_case = get_evaluate_conversation_use_case(
                        conversation_service,
                        alert_service,
                        attribute_service,
                        evaluator,
                        flow_execution_service,
                        evaluation_run_service,
                        quota_service,
                    )
                    try:
                        await evaluate_conversation_use_case.execute(job.target_id)
                    except (NegativeBalanceLimitExceededError, QuotaExceededError):
                        logger.warning(
                            "EvaluationJobHandler: quota exceeded for conversation %s; dropping job",
                            job.target_id,
                        )
                    db.commit()
                    return

                logger.error("EvaluationJobHandler: Unknown task type %s", job.task_type)
                raise ValueError(f"Unknown evaluation task type: {job.task_type}")
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    logger.exception("SessionAwareEvaluationJobHandler: rollback failed")
                raise
            finally:
                if should_close:
                    db.close()

    # Create handlers with session-per-job pattern.
    eval_handler = SessionAwareEvaluationJobHandler()
    sync_handler = SessionAwareDataSyncJobHandler()

    from src.constants import (
        DATA_SYNC_POLL_INTERVAL_SECONDS,
        EVAL_POLL_INTERVAL_SECONDS,
        MAX_PARALLEL_EVALUATIONS,
        MAX_PARALLEL_PROJECT_SYNCS,
        JobTaskType,
    )

    groups = [
        JobGroupConfig(
            name="evaluation",
            task_types=JobTaskType.evaluation_task_types(),
            max_workers_count=MAX_PARALLEL_EVALUATIONS,
            poll_interval_seconds=EVAL_POLL_INTERVAL_SECONDS,
            handler=eval_handler,
        ),
        JobGroupConfig(
            name="data_sync",
            task_types=JobTaskType.sync_types(),
            max_workers_count=MAX_PARALLEL_PROJECT_SYNCS,
            poll_interval_seconds=DATA_SYNC_POLL_INTERVAL_SECONDS,
            handler=sync_handler,
        ),
    ]

    job_queue_processor = JobQueueProcessor(session_factory=WorkerSessionLocal, groups=groups, offload_sync_db_ops=True)

    eval_queue_adder_scheduler = EvaluationQueueAdderScheduler(session_factory=WorkerSessionLocal)
    data_sync_queue_adder_scheduler = DataSyncQueueAdderScheduler(session_factory=WorkerSessionLocal)

    def create_google_sheets_export_services() -> GoogleSheetsExportServices:
        db = WorkerSessionLocal()
        project_service = build_project_service(db)
        conversation_service, _, _, _, _, _, _ = build_evaluation_runtime(db)
        criteria_service = CriteriaService(db)

        return GoogleSheetsExportServices(
            project_service=project_service,
            conversation_service=conversation_service,
            criteria_service=criteria_service,
            cleanup=db.close,
        )

    google_sheets_export_scheduler = GoogleSheetsExportScheduler(
        services_factory=create_google_sheets_export_services,
        google_sheets_service=GoogleSheetsService(),
    )

    return (
        job_queue_processor,
        eval_queue_adder_scheduler,
        data_sync_queue_adder_scheduler,
        google_sheets_export_scheduler,
    )
