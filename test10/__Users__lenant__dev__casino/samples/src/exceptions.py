class CompanyConflictError(Exception):
    """Raised when a user tries to join a company while already belonging to another company."""
    
    def __init__(self, current_company_name: str, invited_company_name: str, user_email: str):
        self.current_company_name = current_company_name
        self.invited_company_name = invited_company_name
        self.user_email = user_email
        super().__init__(
            f"User {user_email} is already a member of '{current_company_name}' "
            f"and cannot join '{invited_company_name}'. Users can only belong to one company at a time."
        ) 


class QuotaExceededError(Exception):
    """Raised when a company has no free evaluations remaining and is not paying."""

    def __init__(self, company_id: str):
        super().__init__(f"Quota exceeded for company {company_id}")


class NegativeBalanceLimitExceededError(Exception):
    """Raised when a paying company has reached the allowed negative balance floor."""

    def __init__(self, company_id: str, balance_usd: float, limit_usd: float):
        self.company_id = company_id
        self.balance_usd = balance_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Negative balance limit exceeded for company {company_id}: "
            f"balance {balance_usd:.4f} <= limit {limit_usd:.4f}"
        )


class PiiRedactorUnavailableError(Exception):
    """Raised when the external PII redactor service cannot be used."""

    def __init__(self, message: str = "PII redactor service unavailable"):
        super().__init__(message)
