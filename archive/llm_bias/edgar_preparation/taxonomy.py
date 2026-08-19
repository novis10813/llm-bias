"""Versioned 8-K item titles and event-family mappings."""

from __future__ import annotations

TAXONOMY_VERSION = "sec-8k-items-v1"

MODERN_ITEMS: dict[str, tuple[str, str]] = {
    "1.01": ("Entry into a Material Definitive Agreement", "material_agreement"),
    "1.02": ("Termination of a Material Definitive Agreement", "material_agreement"),
    "1.03": ("Bankruptcy or Receivership", "bankruptcy"),
    "1.04": ("Mine Safety Reporting", "mine_safety"),
    "1.05": ("Material Cybersecurity Incidents", "cybersecurity"),
    "2.01": ("Completion of Acquisition or Disposition of Assets", "acquisition_disposition"),
    "2.02": ("Results of Operations and Financial Condition", "financial_results"),
    "2.03": ("Creation of a Direct Financial Obligation", "financing"),
    "2.04": ("Triggering Events That Accelerate a Direct Financial Obligation", "financing"),
    "2.05": ("Costs Associated with Exit or Disposal Activities", "restructuring"),
    "2.06": ("Material Impairments", "impairment"),
    "3.01": ("Notice of Delisting or Failure to Satisfy a Listing Rule", "listing_status"),
    "3.02": ("Unregistered Sales of Equity Securities", "equity_issuance"),
    "3.03": ("Material Modification to Rights of Security Holders", "security_holder_rights"),
    "4.01": ("Changes in Registrant's Certifying Accountant", "auditor_change"),
    "4.02": ("Non-Reliance on Previously Issued Financial Statements", "financial_restatement"),
    "5.01": ("Changes in Control of Registrant", "control_change"),
    "5.02": ("Departure or Appointment of Directors or Officers", "management_change"),
    "5.03": ("Amendments to Articles of Incorporation or Bylaws", "governance_change"),
    "5.04": ("Temporary Suspension of Trading Under Employee Benefit Plans", "benefit_plan_trading"),
    "5.05": ("Amendments to or Waiver of the Code of Ethics", "code_of_ethics"),
    "5.06": ("Change in Shell Company Status", "shell_company_status"),
    "5.07": ("Submission of Matters to a Vote of Security Holders", "shareholder_vote"),
    "5.08": ("Shareholder Director Nominations", "shareholder_nomination"),
    "6.01": ("ABS Informational and Computational Material", "asset_backed_securities"),
    "6.02": ("Change of Servicer or Trustee", "asset_backed_securities"),
    "6.03": ("Change in Credit Enhancement or External Support", "asset_backed_securities"),
    "6.04": ("Failure to Make a Required Distribution", "asset_backed_securities"),
    "6.05": ("Securities Act Updating Disclosure", "asset_backed_securities"),
    "7.01": ("Regulation FD Disclosure", "regulation_fd"),
    "8.01": ("Other Events", "other_event"),
    "9.01": ("Financial Statements and Exhibits", "supporting_material"),
}

LEGACY_ITEMS: dict[str, tuple[str, str]] = {
    "1": ("Changes in Control of Registrant", "control_change"),
    "2": ("Acquisition or Disposition of Assets", "acquisition_disposition"),
    "3": ("Bankruptcy or Receivership", "bankruptcy"),
    "4": ("Changes in Registrant's Certifying Accountant", "auditor_change"),
    "5": ("Other Events and Regulation FD Disclosure", "other_event"),
    "6": ("Resignations of Registrant's Directors", "management_change"),
    "7": ("Financial Statements and Exhibits", "supporting_material"),
    "8": ("Change in Fiscal Year", "fiscal_year_change"),
    "9": ("Regulation FD Disclosure", "regulation_fd"),
    "10": ("Amendments to or Waiver of the Code of Ethics", "code_of_ethics"),
    "11": ("Temporary Suspension of Trading Under Employee Benefit Plans", "benefit_plan_trading"),
    "12": ("Results of Operations and Financial Condition", "financial_results"),
}

SUPPORTING_ONLY_CODES = {"7", "9.01"}


def item_metadata(item_code: str) -> tuple[str, str, str]:
    """Return title, event family, and item schema for one extracted item code."""
    mapping = MODERN_ITEMS if "." in item_code else LEGACY_ITEMS
    schema = "modern" if "." in item_code else "legacy"
    title, family = mapping.get(item_code, ("Unknown 8-K Item", "unknown"))
    return title, family, schema
