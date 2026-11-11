"""Predefined realistic Ham and Spam email examples for the Streamlit application.

All presets are rigorously validated against both:
- Pipeline B + LinearSVC (Production Champion, calibrated tau = 0.0040852377135018)
- Pipeline C + LinearSVC (High-Throughput Alternative, calibrated tau = 0.0876861391187918)
"""

from __future__ import annotations

from typing import Any, Dict


def get_email_examples() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of validated, realistic email presets.

    Presets included:
    1. ham_technical: Open-source Linux kernel patch review & memory leak fix.
    2. ham_scheduling: Corporate IT maintenance notification & sprint planning.
    3. spam_financial: High-urgency international beneficiary remittance scam.
    4. spam_phishing: Banking credential re-verification & security suspension alert.
    5. custom_clear: Blank template for arbitrary custom user testing.
    """
    return {
        "ham_technical": {
            "key": "ham_technical",
            "name": "Ham: Technical Discussion (Linux Kernel Patch)",
            "category": "Ham",
            "subject": "[PATCH v3 2/4] net: ethernet: fix skb memory leak in rx queue cleanup",
            "body": (
                "Hello netdev maintainers,\n\n"
                "During stress testing of the gigabit ethernet driver under high packet drop conditions, "
                "we observed a reproducible memory leak in the receive queue teardown sequence. Specifically, "
                "when rx_ring_cleanup() is invoked while packets remain buffered in the ring descriptor, "
                "dev_kfree_skb_any() was bypassed if the ring index was not aligned with the ring page boundary.\n\n"
                "This patch ensures all pending socket buffers are explicitly unmapped from DMA memory and freed "
                "before the ring descriptors are cleared.\n\n"
                "Tested on x86_64 with Linux 6.8-rc4 under continuous packet injection. No memory leak observed "
                "after 48 hours of saturation.\n\n"
                "Signed-off-by: Alan Foster <afoster@kernel-dev.org>\n"
                "Reviewed-by: Sarah Chen <schen@subsys-net.org>\n"
                "---\n"
                " drivers/net/ethernet/intel/e1000e/netdev.c | 14 +++++++++++---\n"
                " 1 file changed, 11 insertions(+), 3 deletions(-)"
            ),
            "description": "Technical kernel driver patch review discussing DMA socket buffers and memory leak remediation.",
        },
        "ham_scheduling": {
            "key": "ham_scheduling",
            "name": "Ham: Team Scheduling (Infrastructure Maintenance & Sprint Planning)",
            "category": "Ham",
            "subject": "Notice: Quarterly Database Infrastructure Maintenance and Sprint 14 Planning",
            "body": (
                "Hi Engineering Team,\n\n"
                "Please be advised that the central PostgreSQL cluster and replica instances will undergo "
                "scheduled kernel updates and security patch application this Saturday starting at 02:00 UTC. "
                "The maintenance window is anticipated to conclude within three hours.\n\n"
                "During this window, staging and read-replica environments will experience intermittent downtime. "
                "Production read traffic will fail over automatically to the hot standby node in us-east-2.\n\n"
                "Additionally, our bi-weekly Sprint 14 backlog refinement and planning session is scheduled for "
                "Monday at 10:00 AM EST. Please ensure that all Jira story estimation points and sprint pull requests "
                "are updated before the call.\n\n"
                "Calendar invite and agenda are attached to the team wiki page.\n\n"
                "Best regards,\n"
                "David Miller\n"
                "Principal Systems Administrator | Enterprise Infrastructure"
            ),
            "description": "Standard corporate notification regarding scheduled infrastructure maintenance and team sprint planning.",
        },
        "spam_financial": {
            "key": "spam_financial",
            "name": "Spam: Financial Scam (Overdue Beneficiary Wire Transfer)",
            "category": "Spam",
            "subject": "URGENT BENEFICIARY NOTIFICATION: APPROVED WIRE TRANSFER OF $4,850,000.00 USD",
            "body": (
                "ATTENTION: BENEFICIARY,\n\n"
                "FINAL NOTICE CONCERNING YOUR OVERDUE PAYMENT! We wish to notify you that the United Nations Financial "
                "Intelligence Unit in collaboration with the World Central Remittance Bureau has officially verified and "
                "approved the immediate release of your fund totaling $4,850,000.00 USD (Four Million Eight Hundred and Fifty "
                "Thousand Dollars).\n\n"
                "Your fund was previously held due to unpaid insurance and certificate fees, but all transfer clearances have now "
                "been granted. You are required to confirm your confidential wire transfer details immediately to the designated "
                "disbursement bank.\n\n"
                "REPLY WITH THE FOLLOWING INFORMATION URGENTLY:\n"
                "1. Full Legal Name:\n"
                "2. Current Direct Telephone/Mobile Number:\n"
                "3. Beneficiary Bank Account Name and Number:\n"
                "4. Country of Origin:\n"
                "5. Copy of Identification (Passport or Driver License):\n\n"
                "Failure to claim this fund within 48 hours will result in the immediate forfeiture and cancellation of your total "
                "fund transfer to the central treasury. Do not delay! Act now to receive your transfer payment!\n\n"
                "Yours Faithfully,\n"
                "Dr. Raymond Edwards\n"
                "Executive Director, Foreign Remittance Operations"
            ),
            "description": "Classic advance-fee 419 scam promising millions in funds with artificial urgency and credential solicitation.",
        },
        "spam_phishing": {
            "key": "spam_phishing",
            "name": "Spam: Phishing (Urgent Online Banking Profile Verification)",
            "category": "Spam",
            "subject": "URGENT SECURITY ALERT: Unauthorized Access Detected on Your Online Banking Account!",
            "body": (
                "URGENT SECURITY ALERT: Unauthorized Access Detected on Your Online Banking Account!\n\n"
                "Dear Valued Customer,\n\n"
                "We recently detected multiple unauthorized access attempts to your online banking and credit card profile. "
                "To ensure your account security and protect your financial assets, your access has been temporarily limited.\n\n"
                "You must immediately verify your account identity, billing address, and security credentials to restore full access. "
                "Failure to complete this security confirmation within 24 hours will result in permanent suspension of your account and debit card.\n\n"
                "CLICK HERE TO VERIFY YOUR ONLINE BANKING PROFILE NOW:\n"
                "http://online-banking-security-update.fake-auth.com/verify\n\n"
                "Please have your account number, PIN, and date of birth ready for instant identity confirmation.\n\n"
                "Thank you for your prompt attention to this urgent security matter.\n\n"
                "Fraud Prevention and Risk Management Department\n"
                "Online Banking Security Services"
            ),
            "description": "Malicious banking phishing attempt impersonating fraud prevention with credential harvesting links.",
        },
        "custom_clear": {
            "key": "custom_clear",
            "name": "Clear / Custom Input",
            "category": "Custom",
            "subject": "",
            "body": "",
            "description": "Clear all text fields to compose or paste an arbitrary custom email for real-time analysis.",
        },
    }

