from __future__ import annotations


class RecommendationAgent:
    def run(self, documents: list[dict]) -> list[str]:
        if not documents:
            return ["Generate anomaly reports before making recommendations."]

        actions = {
            "Verify insurance claim and payer details.",
            "Review physician notes and clinical justification.",
            "Check for duplicate billing or unusual charge bundling.",
            "Compare the case with department-level charge norms.",
        }
        for document in documents:
            metadata = document.get("metadata", {})
            reason = str(metadata.get("reason", "")).lower()
            if "no payment" in reason:
                actions.add("Follow up on missing payment or claim adjudication status.")
            if "denied" in reason or "rejected" in reason:
                actions.add("Audit denial reason codes and resubmission history.")
        return sorted(actions)
