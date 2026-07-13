"""ARAG_PHARMA — ClinicalTrials.gov API """
import httpx
from datetime import datetime, timezone
from loguru import logger
from config.settings import settings


class ClinicalTrialsClient:
    BASE = settings.CLINICALTRIALS_API
    TIMEOUT = 20.0

    async def search(self, condition: str = "", intervention: str = "", max_results: int = 5) -> list[dict]:
        query = " AND ".join(filter(None, [condition, intervention]))
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
            "fields": ",".join([
                "NCTId", "BriefTitle", "Condition", "InterventionName",
                "Phase", "OverallStatus", "BriefSummary", "EligibilityCriteria",
                "StartDate", "PrimaryCompletionDate", "Sponsor",
            ]),
        }
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(f"{self.BASE}/studies", params=params)
                resp.raise_for_status()
                studies = resp.json().get("studies", [])
                return [self._parse(s) for s in studies]
        except Exception as e:
            logger.error(f"ClinicalTrials API error: {e}")
            return []

    def _parse(self, study: dict) -> dict:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        desc_mod = proto.get("descriptionModule", {})
        conditions = proto.get("conditionsModule", {}).get("conditions", [])
        interventions = [
            i.get("name", "") for i in
            proto.get("armsInterventionsModule", {}).get("interventions", [])[:3]
        ]
        phases = proto.get("designModule", {}).get("phases", [])
        nct_id = id_mod.get("nctId", "")
        content = (
            f"[ClinicalTrials.gov — {nct_id}]\n"
            f"Title: {id_mod.get('briefTitle', '')}\n"
            f"Status: {status_mod.get('overallStatus', '')} | Phase: {', '.join(phases) or 'N/A'}\n"
            f"Conditions: {', '.join(conditions[:3])}\n"
            f"Interventions: {', '.join(interventions)}\n"
            f"Sponsor: {proto.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', '')}\n"
            f"Summary: {desc_mod.get('briefSummary', '')[:600]}\n"
            f"Eligibility: {proto.get('eligibilityModule', {}).get('eligibilityCriteria', '')[:400]}"
        )
        return {
            "source": "ClinicalTrials.gov",
            "source_type": "clinical_trial",
            "content": content,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "nct_id": nct_id,
        }
