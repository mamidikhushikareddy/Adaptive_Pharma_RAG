"""ARAG_PHARMA — FDA OpenFDA API Client"""
import httpx
from datetime import datetime, timezone
from loguru import logger
from config.settings import settings


class FDAClient:
    BASE = settings.FDA_API_BASE
    TIMEOUT = 15.0

    async def search_drug_label(self, drug_name: str, limit: int = 3) -> list[dict]:
        url = f"{self.BASE}/drug/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": limit,
        }
        return await self._get(url, params, "drug_label", drug_name)

    async def search_adverse_events(self, drug_name: str, limit: int = 5) -> list[dict]:
        url = f"{self.BASE}/drug/event.json"
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": limit,
        }
        return await self._get(url, params, "adverse_event", drug_name)

    async def _get(self, url: str, params: dict, source_type: str, drug: str) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                docs = []
                for r in results:
                    content = self._extract_content(r, source_type, drug)
                    if content:
                        docs.append({
                            "source": "FDA" if source_type == "drug_label" else "FAERS",
                            "source_type": source_type,
                            "content": content,
                            "url": f"https://api.fda.gov/drug/{source_type.replace('_','/')}.json",
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "drug": drug,
                        })
                return docs
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            logger.error(f"FDA API error: {e}")
            return []
        except Exception as e:
            logger.error(f"FDA client error: {e}")
            return []

    def _extract_content(self, result: dict, source_type: str, drug: str) -> str:
        if source_type == "drug_label":
            openfda = result.get("openfda", {})
            brand = openfda.get("brand_name", [drug])[0]
            generic = openfda.get("generic_name", [drug])[0]
            parts = [f"Drug: {brand} ({generic})\nSource: FDA Drug Label\n"]
            for label, key in [
                ("Indications", "indications_and_usage"),
                ("Warnings", "warnings"),
                ("Drug Interactions", "drug_interactions"),
                ("Dosage", "dosage_and_administration"),
                ("Contraindications", "contraindications"),
                ("Adverse Reactions", "adverse_reactions"),
            ]:
                val = result.get(key, [])
                if val:
                    parts.append(f"{label}:\n{val[0][:600]}\n")
            return "\n".join(parts)

        elif source_type == "adverse_event":
            patient = result.get("patient", {})
            drugs = [d.get("medicinalproduct", "") for d in patient.get("drug", [])[:3]]
            reactions = [r.get("reactionmeddrapt", "") for r in patient.get("reaction", [])[:5]]
            serious = result.get("serious", 0)
            return (
                f"FDA FAERS Adverse Event Report\n"
                f"Drugs: {', '.join(drugs)}\n"
                f"Reactions: {', '.join(reactions)}\n"
                f"Serious: {'Yes' if serious else 'No'}\n"
            )
        return ""
