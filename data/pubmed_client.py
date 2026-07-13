"""ARAG_PHARMA — PubMed NCBI API Client"""
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from loguru import logger
from config.settings import settings


class PubMedClient:
    BASE = settings.PUBMED_API_BASE
    EMAIL = settings.PUBMED_EMAIL
    TIMEOUT = 20.0

    async def search_and_fetch(self, query: str, max_results: int = 5) -> list[dict]:
        pmids = await self._search(query, max_results)
        if not pmids:
            return []
        return await self._fetch(pmids)

    async def _search(self, query: str, max_results: int) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(f"{self.BASE}/esearch.fcgi", params={
                    "db": "pubmed", "term": query, "retmax": max_results,
                    "retmode": "json", "sort": "relevance", "email": self.EMAIL,
                })
                resp.raise_for_status()
                return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    async def _fetch(self, pmids: list[str]) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(f"{self.BASE}/efetch.fcgi", params={
                    "db": "pubmed", "id": ",".join(pmids),
                    "retmode": "xml", "rettype": "abstract", "email": self.EMAIL,
                })
                resp.raise_for_status()
                return self._parse_xml(resp.text)
        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[dict]:
        docs = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    medline = article.find("MedlineCitation")
                    art = medline.find("Article")
                    title = getattr(art.find("ArticleTitle"), "text", "No title")
                    abstract = " ".join(
                        (el.text or "") for el in art.findall(".//AbstractText") if el.text
                    )
                    pmid = getattr(medline.find("PMID"), "text", "")
                    journal = getattr(art.find(".//Journal/Title"), "text", "Unknown Journal")
                    year = getattr(art.find(".//PubDate/Year"), "text", "Unknown Year")
                    authors = [
                        getattr(a.find("LastName"), "text", "")
                        for a in art.findall(".//Author")[:3]
                        if a.find("LastName") is not None
                    ]
                    if abstract:
                        content = (
                            f"[PubMed PMID:{pmid}]\n"
                            f"Title: {title}\n"
                            f"Journal: {journal} ({year})\n"
                            f"Authors: {', '.join(authors)}\n"
                            f"Abstract: {abstract[:1200]}"
                        )
                        docs.append({
                            "source": "PubMed",
                            "source_type": "literature",
                            "content": content,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "metadata": {"pmid": pmid, "year": year},
                        })
                except Exception:
                    continue
        except ET.ParseError as e:
            logger.error(f"PubMed XML parse error: {e}")
        return docs
