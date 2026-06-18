import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


async def fetch_company_details(phone: str) -> dict:
    base_url = os.getenv("AL_BACKEND_BASE_URL", "")
    api_key = os.getenv("AL_BACKEND_API_KEY", "")

    url = f"{base_url}/api/bot/fetchCompanyDetails"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"incoming_number": phone}, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            logger.info(f"Fetched company details for {phone}: firm={data.get('firm', {}).get('name')}")
            return data


async def fetch_case_type_questions(company_id: str, case_type_id: str) -> dict:
    base_url = os.getenv("AL_BACKEND_BASE_URL", "")
    api_key = os.getenv("AL_BACKEND_API_KEY", "")

    url = f"{base_url}/api/bot/fetchCaseTypeQuestions"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"company_id": company_id, "case_type_id": case_type_id},
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            logger.info(
                f"Fetched questions for case_type={data.get('case_type_slug')} "
                f"({len(data.get('questions', []))} questions, "
                f"{len(data.get('transfer_rules', []))} transfer rules)"
            )
            return data
