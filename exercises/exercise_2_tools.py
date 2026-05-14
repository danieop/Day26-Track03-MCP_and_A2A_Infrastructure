"""Bai tap 2: Them tools va knowledge base."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc", "hop dong", "hợp đồng"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include expectation damages, consequential damages, specific performance, and cover "
            "damages. Statute of limitations is typically 4 years under UCC 2-725."
        ),
    },
    {
        "id": "vietnam_labor_law",
        "keywords": [
            "lao dong",
            "lao động",
            "sa thai",
            "sa thải",
            "hop dong lao dong",
            "hợp đồng lao động",
            "nguoi lao dong",
            "người lao động",
        ],
        "text": (
            "Theo Bo luat Lao dong Viet Nam, sa thai chi duoc ap dung trong cac truong hop "
            "luat dinh va phai tuan thu dung trinh tu, thu tuc. Nguoi lao dong co the yeu cau "
            "hoa giai vien lao dong, Hoi dong trong tai lao dong hoac Toa an giai quyet tranh chap."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tim kiem trong knowledge base phap ly."""
    query_lower = query.lower()
    matches = [
        f"[{entry['id']}] {entry['text']}"
        for entry in LEGAL_KNOWLEDGE
        if any(keyword in query_lower for keyword in entry["keywords"])
    ]
    return "\n\n".join(matches) if matches else "Khong tim thay thong tin lien quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiem tra thoi hieu khoi kien theo loai tranh chap."""
    case_type_lower = case_type.lower()
    if any(kw in case_type_lower for kw in ["contract", "hop dong", "hợp đồng"]):
        return (
            "Tranh chap hop dong thong thuong co thoi hieu khoi kien 3 nam ke tu ngay "
            "nguoi co quyen yeu cau biet hoac phai biet quyen, loi ich hop phap bi xam pham."
        )
    if any(kw in case_type_lower for kw in ["labor", "lao dong", "lao động", "sa thai", "sa thải"]):
        return (
            "Tranh chap lao dong ca nhan thuong co thoi hieu yeu cau Toa an giai quyet la "
            "1 nam ke tu ngay phat hien hanh vi ma moi ben cho rang quyen, loi ich hop phap "
            "cua minh bi vi pham."
        )
    if any(kw in case_type_lower for kw in ["tort", "boi thuong", "bồi thường", "damage"]):
        return (
            "Yeu cau boi thuong thiet hai ngoai hop dong thuong co thoi hieu khoi kien 3 nam "
            "ke tu ngay nguoi co quyen yeu cau biet hoac phai biet quyen, loi ich hop phap bi xam pham."
        )
    return "Khong xac dinh duoc loai vu viec. Hay neu ro tranh chap hop dong, lao dong hoac boi thuong."


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)

    question = "Thoi hieu khoi kien vu vi pham hop dong la bao lau?"

    messages = [
        SystemMessage(content="Ban la chuyen gia phap ly. Su dung tools de tra cuu thong tin."),
        HumanMessage(content=question),
    ]

    print(f"Cau hoi: {question}\n")

    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    tool_map = {tool.name: tool for tool in tools}
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Calling tool: {tool_call['name']}")
            tool_result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\nKet qua:\n{final_response.content}")
    else:
        print(f"\nKet qua:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
