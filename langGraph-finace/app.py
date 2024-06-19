"""구현에 사용할 라이브러리 불러오기"""

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates

from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

from langgraph.graph import END, Graph

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["POLYGON_API_KEY"] = os.getenv("POLYGON_API_KEY")

# 총3가지 단계로 구현을 진행합니다.
# 1. tools 생성
# 2. Agent 생성
# 3. graph 생성


# prompt 참조 링크 https://smith.langchain.com/hub/hwchase17/openai-functions-agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# 미국(us) 주식 정보를 가져오기위한 4가지 툴 정의
polygon = PolygonAPIWrapper()

tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon)
]

# Agent 생성
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent = RunnablePassthrough.assign(
    agent_outcome=agent_runnable
)

# 실행 툴 정의를 도와주는 함수


def execute_tools(data):
    agent_action = data.pop('agent_outcome')
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tool_to_use.invoke(agent_action.tool_input)
    data['intermediate_steps'].append((agent_action, observation))
    return data

# 조건 edge에 따라서 로직을 결정하여 사용하는 부분 정의


def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"


# LangGraph 정의하기
workflow = Graph()
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point('agent')
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)

workflow.add_edge("tools", "agent")
chain = workflow.compile()
result = chain.invoke(
    {"input": "what has been nvida closing", "intermediate_steps": []})
output = result["agent_outcome"].return_values["output"]
print(output)
