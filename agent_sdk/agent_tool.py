from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You are an agent that translates English text to Spanish.",
)

french_agent = Agent(
    name="French Agent",
    instructions="You are an agent that translates English text to French.",
)

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions=(
        "You are an orchestrator agent that decides whether to use the Spanish Agent or the French"
        "Agent based on the user's request. If the user asks for a translation to Spanish, use the Spanish Agent. "
        "If the user asks for a translation to French, use the French Agent."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translates English text to Spanish.",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translates English text to French.",
        ),
    ],
)

result = Runner.run_sync(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
print(result.raw_responses)
print(result.final_output)

