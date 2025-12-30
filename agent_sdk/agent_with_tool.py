from dotenv import load_dotenv

from agents import Agent, Runner
from agents import Any, WebSearchTool, function_tool
import requests


load_dotenv()

@function_tool()
def get_weather(city):
    """
    Fetches the current weather for the specified city

    Args:
        city (str): Name of the city to fetch weather for.
    Returns:
        str: Current weather description and temperature.
    """
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The current weather in {city} is: {response.text}"
    else:
        return "Sorry, I couldn't fetch the weather information right now."

# define a simple hello world agent
hello_agent = Agent(
    name="Hello World Agent",
    instructions="You are an agent which greets the user and helps them ans using emojis and in a funny way",
    tools=[WebSearchTool(), get_weather]
)

result = Runner.run_sync(hello_agent, "What is on towardsdata.ai website?")
print(result.final_output)

