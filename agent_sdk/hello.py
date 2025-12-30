from dotenv import load_dotenv

from agents import Agent, Runner

load_dotenv()

# define a simple hello world agent
hello_agent = Agent(
    name="Hello World Agent",
    instructions="You are an agent which greets the user and helps them ans using emojis and in a funny way"
)

result = Runner.run_sync(hello_agent, "hey there, my name is Garg")
print(result.final_output)

