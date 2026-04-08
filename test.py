import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool

# Ensure the TAVILY_API_KEY environment variable is set
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# Initialize the tool
tavily_tool = TavilySearchTool(auto_install=False)

# Create an agent that uses the tool
researcher = Agent(
    role='Report Researcher',
    goal='Find the relavant reports and data on given topic.',
    backstory='An expert report researcher specializing in technology.',
    tools=[tavily_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description='Find brsr reports of tcs for 2023.',
    expected_output='PDF file links to the BRSR reports of TCS for the years 2023',
    agent=researcher
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

result = crew.kickoff()
print(result)