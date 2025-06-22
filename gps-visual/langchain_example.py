import requests
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SERVER_URL = "http://localhost:3000"

@tool
def search_nearby_places(location: str, query: str, radius: int = 5000) -> dict:
    """
    Searches for nearby places.
    Use this tool when you need to find places like 'coffee shops' or 'restaurants' near a specific location.
    'location' should be a "latitude,longitude" string.
    'query' is the type of place to search for, e.g., "pizza".
    'radius' is the search radius in meters, defaulting to 5000.
    """
    print(f"Searching for '{query}' near '{location}'...")
    response = requests.post(
        f"{SERVER_URL}/api/nearby-places",
        json={"location": location, "query": query, "radius": radius},
    )
    response.raise_for_status()
    return response.json()

@tool
def get_directions(origin: str, destination: str, destination_name: str = None) -> dict:
    """
    Gets walking directions between two points.
    'origin' and 'destination' should be "latitude,longitude" strings or addresses.
    'destination_name' is an optional friendly name for the destination.
    """
    print(f"Getting directions from '{origin}' to '{destination}'...")
    payload = {"origin": origin, "destination": destination}
    if destination_name:
        payload["destinationName"] = destination_name
    
    response = requests.post(
        f"{SERVER_URL}/api/directions",
        json=payload,
    )
    response.raise_for_status()
    return response.json()

@tool
def analyze_image(image_path: str, service: str = "gemini") -> dict:
    """
    Analyzes an image using either the 'gemini' or 'groq' service.
    'image_path' should be the local file path to the image.
    'service' can be 'gemini' (default) or 'groq'.
    """
    print(f"Analyzing '{image_path}' using '{service}'...")
    endpoint = "analyze-image" if service == "gemini" else "analyze-image-groq"
    
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(f"{SERVER_URL}/api/{endpoint}", files=files)
    
    response.raise_for_status()
    return response.json()



from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


llm = ChatAnthropic(temperature=0, model="claude-3-opus-20240229")

tools = [search_nearby_places, get_directions, analyze_image]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create an agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "Can you find a pizzeria near 40.7128,-74.0060?"})
agent_executor.invoke({"input": "What do you see in 2.png? Use the gemini service"})
