import asyncio
from fastmcp import Client

async def main():
    # Replace this with your server URL or transport
    # If your MCP server runs HTTP on localhost:9000 with path /mcp:
    server_url = "http://localhost:9000/mcp"

    client = Client(server_url)

    async with client:
        # Optional: Ping the server to test connection
        await client.ping()
        print("Connected to MCP server!")

        # List tools available on the server
        tools = await client.list_tools()
        print("Available tools:", tools)

        # Call a tool named "greet" with an argument
        result = await client.call_tool("remote_grep", {"query": "Tool calling code demo"})
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
