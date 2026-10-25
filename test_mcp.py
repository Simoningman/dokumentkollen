import asyncio
from mcp import Client
from mcp_server import mcp

# Byt till din riktiga fil (högerklicka på filen → "Kopiera som sökväg")
PDF = r"C:\Users\Simon\Desktop\bilder\Årsredovisning Göteborg Energi AB 2024.pdf"


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Verktyg:", [t.name for t in tools.tools])

        r = await client.call_tool("load_document", {"path": PDF})
        print(r.content)

        r = await client.call_tool("search_documents", {"query": "omsättning", "top_k": 3})
        print(r.content)


asyncio.run(main())