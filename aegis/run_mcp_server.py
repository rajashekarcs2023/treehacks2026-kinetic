"""
Standalone MCP SSE server for Poke integration.
Runs the AEGIS MCP tools on port 8001 with SSE transport.

Usage:
    python -m aegis.run_mcp_server

Then expose via ngrok:
    ngrok http 8001

Add the ngrok URL + /sse to Poke as a Custom Integration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.mcp_server import mcp

if __name__ == "__main__":
    print("[MCP] Starting standalone SSE server on port 8001...")
    print("[MCP] Poke URL will be: http://localhost:8001/sse")
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
