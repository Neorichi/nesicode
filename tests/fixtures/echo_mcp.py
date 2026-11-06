"""A tiny MCP server for tests: one tool that echoes and one that fails."""
from mcp.server.mcpserver import MCPServer as FastMCP

app = FastMCP("echo")


@app.tool()
def echo(text: str) -> str:
    """Repeat the text back."""
    return f"eco: {text}"


@app.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


if __name__ == "__main__":
    app.run()
