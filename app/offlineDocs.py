from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.openapi.utils import get_openapi
import json

def generate_offline_docs(app: FastAPI) -> HTMLResponse:
    openapi = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schemas = openapi.get("components", {}).get("schemas", {})

    def render_schema_example(schema_ref: str) -> str:
        if not schema_ref or not schema_ref.startswith("#/components/schemas/"):
            return "{}"
        name = schema_ref.split("/")[-1]
        if name == "MatchRequest":
            return json.dumps({
                "user_id": "string",
                "topic": "string",
                "urgency_level": "none"
            }, indent=2)
        elif name == "MatchResponse":
            return json.dumps({
                "user_id": "string",
                "matched_peers": [
                    {
                        "peer_id": "string",
                        "name": "string",
                        "college": "string",
                        "karma_in_topic": 0,
                        "match_score": 0,
                        "predicted_help_probability": 0,
                        "last_helped_on": "string",
                        "match_reason": ["string"]
                    }
                ],
                "status": "string"
            }, indent=2)
        elif name == "HTTPValidationError":
            return json.dumps({
                "detail": [
                    {
                        "loc": ["string", 0],
                        "msg": "string",
                        "type": "string"
                    }
                ]
            }, indent=2)
        return "{}"

    html = f"""
    <html>
    <head>
        <title>{app.title} – Offline Docs</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 2em; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code, pre {{ background-color: #f9f9f9; padding: 4px; border-radius: 4px; display: block; }}
        </style>
    </head>
    <body>
        <h1>{openapi['info']['title']} – Offline API Docs</h1>
        <p><strong>Version:</strong> {openapi['info']['version']}</p>
        <p>{openapi['info']['description']}</p>
        <h2>📘 Endpoints</h2>
    """

    for path, methods in openapi["paths"].items():
        for method, meta in methods.items():
            html += f"<h3>{method.upper()} <code>{path}</code></h3>"
            html += f"<p><strong>Description:</strong> {meta.get('description', '')}</p>"

            # Request body
            if "requestBody" in meta:
                html += "<h4>Request Body</h4>"
                for mime, content in meta["requestBody"]["content"].items():
                    schema = content.get("schema", {})
                    html += f"<p><strong>Media type:</strong> <code>{mime}</code></p>"
                    example_json = render_schema_example(schema.get("$ref", ""))
                    html += f"<pre>{example_json}</pre>"

            # Responses
            if "responses" in meta:
                html += "<h4>Responses</h4><table><tr><th>Code</th><th>Description</th></tr>"
                for code, resp in meta["responses"].items():
                    desc = resp.get("description", "")
                    html += f"<tr><td>{code}</td><td>{desc}</td></tr>"
                html += "</table>"

                # Examples
                for code, resp in meta["responses"].items():
                    for mime, content in resp.get("content", {}).items():
                        schema = content.get("schema", {})
                        example_json = render_schema_example(schema.get("$ref", ""))
                        if example_json != "{}":
                            html += f"<h3>Response – {code}</h4>"
                            html += f"<p><strong>Media type:</strong> <code>{mime}</code></p>"
                            html += f"<pre>{example_json}</pre>"
            html+= "<hr>"

    # Schemas
    html += "<h2>📦 Schemas</h2>"
    for name, schema in schemas.items():
        html += f"<h3>{name}</h3>"
        if "enum" in schema:
            html += f"<p><strong>Enum Values:</strong> {', '.join(schema['enum'])}</p>"
        if "properties" in schema:
            html += "<table><tr><th>Field</th><th>Type</th><th>Title</th></tr>"
            for prop, details in schema["properties"].items():
                dtype = details.get("type", "object")
                title = details.get("title", "")
                html += f"<tr><td>{prop}</td><td>{dtype}</td><td>{title}</td></tr>"
            html += "</table>"

    html += "</body></html>"
    return HTMLResponse(content=html)
