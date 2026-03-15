import asyncio
import json
from src.echo_agent import extract_images_from_pdf_bytes, parse_metrics_with_gemini

async def test():
    with open("mock_echo_report_v2.pdf", "rb") as f:
        pdf_bytes = f.read()
    images = extract_images_from_pdf_bytes(pdf_bytes)
    metrics = parse_metrics_with_gemini(images)
    print(metrics.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test())
