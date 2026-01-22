import re
from notion_client import Client
from typing import Generator
from src.config import NOTION_API_KEY, ROOT_PAGES


def get_client() -> Client:
    if not NOTION_API_KEY:
        raise ValueError("NOTION_API_KEY not set. Add it to your .env file.")
    return Client(auth=NOTION_API_KEY)


def parse_page_id(url_or_id: str) -> str:
    """Extract page ID from a Notion URL or return as-is if already an ID."""
    # Already a UUID-like ID
    if re.match(r'^[a-f0-9-]{32,36}$', url_or_id.replace('-', '')):
        return url_or_id.replace('-', '')

    # Extract from URL: notion.so/Page-Title-abc123 or notion.so/workspace/abc123
    match = re.search(r'([a-f0-9]{32})(?:\?|$)', url_or_id)
    if match:
        return match.group(1)

    # Try extracting the last segment after the last dash
    match = re.search(r'-([a-f0-9]{32})(?:\?|$)', url_or_id)
    if match:
        return match.group(1)

    # Last resort: take the last 32 hex chars
    hex_chars = re.findall(r'[a-f0-9]', url_or_id.lower())
    if len(hex_chars) >= 32:
        return ''.join(hex_chars[-32:])

    raise ValueError(f"Could not parse page ID from: {url_or_id}")


def extract_text_from_block(block: dict) -> str:
    """Extract plain text from a Notion block."""
    block_type = block.get("type")
    if not block_type:
        return ""

    block_data = block.get(block_type, {})

    # Handle rich text blocks
    if "rich_text" in block_data:
        return "".join(t.get("plain_text", "") for t in block_data["rich_text"])

    # Handle special block types
    if block_type == "child_page":
        return ""  # Don't include child page title in parent content
    if block_type == "child_database":
        return ""
    if block_type == "equation":
        return block_data.get("expression", "")
    if block_type == "code":
        code = "".join(t.get("plain_text", "") for t in block_data.get("rich_text", []))
        lang = block_data.get("language", "")
        return f"```{lang}\n{code}\n```"

    return ""


def get_page_content_and_children(client: Client, page_id: str) -> tuple[str, list[str]]:
    """
    Get text content from a page and collect child page IDs.
    Returns (content_text, list_of_child_page_ids)
    """
    blocks = []
    cursor = None

    while True:
        response = client.blocks.children.list(
            block_id=page_id,
            start_cursor=cursor
        )
        blocks.extend(response.get("results", []))

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    texts = []
    child_page_ids = []

    for block in blocks:
        block_type = block.get("type")

        # Collect child pages for recursive processing
        if block_type == "child_page":
            child_page_ids.append(block["id"])
            continue

        if block_type == "child_database":
            # Could expand this to fetch database entries
            continue

        text = extract_text_from_block(block)
        if text:
            texts.append(text)

        # Handle nested blocks (toggles, callouts, etc.) but not child_pages
        if block.get("has_children") and block_type not in ["child_page", "child_database"]:
            nested_content, nested_children = get_page_content_and_children(client, block["id"])
            if nested_content:
                texts.append(nested_content)
            child_page_ids.extend(nested_children)

    return "\n".join(texts), child_page_ids


def get_page_title(page: dict) -> str:
    """Extract title from a Notion page object."""
    props = page.get("properties", {})

    # Try common title property names
    for key in ["title", "Title", "Name", "name"]:
        if key in props:
            title_prop = props[key]
            if title_prop.get("type") == "title":
                title_items = title_prop.get("title", [])
                if title_items:
                    return "".join(t.get("plain_text", "") for t in title_items)

    # Fallback: check all properties for title type
    for prop in props.values():
        if prop.get("type") == "title":
            title_items = prop.get("title", [])
            if title_items:
                return "".join(t.get("plain_text", "") for t in title_items)

    return "Untitled"


def get_page_url(page: dict) -> str:
    """Get the Notion URL for a page."""
    return page.get("url", "")


def fetch_page_recursive(
    client: Client,
    page_id: str,
    visited: set[str]
) -> Generator[dict, None, None]:
    """Fetch a page and all its child pages recursively."""
    # Normalize ID
    normalized_id = page_id.replace('-', '')

    if normalized_id in visited:
        return
    visited.add(normalized_id)

    # Fetch page metadata
    try:
        page = client.pages.retrieve(page_id=page_id)
    except Exception as e:
        print(f"  Warning: Could not fetch page {page_id}: {e}")
        return

    title = get_page_title(page)
    url = get_page_url(page)
    last_edited = page.get("last_edited_time", "")

    # Get content and find child pages
    content, child_page_ids = get_page_content_and_children(client, page_id)

    if content.strip():
        yield {
            "id": page_id,
            "title": title,
            "url": url,
            "last_edited": last_edited,
            "content": content
        }

    # Recursively fetch child pages
    for child_id in child_page_ids:
        yield from fetch_page_recursive(client, child_id, visited)


def fetch_root_pages(client: Client) -> Generator[dict, None, None]:
    """Fetch all configured root pages and their children recursively."""
    if not ROOT_PAGES:
        raise ValueError(
            "No root pages configured. Add page URLs to ROOT_PAGES in src/config.py"
        )

    visited = set()

    for url_or_id in ROOT_PAGES:
        try:
            page_id = parse_page_id(url_or_id)
            yield from fetch_page_recursive(client, page_id, visited)
        except Exception as e:
            print(f"  Warning: Failed to process {url_or_id}: {e}")
