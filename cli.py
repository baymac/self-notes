#!/usr/bin/env python3
"""
Self-Notes CLI - Query your Notion workspace with local AI
"""
import argparse
import sys
import textwrap
import warnings

# Suppress pydantic v1 compatibility warning (langchain internal)
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from src.indexer import index_all_pages, get_indexed_sources
from src.query import query


def format_answer(text: str, width: int = 80) -> str:
    """Wrap text for readable terminal output."""
    paragraphs = text.strip().split('\n\n')
    formatted = []
    for p in paragraphs:
        # Preserve list items and code blocks
        if p.strip().startswith(('-', '*', '1.', '2.', '3.', '```')):
            formatted.append(p)
        else:
            wrapped = textwrap.fill(p, width=width)
            formatted.append(wrapped)
    return '\n\n'.join(formatted)


def cmd_index(args):
    """Index all Notion pages."""
    print("\n" + "=" * 50)
    print("  INDEXING NOTION PAGES")
    print("=" * 50 + "\n")

    try:
        count = index_all_pages(verbose=True)
        print("\n" + "-" * 50)
        if count == 0:
            print("No pages found.")
            print("Make sure you've shared pages with your Notion integration.")
        else:
            print(f"Done. {count} chunks indexed.")
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


def cmd_ask(args):
    """Ask a question."""
    question = " ".join(args.question)
    if not question:
        print("Please provide a question")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"  Q: {question}")
    print("=" * 50 + "\n")

    result = query(question, verbose=False)

    print("-" * 50)
    print("  ANSWER")
    print("-" * 50 + "\n")

    print(format_answer(result['answer']))

    if result["sources"]:
        print("\n" + "-" * 50)
        print("  SOURCES")
        print("-" * 50 + "\n")
        for src in result["sources"]:
            print(f"  [{src['title']}]")
            if src["url"]:
                print(f"  {src['url']}")
            print()

    print("=" * 50 + "\n")


def cmd_sources(args):
    """List all indexed sources."""
    sources = get_indexed_sources()

    print("\n" + "=" * 50)
    print("  INDEXED PAGES")
    print("=" * 50 + "\n")

    if not sources:
        print("No pages indexed yet.")
        print("Run 'python cli.py index' first.\n")
        return

    print(f"Total: {len(sources)} pages\n")
    print("-" * 50 + "\n")

    for src in sources:
        print(f"  {src['title']}")
        if src["url"]:
            print(f"  {src['url']}")
        print()

    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query your Notion workspace with local AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # index command
    subparsers.add_parser("index", help="Sync and index Notion pages")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", nargs="+", help="Your question")

    # sources command
    subparsers.add_parser("sources", help="List indexed pages")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "sources":
        cmd_sources(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
