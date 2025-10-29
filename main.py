#!/usr/bin/env python3
"""
CogniVerse Main Entry Point
Unified interface for all CogniVerse systems
"""

import sys
import json
from pathlib import Path
from CogniVerse_API_Bridge import get_cogniverse_bridge, CogniVerseRequest


def interactive_mode():
    """Interactive CLI mode"""
    print("🎭 CogniVerse Interactive Mode")
    print("=" * 50)

    bridge = get_cogniverse_bridge()

    # Show system status
    status = bridge.get_system_status()
    print("\n📊 System Status:")
    for system, available in status.items():
        if system != "timestamp":
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {system.replace('_', ' ').title()}")

    print("\nAvailable operations:")
    print("1. Generate narrative story")
    print("2. Route to experts")
    print("3. Full pipeline (route + generate)")
    print("4. System status")
    print("5. Exit")

    while True:
        try:
            choice = input("\nSelect operation (1-5): ").strip()

            if choice == "1":
                prompt = input("Enter story prompt: ").strip()
                genre = input("Genre (optional): ").strip() or None
                tone = input("Tone (optional): ").strip() or None

                request = CogniVerseRequest(
                    prompt=prompt, operation_type="narrative", genre=genre, tone=tone
                )

                response = bridge.process_request(request)

                if response.success:
                    print("\n📖 Generated Story:")
                    print("-" * 30)
                    print(response.result["story"])
                    print(f"\n📊 Metadata: {response.metadata}")
                else:
                    print(f"❌ Error: {response.error}")

            elif choice == "2":
                prompt = input("Enter text to route: ").strip()

                request = CogniVerseRequest(prompt=prompt, operation_type="routing")

                response = bridge.process_request(request)

                if response.success:
                    print("\n🎯 Expert Routing Results:")
                    print("-" * 30)
                    for expert, score in response.result["selected_experts"]:
                        print(f"  • {expert}: {score:.3f}")
                else:
                    print(f"❌ Error: {response.error}")

            elif choice == "3":
                prompt = input("Enter story prompt: ").strip()
                genre = input("Genre (optional): ").strip() or None
                tone = input("Tone (optional): ").strip() or None

                request = CogniVerseRequest(
                    prompt=prompt,
                    operation_type="full_pipeline",
                    genre=genre,
                    tone=tone,
                )

                response = bridge.process_request(request)

                if response.success:
                    print("\n🔄 Full Pipeline Results:")
                    print("-" * 30)

                    if "routing" in response.result:
                        print("🎯 Routing:")
                        for expert, score in response.result["routing"][
                            "selected_experts"
                        ]:
                            print(f"  • {expert}: {score:.3f}")

                    if "narrative" in response.result:
                        print("\n📖 Generated Story:")
                        print(response.result["narrative"]["story"])

                    print(
                        f"\n📊 Steps completed: {response.metadata['steps_completed']}"
                    )
                else:
                    print(f"❌ Error: {response.error}")

            elif choice == "4":
                status = bridge.get_system_status()
                print("\n📊 System Status:")
                for system, available in status.items():
                    if system != "timestamp":
                        status_icon = "✅" if available else "❌"
                        print(f"  {status_icon} {system.replace('_', ' ').title()}")

            elif choice == "5":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please select 1-5.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_test():
    """Quick test of all systems"""
    print("🧪 CogniVerse Quick Test")
    print("=" * 30)

    bridge = get_cogniverse_bridge()

    # Test routing
    print("Testing expert routing...")
    request = CogniVerseRequest(
        prompt="A detective investigates a mysterious crime", operation_type="routing"
    )
    response = bridge.process_request(request)

    if response.success:
        print("✅ Routing test passed")
        for expert, score in response.result["selected_experts"]:
            print(f"  • {expert}: {score:.3f}")
    else:
        print(f"❌ Routing test failed: {response.error}")

    # Test narrative generation
    print("\nTesting narrative generation...")
    request = CogniVerseRequest(
        prompt="A detective investigates a mysterious crime",
        operation_type="narrative",
        genre="mystery",
        tone="dramatic",
    )
    response = bridge.process_request(request)

    if response.success:
        print("✅ Narrative test passed")
        story = response.result["story"]
        print(f"  Generated {len(story)} characters")
        print(f"  Acts: {len(response.result['acts'])}")
    else:
        print(f"❌ Narrative test failed: {response.error}")

    print("\n🎉 Quick test completed!")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            print("Usage: python main.py [test|interactive]")
            print("  test       - Run quick system test")
            print("  interactive - Start interactive mode")
            print("  (no args)  - Start interactive mode")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
