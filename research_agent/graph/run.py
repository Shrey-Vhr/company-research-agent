import sys
import uuid

from langgraph.types import Command

from research_agent.graph.build import build_graph


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: python -m research_agent.graph.run "your question"')

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}

    result = graph.invoke({"question": question, "evidence": []}, config)

    # invoke() returns early with "__interrupt__" when a node calls interrupt()
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n===== DRAFT =====\n")
        print(payload["draft"])
        if not payload["critic_passed"]:
            print("\n----- CRITIQUE -----")
            for item in payload["critique"]:
                print(f"- {item}")

        answer = input("\nApprove? (y / or type your requested changes) ").strip()
        if answer.lower() == "y":
            resume = {"action": "approve"}
        else:
            resume = {"action": "revise", "feedback": answer}
        result = graph.invoke(Command(resume=resume), config)

    print("\n===== FINAL REPORT =====\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
