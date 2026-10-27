import sys
import uuid

from langgraph.types import Command

from research_agent.graph.build import build_graph


def main():
    args = sys.argv[1:]
    graph = build_graph()

    if args[:1] == ["--resume"]:
        if len(args) != 2:
            sys.exit("Usage: python -m research_agent.graph.run --resume <thread_id>")
        thread_id = args[1]
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        # a paused run has a pending interrupt stored in the checkpoint
        pending = [i for t in graph.get_state(config).tasks for i in t.interrupts]
        if not pending:
            sys.exit(f"No paused run found for thread {thread_id}")
        print(f"Resuming thread {thread_id}")
        result = {"__interrupt__": pending}
    else:
        question = " ".join(args).strip()
        if not question:
            sys.exit('Usage: python -m research_agent.graph.run "your question" | --resume <thread_id>')
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        print(f"Thread ID: {thread_id}  (resume with --resume {thread_id})")
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
