from agent import Agent
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

debug = True
def main():
    print("============Starting============")

    name = "Eric"
    description = ("Eric is a charismatic and outgoing individual, known for his warm and approachable demeanor. He is "
                   "driven by ambition and a strong work ethic, inspiring those around him with his creative thinking "
                   "and innovative solutions. Empathetic and caring, he builds deep relationships while balancing "
                   "socializing with introspection. Overall, Eric's captivating and magnetic personality makes him a "
                   "popular and respected figure among his peers.")
    agent = Agent(name, description)

    user_name = input("Enter name: ")
    user_description = ""  # Potentially abstract "User" into own class and update description overtime

    prompt_meta = ('### Instruction: \n{}\n### Respond in a couple of sentences. Try to keep the conversation going. '
                   'Response:')

    try:
        while True:
            if debug: print("============Main Loop============")

            # Run agent
            user_input = input("Enter prompt for agent: ")
            response = agent.respond(prompt_meta, user_name, user_description, user_input)
            print(f"============Agent response============\n{response}\n\n")

            # Update memories
            agent.add_memory(user_name, user_input)
            if debug: print(f"============{agent.name} remembers============\n{agent.memories[-1]}\n\n")

            # Reflect on memories
            if agent.should_reflect():
                agent.reflect_on_memories(prompt_meta)
                if debug: print(f"============{agent.name} reflections============\n"
                                f"{' '.join([str(memory) for memory in agent.memories[-3:]])}\n\n")

    except KeyboardInterrupt:
        print("============Exiting============")


if __name__ == '__main__':
    main()
