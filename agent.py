from utils.text_generation import generate, get_rating
from datetime import datetime
from memory import *
from utils.embeddings import *

debug = True


class Agent:
    def __init__(self, name, description, memory_reflection_threshold=20):
        self.name = name
        self.description = description
        self.memories = []
        self.history = []
        self.last_reflected_memory_index = 0
        self.memory_reflection_threshold = memory_reflection_threshold
        
    def __repr__(self):
        return f"Agent({self.name}, {self.description})"
    
    def respond(self, prompt_meta, user_name, user_description, user_input):
        recent_history_limit = 4
        now = datetime.now()

        prompt = f"You are {self.name}. It is currently {now}. You are interacting with {user_name}. "

        relevant_memory_string = ""
        for memory in get_relevant_memories(user_input, self.memories):
            relevant_memory_string += str(memory)

        prompt += f"Consider the following relevant memories: {relevant_memory_string}.\n"

        # prompt += f" You know the following about {user_name}: {user_description}"

        prompt += f"{user_name}: {user_input}\nResponse: "
        response = generate(prompt_meta.format(prompt))

        if debug: print(f"============Agent Prompt============\n{prompt}\n\n")

        self.history.append(f"{user_name}: {user_input}")
        self.history.append(f"{self.name}: {response}")
        return response
    
    def add_memory(self, user_name, action_result):
        embed = get_embedding(action_result)
        print(f"============Embedding is {len(embed)}============\n\n")
        print(f"============Embedding type is {type(embed[0])}============\n\n")
        self.memories.append(Memory(user_name, action_result, embed))
        self.memories[-1].importance = self.calculate_importance(self.memories[-1])

    def calculate_importance(self, memory) -> int:
        prompt = ("You are a memory importance AI. Given the character's profile and the memory description, "
                  "rate the importance of the memory on a scale of 1 to 10, where 1 is purely mundane (e.g., "
                  f"brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance). "
                  "Be sure to make your rating relative to the character's personality and concerns.\n\n"
                  f"Let's Begin!\n\n Name: {self.name}\nBio: {self.description}\nMemory:{memory.description}\n\n"
                  "Respond with a single number. Response: ")

        res = generate(prompt)
        rating = get_rating(res)
        max_attempts = 2
        current_attempt = 0
        while rating is None and current_attempt < max_attempts:
            rating = get_rating(res)
            current_attempt += 1
        if rating is None:
            rating = 0

        return rating

    def should_reflect(self) -> bool:
        memories_since_last_reflection = self.memories[self.last_reflected_memory_index:]
        cumulative_importance = sum(
            [memory.importance for memory in memories_since_last_reflection]
        )
        if debug: print(f"============cumulative_importance: {cumulative_importance}============")

        return cumulative_importance >= self.memory_reflection_threshold

    def reflect_on_memories(self, prompt_meta):
        # recent_memory_string = ' '.join(self.memories[-5:])
        memories_since_last_reflection = self.memories[self.last_reflected_memory_index:]
        memories_since_last_reflection_string = ' '.join(
            [str(memory) for memory in memories_since_last_reflection]
        )

        prompt = f"""\
Here are a list of statements:
{memories_since_last_reflection_string}

Given only the information above, what are 3 most salient high-level \
questions we can answer about the subjects in the statements?

Answer by splitting questions with |
Example: What does Joe like?|What does own?|What kind of person is Joe?
"""
        questions = generate(prompt_meta.format(prompt)).split('|')
        if debug: print(f"============{self.name} Reflection Questions============\n{'.'.join(questions)}\n\n")

        prompt = """\
Here are a list of statements:
{0}

What 5 high-level insights can you infer from the above statements?"""

        # responses = []
        for question in questions:
            relevant_memory_string = ""
            for memory in get_relevant_memories(question, self.memories):
                relevant_memory_string += str(memory)
                memory.update_last_accessed()

            response = (generate(prompt_meta.format(prompt.format(relevant_memory_string))))
            self.add_memory(self.name, response)

        self.last_reflected_memory_index = len(self.memories)
