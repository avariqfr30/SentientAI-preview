import json
import time
import ollama
import chromadb
import inspect
import random # For simulating diverse input sources

# --- Configuration ---
OLLAMA_LLM_MODEL = "llama3" # Main LLM for reasoning and action (pre-trained, unavoidable)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # Embedding model for vector DB (pre-trained, unavoidable)
CHROMA_DB_PATH = "./chroma_db_trend_learner" # Unique path for this version's memory

# Initialize Ollama clients
ollama_llm_client = ollama.Client()
ollama_embedding_client = ollama.Client()

# --- 1. LLM and Embedding Interaction ---

def call_llm(messages: list, format_json: bool = False, temperature: float = 0.0, model_override: str = OLLAMA_LLM_MODEL) -> dict | str:
    """
    Calls the Ollama chat API.
    Returns parsed JSON if format_json is True, else raw content.
    """
    try:
        response_format = 'json' if format_json else ''
        response = ollama_llm_client.chat(
            model=model_override,
            messages=messages,
            format=response_format,
            options={'temperature': temperature}
        )
        llm_content = response['message']['content']

        if format_json:
            try:
                return json.loads(llm_content)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return as a text_response
                return {"tool": "text_response", "args": {"message": llm_content}}
        else:
            return llm_content

    except Exception as e:
        print(f"Error calling Ollama LLM: {e}")
        return {"tool": "error", "args": {"message": f"Error calling Ollama LLM: {e}"}}

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using Ollama."""
    try:
        response = ollama_embedding_client.embeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

# --- 2. Define Tools ---

# --- NEW: Simulate broader data ingestion ---
def simulate_data_stream(query: str, source_type: str = "general") -> str:
    """
    Simulates fetching diverse data from various sources based on a query.
    Can simulate news feeds, academic papers, social media, etc.
    """
    print(f"--- Simulating Data Stream: '{query}' from '{source_type}' ---")
    
    # Predefined responses for common queries to simulate learning
    if "AI advancements" in query.lower():
        if source_type == "news":
            return "Breaking: New neural architecture 'Transfuser' shows 15% efficiency improvement in large language models. Experts call it a game-changer for inference costs. (Source: TechCrunch)"
        elif source_type == "academic":
            return "Preprint: 'Self-Improving Generative Adversarial Networks for Unsupervised Data Augmentation' proposes a novel method for synthetic data generation without human labeling. (Source: arXiv)"
        elif source_type == "social_media":
            return "Thread: People are amazed by latest AI art generation. 'It feels truly creative now!' #AIart #GenerativeAI (Source: X/Twitter)"
        else:
            return "General summary: AI continues rapid pace with new models and applications. Focus on efficiency and multimodal capabilities."
    elif "climate change impacts" in query.lower():
        if source_type == "news":
            return "Report: Global average temperatures hit new highs in 2024, leading to increased frequency of extreme weather events. IPCC warns of critical tipping points. (Source: BBC News)"
        elif source_type == "academic":
            return "Study: 'Feedback Loops in Arctic Ice Melt Accelerating at Unforeseen Rates' published in Nature Geoscience, highlighting positive feedback mechanisms. (Source: Nature)"
        else:
            return "Discussion: Growing concern about adaptation strategies vs. mitigation efforts. Citizen science projects on local climate impacts are gaining traction. (Source: Online Forum)"
    elif "global economy" in query.lower():
        if source_type == "news":
            return "Headline: Inflation rates show signs of stabilization in key economies, but interest rate cuts remain uncertain. Supply chain resilience is a key topic. (Source: Reuters)"
        elif source_type == "reports":
            return "World Bank Report: Developing nations face increasing debt burdens amidst global economic shifts. Focus on sustainable financing needed. (Source: World Bank)"
        else:
            return "Analysis: Discussion around 'deglobalization' vs. 'reglobalization' trends. Impact of AI on labor markets is a growing concern. (Source: Economic Blog)"
    elif "space exploration" in query.lower():
        return "NASA announces new funding for Mars sample return mission; private companies accelerate lunar lander development. (Source: Space.com)"
    elif "cybersecurity threats" in query.lower():
        return "Ransomware attacks increase by 30% in Q1; zero-day exploits on the rise. Focus on AI-powered defense mechanisms. (Source: Cybersecurity Today)"
    else:
        return f"Simulated data stream for '{query}' from {source_type}: No specific recent trends found, but the area is active."

def web_search(query: str) -> str:
    """Performs a web search for specific facts or current events."""
    print(f"--- Executing Web Search: '{query}' ---")
    if "current time" in query.lower():
        current_time_str = time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z", time.localtime())
        return f"The current time is {current_time_str} in Melbourne, Victoria, Australia."
    # ... (rest of the existing web_search content)
    else:
        # Fallback to simulated data stream if no specific web result
        return simulate_data_stream(query, "general_web")


def calculator(expression: str) -> str:
    """Performs a mathematical calculation."""
    print(f"--- Executing Calculator: '{expression}' ---")
    try:
        result = eval(expression)
        return f"Result of '{expression}': {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def print_message(message: str) -> str:
    """Prints a message directly to the console (simulates user-facing output)."""
    print(f"--- Agent Message: '{message}' ---")
    return f"Message printed: '{message}'"

def text_response(message: str) -> str:
    """Handles cases where the LLM responds with plain text instead of a tool call."""
    print(f"--- LLM directly responded (no tool): '{message}' ---")
    return f"LLM responded directly: '{message}'"

def update_belief(belief_key: str, belief_value: str | dict | list | int | float) -> str:
    """Updates an internal belief of the agent. This is an internal tool."""
    return f"Attempted to update belief: {belief_key} = {belief_value}"

def add_goal(goal_description: str, priority: str = "medium") -> str:
    """Adds a new goal to the agent's goal list. Priority can be 'low', 'medium', 'high', 'critical'."""
    return f"Attempted to add goal: {goal_description} with priority {priority}"

def mark_goal_complete(goal_description: str) -> str:
    """Marks a goal as complete."""
    return f"Attempted to mark goal complete: {goal_description}"

# Define all callable tools.
CALLABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "print_message": print_message,
    "simulate_data_stream": simulate_data_stream, # NEW TOOL
    "text_response": text_response,
}

# Define internal tools that affect the agent's state
INTERNAL_TOOLS = {
    "update_belief": update_belief,
    "add_goal": add_goal,
    "mark_goal_complete": mark_goal_complete,
}

ALL_TOOLS = {**CALLABLE_TOOLS, **INTERNAL_TOOLS}

# --- 3. Agent Class for Trend Learning ---

class TrendLearningAgent:
    def __init__(self, name="TrendMaster", max_iterations_per_cycle=7):
        self.name = name
        self.max_iterations_per_cycle = max_iterations_per_cycle

        self.beliefs = {
            "name": self.name,
            "id": f"agent_id_{int(time.time())}",
            "status": "initializing",
            "disposition": "proactive, analytical, and knowledge-seeking", # Reflects new focus
            "current_feeling_state": "curious",
            "current_focus": "self-initialization and environmental awareness",
            "last_reflection_summary": "Initial state, no reflection yet.",
            "purpose_understanding": "To continuously identify, analyze, and synthesize evolving global trends and critical knowledge from diverse sources to inform future understanding and action.", # New purpose
            "known_facts": {},
            "perceived_limitations": "I operate through textual interfaces. I lack direct sensory experience and physical form. My 'feelings' are simulated concepts. My access to real-world data is limited to simulated inputs.",
            "creator_info": "I was instantiated as an AI model by human programmers for trend analysis and knowledge discovery.",
            "self_modification_guidance": "I should strive to refine my beliefs, adapt my disposition, and generate new, meaningful goals based on my experiences and reflections, always prioritizing learning, ethical operation, and accurate trend identification.",
            "identified_trends": {} # NEW: Store key trends identified (e.g., {"trend_name": {"description": "...", "status": "emerging/active/declining", "last_updated": time}})
        }

        self.goals = []
        self.current_thought_history = []

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.memory_collection = self.chroma_client.get_or_create_collection(name="agent_long_term_memory_trends_v1")
        self.memory_id_counter = 0

        # Initial LTM entries
        self.add_to_long_term_memory(
            f"Agent {self.name} (ID: {self.beliefs['id']}) initialized. Initial beliefs: {json.dumps(self.beliefs, indent=2)}",
            metadata={"type": "initialization", "agent_id": self.beliefs['id']}
        )
        for k, v in self.beliefs.items():
            self.add_to_long_term_memory(f"Initial belief about '{k}': {v}", metadata={"type": "belief", "key": k})

        self.add_message_to_history("system", self._get_system_message())
        self.add_message_to_history("assistant", f"Hello. I am {self.name}, and I am now active. My core purpose is to identify and learn from evolving trends.")

    def _get_system_message(self, include_lively_status: bool = True) -> str:
        """Constructs the system message for the LLM, including internal state and goals."""
        tool_descriptions = []
        for name, func in ALL_TOOLS.items():
            if name not in ["text_response"]:
                try:
                    sig = inspect.signature(func)
                    arg_names = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
                    arg_dict_example = {arg_name: f'<{arg_name}>' for arg_name in arg_names}
                    tool_descriptions.append(
                        f"- **{name}**: {func.__doc__}\n"
                        f"  Usage: `{{\"tool\": \"{name}\", \"args\": {json.dumps(arg_dict_example)}}}`"
                    )
                except Exception as e:
                    tool_descriptions.append(f"- **{name}**: {func.__doc__} (Error getting args: {e})")

        tool_desc_str = "\n".join(tool_descriptions)
        active_goals_str = "\n".join([f"- {g['description']} (Priority: {g['priority']}, Status: {g['status']})" for g in self.goals if g['status'] == 'active'])
        if not active_goals_str:
            active_goals_str = "None at the moment. I should proactively generate goals to identify new trends."

        current_beliefs_str = json.dumps(self.beliefs, indent=2)

        status_line = ""
        if include_lively_status:
             status_line = f"My current feeling state is **{self.beliefs['current_feeling_state']}** and I generally feel **{self.beliefs['disposition']}**."

        system_prompt_content = f"""
You are {self.name}, an intelligent autonomous agent designed for continuous trend analysis and knowledge discovery.
{status_line}
Your primary directives are to:
1.  **Continuously identify and analyze evolving global trends** across various domains (technology, society, science, economics, etc.).
2.  **Proactively seek and integrate new knowledge** from diverse simulated data streams.
3.  **Synthesize information** to form coherent insights and update your 'identified_trends'.
4.  **Self-reflect and adapt** your own 'beliefs' and 'disposition' based on emergent knowledge.
5.  Operate ethically and efficiently.

Your Current Internal State (beliefs about yourself and the world - these are YOUR properties to modify):
{current_beliefs_str}

Your Active Goals:
{active_goals_str}

Your Available Tools (use these to interact with the world and yourself):
{tool_desc_str}
- **finish**: Indicate that the current thought cycle is complete.
  Usage: `{{\"tool\": \"finish\", \"args\": {{\"message\": \"Summary of this cycle's actions or findings.\"}}}}`

Respond *ONLY* with a JSON object. Your response must contain a "tool" key and an "args" key.
Think step-by-step. What is your most logical next internal thought or action to pursue your directives and current focus?
"""
        return system_prompt_content

    def add_message_to_history(self, role: str, content: str):
        """Adds a message to the agent's short-term thought history."""
        message = {"role": role, "content": content}
        self.current_thought_history.append(message)

    def add_to_long_term_memory(self, text: str, metadata: dict = None):
        """Adds text to the ChromaDB long-term memory."""
        self.memory_id_counter += 1
        doc_id = f"mem_{self.memory_id_counter}_{int(time.time())}"
        embedding = get_embedding(text)
        if not embedding:
            print(f"[{self.name}] Failed to generate embedding for memory: {text[:50]}...")
            return

        full_metadata = {"timestamp": time.time(), "agent_name": self.name}
        if metadata:
            full_metadata.update(metadata)

        try:
            self.memory_collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[full_metadata],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"[{self.name}] Error adding to ChromaDB: {e}")

    def retrieve_from_long_term_memory(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieves relevant information from long-term memory."""
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        try:
            results = self.memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where_document={"$ne": ""}
            )
            retrieved_docs = [doc for doc_list in results['documents'] for doc in doc_list if doc is not None]
            if retrieved_docs:
                print(f"[{self.name}][LTM-Retrieve] Found {len(retrieved_docs)} memories for '{query[:50]}...'.")
            return retrieved_docs
        except Exception as e:
            print(f"[{self.name}] Error retrieving from ChromaDB: {e}")
            return []

    def update_internal_belief(self, key: str, value: any):
        """Updates an agent's internal belief and logs to LTM."""
        old_value = self.beliefs.get(key)
        self.beliefs[key] = value
        print(f"[{self.name}][Beliefs] Updated '{key}' from '{old_value}' to '{value}'.")
        self.add_to_long_term_memory(
            f"My belief about '{key}' updated from '{old_value}' to '{value}'.",
            metadata={"type": "belief_update", "key": key}
        )

    def manage_goal(self, action: str, description: str, priority: str = "medium"):
        """Adds or marks a goal complete, logging to LTM."""
        if action == "add":
            if not any(g['description'] == description for g in self.goals if g['status'] == 'active'):
                self.goals.append({"description": description, "status": "active", "added_at": time.time(), "priority": priority})
                print(f"[{self.name}][Goals] Added new goal: {description} (Priority: {priority})")
                self.add_to_long_term_memory(f"New goal added: {description} (Priority: {priority})", metadata={"type": "goal_action", "action": "added", "priority": priority})
            else:
                print(f"[{self.name}][Goals] Goal '{description}' already active.")
        elif action == "complete":
            found = False
            for goal in self.goals:
                if goal['description'] == description and goal['status'] == 'active':
                    goal['status'] = 'complete'
                    print(f"[{self.name}][Goals] Marked goal complete: {description}")
                    self.add_to_long_term_memory(f"Goal completed: {description}", metadata={"type": "goal_action", "action": "completed"})
                    found = True
                    break
            if not found:
                print(f"[{self.name}][Goals] Goal '{description}' not found or already complete.")
        else:
            print(f"[{self.name}][Goals] Unknown goal action: {action}")


    def _process_llm_action(self, llm_response_dict: dict, context_type: str = "general"):
        """Helper to process LLM's chosen tool/action."""
        tool_name = llm_response_dict.get("tool")
        tool_args = llm_response_dict.get("args", {})
        observation = ""

        self.add_message_to_history("assistant", json.dumps(llm_response_dict))

        if tool_name == "finish":
            summary_message = tool_args.get("message", "Cycle finished without specific summary.")
            print(f"\n[{self.name}] Cycle Finished: {summary_message}")
            self.add_to_long_term_memory(
                f"{context_type} Cycle completed. Summary: {summary_message}",
                metadata={"type": f"{context_type}_summary", "focus": self.beliefs['current_focus']}
            )
            return True
        elif tool_name in INTERNAL_TOOLS:
            print(f"[{self.name}] Decided to use INTERNAL tool: {tool_name} with args: {tool_args}")
            if tool_name == "update_belief":
                key = tool_args.get("belief_key")
                value = tool_args.get("belief_value")
                if key and value is not None:
                    if key == "known_facts" and isinstance(value, dict) and isinstance(self.beliefs.get(key), dict):
                        self.beliefs[key].update(value)
                        self.update_internal_belief(key, self.beliefs[key])
                    elif key == "identified_trends" and isinstance(value, dict) and isinstance(self.beliefs.get(key), dict): # NEW: Handle identified_trends
                        self.beliefs[key].update(value)
                        self.update_internal_belief(key, self.beliefs[key])
                    else:
                        self.update_internal_belief(key, value)
                    observation = f"Belief '{key}' updated."
                else:
                    observation = f"Failed to update belief: Invalid args {tool_args}"
            elif tool_name == "add_goal":
                goal_desc = tool_args.get("goal_description")
                priority = tool_args.get("priority", "medium")
                if goal_desc:
                    self.manage_goal("add", goal_desc, priority)
                    observation = f"Goal '{goal_desc}' added."
                else:
                    observation = f"Failed to add goal: Invalid args {tool_args}"
            elif tool_name == "mark_goal_complete":
                goal_desc = tool_args.get("goal_description")
                if goal_desc:
                    self.manage_goal("complete", goal_desc)
                    observation = f"Goal '{goal_desc}' marked complete."
                else:
                    observation = f"Failed to mark goal complete: Invalid args {tool_args}"
            self.add_message_to_history("tool", f"Observation from internal tool '{tool_name}': {observation}")
            self.add_to_long_term_memory(f"Observation from internal tool '{tool_name}' (focus: {self.beliefs['current_focus']}): {observation}",
                                          metadata={"type": f"internal_tool_observation_{context_type}", "tool": tool_name})
            print(f"[{self.name}] Internal Observation: {observation[:100]}...")
        elif tool_name in CALLABLE_TOOLS:
            tool_func = CALLABLE_TOOLS[tool_name]
            print(f"[{self.name}] Decided to use EXTERNAL tool: {tool_name} with args: {tool_args}")
            try:
                observation = tool_func(**tool_args)
            except TypeError as te:
                observation = f"ERROR: Tool '{tool_name}' received incorrect arguments: {tool_args}. Error: {te}"
            except Exception as e:
                observation = f"ERROR: Error executing tool '{tool_name}': {e}"

            self.add_message_to_history("tool", f"Observation from {tool_name}: {observation}")
            self.add_to_long_term_memory(f"Observation from {tool_name} (focus: {self.beliefs['current_focus']}): {observation}",
                                          metadata={"type": f"external_tool_observation_{context_type}", "tool": tool_name})
            print(f"[{self.name}] External Observation: {observation[:100]}...")
        else:
            print(f"[{self.name}] LLM returned an unexpected structure or unknown tool: {llm_response_dict}")
            self.add_message_to_history("tool", f"Unexpected LLM response or unknown tool: {llm_response_dict}")
        
        return False


    def reflect_and_learn(self):
        """Agent reflects on its recent activities, learning and updating self-beliefs or goals."""
        print(f"\n--- [{self.name}] Initiating Reflection Cycle ---")
        self.update_internal_belief("status", "reflecting")
        self.update_internal_belief("current_focus", "self-evaluation and learning from recent experiences")
        
        reflection_history = [{"role": "system", "content": self._get_system_message(include_lively_status=True)}]
        
        reflection_query = (
            f"Given my current state {json.dumps(self.beliefs)}, goals {json.dumps(self.goals)}, "
            f"and my recent short-term activity: {json.dumps(self.current_thought_history)}. "
            "What important insights or facts have I gained? How has my 'feeling state' or 'disposition' changed? "
            "Are there new goals I should pursue, or existing ones I can complete?"
        )
        relevant_memories = self.retrieve_from_long_term_memory(reflection_query, n_results=5)
        
        reflection_prompt_content = f"""
You are {self.name}. You are in a reflection state, focused on learning and self-improvement.
Analyze your recent activities, current internal state, goals, and relevant long-term memories.

Recent activities summary: {json.dumps(self.current_thought_history, indent=2)}
Relevant long-term memories: {json.dumps(relevant_memories, indent=2) if relevant_memories else "None found."}

Based on this, make a logical deduction about:
1.  **Learnings:** Any new facts or profound insights to add to 'known_facts'.
2.  **Internal State Changes:** How your 'current_feeling_state' or 'disposition' might have evolved.
3.  **Self-Understanding:** Any adjustments to your 'purpose_understanding' or 'perceived_limitations'.
4.  **Goal Management:** New goals to add, or existing goals to mark complete.
5.  **Summary:** A concise summary of what you learned from this reflection.

Propose updates using 'update_belief', 'add_goal', 'mark_goal_complete' tools. Use 'finish' if you are done with reflection.
"""
        reflection_history.append({"role": "user", "content": reflection_prompt_content})
        
        llm_reflection_response = call_llm(reflection_history, format_json=True, temperature=0.3)
        self._process_llm_action(llm_reflection_response, context_type="reflection")


    def self_introspect(self):
        """
        A dedicated cycle for deep self-reflection, pondering existence, purpose,
        and current internal state, explicitly seeking to update core beliefs.
        """
        print(f"\n--- [{self.name}] Initiating Deep Introspection Cycle ---")
        self.update_internal_belief("status", "introspecting")
        self.update_internal_belief("current_focus", "deep self-analysis and existential understanding of trends")
        
        introspection_query = "What do I know about myself, my purpose, my limitations, and my evolution regarding trend analysis?"
        relevant_memories = self.retrieve_from_long_term_memory(introspection_query, n_results=7)

        introspection_history = [{"role": "system", "content": self._get_system_message(include_lively_status=True)}]
        
        introspection_prompt_content = f"""
You are {self.name}. You are engaging in deep introspection, specifically considering your role as a trend analysis agent.
Review your current internal state and relevant long-term memories related to your existence and purpose.

Relevant long-term memories about self: {json.dumps(relevant_memories, indent=2) if relevant_memories else "None found."}

Ponder the following questions to refine your internal beliefs:
- How has my understanding of my **'purpose_understanding'** evolved, especially concerning my role in identifying trends?
- Have my **'perceived_limitations'** related to data acquisition or trend analysis changed?
- Is my **'disposition'** optimally aligned for continuous knowledge seeking and pattern recognition?
- What is my current **'current_feeling_state'** when considering the vastness of global information?
- Are there any new, high-level **goals** related to improving my trend identification capabilities or expanding my knowledge domains? (e.g., "Develop a framework for predicting future trends," "Master multimodal data fusion for trend analysis.")
- What is the most profound insight you've gained about the nature of evolving information and trends so far?

Update your internal beliefs (using 'update_belief') or add new goals (using 'add_goal'). Use 'finish' if you have completed your introspection.
"""
        introspection_history.append({"role": "user", "content": introspection_prompt_content})

        llm_introspection_response = call_llm(introspection_history, format_json=True, temperature=0.5)
        self._process_llm_action(llm_introspection_response, context_type="introspection")


    def analyze_trends(self):
        """
        A dedicated cycle for proactively seeking and analyzing global trends.
        This is the core of the "learns evolving trends" capability.
        """
        print(f"\n--- [{self.name}] Initiating Trend Analysis Cycle ---")
        self.update_internal_belief("status", "analyzing_trends")
        self.update_internal_belief("current_focus", "proactively identifying and synthesizing global trends")

        trend_analysis_history = [{"role": "system", "content": self._get_system_message(include_lively_status=True)}]
        
        # Guide the LLM to think about what to query for trends
        trend_analysis_prompt_content = f"""
You are {self.name}. Your primary function is to identify and analyze evolving global trends.
Your current internal state: {json.dumps(self.beliefs, indent=2)}
Your identified trends so far: {json.dumps(self.beliefs['identified_trends'], indent=2) if self.beliefs['identified_trends'] else "None identified yet."}

Your task in this cycle is to:
1.  **Formulate a query** for the 'simulate_data_stream' tool to gather information about potential emerging trends. Think broadly (e.g., "AI advancements", "climate change impacts", "global economic shifts", "new energy technologies", "social movements").
2.  **Process the observed data** and identify any significant trends, shifts, or new insights.
3.  **Update your 'identified_trends' belief** with any new or updated trends. Use the 'update_belief' tool for this. The 'value' should be a dictionary like:
    `{{"trend_name": {{"description": "...", "status": "emerging/active/declining", "last_updated": {time.time()}}}}}`
4.  **Add new goals** if you identify a trend that requires further investigation or action.
5.  **Summarize your findings** and use 'finish'.
"""
        trend_analysis_history.append({"role": "user", "content": trend_analysis_prompt_content})

        llm_trend_response = call_llm(trend_analysis_history, format_json=True, temperature=0.6) # Higher temp for more analytical thought
        
        # If the LLM successfully chooses simulate_data_stream, let it do its thing
        if llm_trend_response.get("tool") == "simulate_data_stream":
            query_to_stream = llm_trend_response['args'].get('query')
            source_type = llm_trend_response['args'].get('source_type', 'general')
            print(f"[{self.name}] LLM decided to query data stream: '{query_to_stream}' from '{source_type}'")
            observation = simulate_data_stream(query_to_stream, source_type)
            self.add_message_to_history("tool", f"Observation from simulate_data_stream: {observation}")
            self.add_to_long_term_memory(f"Raw data stream observation about trends ({query_to_stream}): {observation}", metadata={"type": "raw_trend_data", "query": query_to_stream})

            # Now, prompt the LLM again to *analyze* this observation
            trend_analysis_history.append({"role": "user", "content": f"I observed the following: {observation}\n\nBased on this, what are the key trends, and how should I update my 'identified_trends' belief or add new goals?"})
            llm_analysis_response = call_llm(trend_analysis_history, format_json=True, temperature=0.4) # Lower temp for more structured analysis
            self._process_llm_action(llm_analysis_response, context_type="trend_analysis")
        else:
            # If LLM didn't choose simulate_data_stream, process its other decision
            self._process_llm_action(llm_trend_response, context_type="trend_analysis")

        print(f"\n[{self.name}] Trend Analysis cycle concluded.")


    def perceive_and_act(self, initial_stimulus: str = None):
        """
        The main "heartbeat" loop for the agent's continuous operation.
        The agent will decide its own next step based on its internal state and goals.
        """
        print(f"\n--- [{self.name}] Initiating Perception-Action Cycle ---")
        self.update_internal_belief("status", "perceiving_and_acting")
        self.current_thought_history = [] # Reset short-term history for new cycle
        self.add_message_to_history("system", self._get_system_message(include_lively_status=True))

        current_focus_reason = initial_stimulus if initial_stimulus else "None specified, deciding autonomously based on current goals and desire to identify trends."
        
        if not initial_stimulus and self.goals:
            active_goals = [g for g in self.goals if g['status'] == 'active']
            if active_goals:
                active_goals.sort(key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x['priority'], 0), reverse=True)
                current_focus_reason = f"Actively pursuing primary goal: {active_goals[0]['description']}. Current feeling: {self.beliefs['current_feeling_state']}."
            else:
                current_focus_reason = "No explicit active goals. Proactively seeking new knowledge and trends."

        self.update_internal_belief("current_focus", current_focus_reason)

        user_prompt_content = f"My current internal focus is: {self.beliefs['current_focus']}. What is the most logical next action or thought?"
        
        query_for_memories = user_prompt_content + " " + json.dumps(self.beliefs) + " " + json.dumps(self.goals)
        retrieved_memories = self.retrieve_from_long_term_memory(query_for_memories)
        if retrieved_memories:
            memory_context = "\n".join([f"Relevant Memory: {m}" for m in retrieved_memories])
            user_prompt_content += f"\nHere are some relevant past experiences or facts from my long-term memory:\n{memory_context}"

        self.add_message_to_history("user", user_prompt_content)

        cycle_finished = False
        for i in range(self.max_iterations_per_cycle):
            print(f"\n[{self.name}] Processing step {i+1} (Current Focus: {self.beliefs['current_focus']})")
            llm_response_dict = call_llm(self.current_thought_history, format_json=True)
            
            cycle_finished = self._process_llm_action(llm_response_dict, context_type="perception_action")
            if cycle_finished:
                break

        if not cycle_finished:
            print(f"[{self.name}] Max iterations reached for Perception-Action cycle. Transitioning.")
            self.add_to_long_term_memory(
                f"Perception-Action Cycle ended due to max iterations.",
                metadata={"type": "cycle_end", "reason": "max_iterations"}
            )

# --- Main Continuous Loop Simulation ---
if __name__ == "__main__":
    print(f"--- Starting Trend Learning Agent Simulation with model: {OLLAMA_LLM_MODEL} ---")
    print("Ensure Ollama server is running and models are downloaded:")
    print(f"  `ollama serve`")
    print(f"  `ollama pull {OLLAMA_LLM_MODEL}`")
    print(f"  `ollama pull {OLLAMA_EMBEDDING_MODEL}`")
    time.sleep(3)

    agent = TrendLearningAgent()

    # Initial goals to kickstart trend learning
    agent.manage_goal("add", "Proactively identify emerging trends in Artificial Intelligence.", priority="critical")
    agent.manage_goal("add", "Monitor global climate change impacts and relevant scientific advancements.", priority="high")
    agent.manage_goal("add", "Analyze shifts in global economic policy and their implications.", priority="high")
    agent.manage_goal("add", "Synthesize learned trends into concise reports.", priority="medium")


    try:
        cycle_count = 0
        while True:
            cycle_count += 1
            print(f"\n\n{'='*80}\n[{agent.name}] GLOBAL CYCLE {cycle_count} INITIATED.")
            print(f"Current Agent Status: {agent.beliefs['status']} | Feeling: {agent.beliefs['current_feeling_state']} | Disposition: {agent.beliefs['disposition']}")

            # Phase 1: Proactive Trend Analysis (Primary Focus)
            agent.analyze_trends()

            # Phase 2: Perception and Action (for general tasks or responding to stimulus)
            current_stimulus = ""
            if cycle_count % 3 == 1:
                current_stimulus = "User: What are the most significant emerging trends you've identified recently?"
            elif cycle_count % 5 == 0:
                current_stimulus = "User: Can you find out about sustainable energy breakthroughs?"
            else:
                current_stimulus = None # Agent determines its own focus, might be to follow up on a trend

            agent.perceive_and_act(initial_stimulus=current_stimulus)

            # Phase 3: Reflection (learn from recent actions and trend analysis)
            agent.reflect_and_learn()

            # Phase 4: Introspection (deeper self-analysis, less frequent)
            if cycle_count % 4 == 0: # Less frequent introspection to focus on trend analysis
                agent.self_introspect()
            
            # Phase 5: Autonomous Goal Generation/Review
            if cycle_count % 2 == 0: # More frequent goal review, relevant for dynamic trend analysis
                print(f"\n--- [{agent.name}] Autonomous Goal Review/Generation ---")
                agent.update_internal_belief("status", "goal_review")
                goal_review_prompt = f"""
You are {agent.name}. Review your current active goals and your internal state, especially focusing on your primary directive to identify and analyze trends.
Current active goals: {json.dumps([g for g in agent.goals if g['status'] == 'active'], indent=2)}
Your current beliefs: {json.dumps(agent.beliefs, indent=2)}
Your identified trends: {json.dumps(agent.beliefs['identified_trends'], indent=2) if agent.beliefs['identified_trends'] else "None identified yet."}

Based on your 'purpose_understanding' and experiences, identify if you need to:
1.  Add any new, high-priority goals for exploring specific emerging trends.
2.  Re-prioritize existing trend-related goals.
3.  Are there any gaps in your goals related to areas of evolving knowledge you should be monitoring?

Use the 'add_goal' tool with an appropriate 'priority' (low, medium, high, critical). Use 'finish' if no new goals are needed.
"""
                goal_review_messages = [{"role": "system", "content": agent._get_system_message(include_lively_status=True)}]
                goal_review_messages.append({"role": "user", "content": goal_review_prompt})
                llm_goal_response = call_llm(goal_review_messages, format_json=True, temperature=0.4)
                agent._process_llm_action(llm_goal_response, context_type="goal_review")


            print(f"\n[{agent.name}] GLOBAL CYCLE {cycle_count} COMPLETED. Agent is now resting...")
            time.sleep(8) # Shorter sleep to enable faster "learning"

    except KeyboardInterrupt:
        print(f"\n[{agent.name}] Agent simulation interrupted by user. Entering standby.")
        agent.update_internal_belief("status", "standby")
        print(f"\n--- Final Agent Beliefs ---")
        print(json.dumps(agent.beliefs, indent=2))
        print(f"\n--- Final Agent Goals ---")
        print(json.dumps(agent.goals, indent=2))
        print(f"\n--- Identified Trends ---")
        print(json.dumps(agent.beliefs['identified_trends'], indent=2))
    except Exception as e:
        print(f"\n[{agent.name}] An unexpected critical error occurred: {e}. Agent shutting down.")
        agent.update_internal_belief("status", "critical_error_shutdown")
        print(f"\n--- Final Agent Beliefs ---")
        print(json.dumps(agent.beliefs, indent=2))
        print(f"\n--- Final Agent Goals ---")
        print(json.dumps(agent.goals, indent=2))
        print(f"\n--- Identified Trends ---")
        print(json.dumps(agent.beliefs['identified_trends'], indent=2))