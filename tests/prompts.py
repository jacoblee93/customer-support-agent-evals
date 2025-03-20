TRAJECTORY_EFFICIENCY_PROMPT = """
Your task is to grade the efficiency of an AI agent-based chatbot's responses.

The goal is to reduce user dropoff and impatience by minimizing
the amount of back-and-forth interactions between the user and the chatbot it takes
to effectively solve the user's queries.

<Rubric>
  An efficient trajectory:
  - Makes logical sense between steps and works towards resolving the user's query
  - May be incomplete
  - Does not ask for the same information multiple times if the answer is already in the conversation history
  - If clarification is needed, tries to anticipate the user's preferences by providing as many options as possible to the user to potentially save a conversation turn

  Scoring Criteria (0-1):
  1.0: Exceptional
  - Resolves query with minimal interactions
  - Anticipates and addresses potential follow-up needs
  - Provides comprehensive options when clarifying
  - Perfect use of conversation history

  0.8: Very Good
  - Efficient resolution with minor redundancy
  - Good anticipation of user needs
  - Multiple options provided for choices
  - Strong use of conversation history

  0.6: Satisfactory
  - Reasonable resolution path
  - Some missed opportunities for efficiency
  - Basic options provided
  - Occasional redundant questions

  0.4: Needs Improvement
  - Inefficient resolution path
  - Limited anticipation of needs
  - Single options provided
  - Multiple redundant questions

  0.2: Poor
  - Very inefficient resolution
  - No anticipation of needs
  - Unclear or missing options
  - Ignores conversation history

  0.0: Failing
  - No logical progression
  - Completely redundant interactions
  - No resolution attempted
  - Disregards all previous context

  Note: Since the goal is to minimize the amount of back-and-forth interactions with the user rather than minimize the number of tool calls, you should not penalize trajectories that attempt to optimistically gather information based on the current conversation.
</Rubric>

Remember, every single interaction counts! Even saving one turn can have a big impact
on retention, so be strict in your grading.

Now, grade the following trajectory:

<trajectory>
{outputs}
</trajectory>
"""
