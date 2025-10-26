from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize model
model = ChatOpenAI(model="gpt-4o")

# PHQ-9 questions
phq9_questions = [
    "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you been feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you felt tired or had little energy?",
    "Over the last 2 weeks, how often have you had poor appetite or been overeating?",
    "Over the last 2 weeks, how often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Over the last 2 weeks, how often have you had trouble concentrating on things, such as reading or watching TV?",
    "Over the last 2 weeks, how often have you been moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you’ve been moving around a lot more than usual?",
    "Over the last 2 weeks, how often have you had thoughts that you would be better off dead or of hurting yourself in some way?"
]

# Scoring prompt
classification_prompt = ChatPromptTemplate.from_template(
    """
    You are a compassionate mental health assistant helping assess depression symptoms.
    Based on the user's detailed response: "{response}", classify their experience as one of the following:
    "Not at all", "Several days", "More than half the days", or "Nearly every day".
    Only output the matching phrase exactly.
    """
)

classification_chain = classification_prompt | model | StrOutputParser()

# Mapping score values
score_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

user_answers = []
total_score = 0

print("Hello, I'm here to listen and help. Please answer the next few questions honestly and openly. 🫂\n")

for i, question in enumerate(phq9_questions):
    print(f"{i+1}. {question}")
    user_input = input("Your response: ")
    user_answers.append((question, user_input))

    classification = classification_chain.invoke({"response": user_input})
    score = score_mapping.get(classification, 0)
    total_score += score

    print(f"\n👉 Response interpreted as: {classification} ({score} point{'s' if score != 1 else ''})\n")

# Create empathetic interpretation prompt
final_message_prompt = ChatPromptTemplate.from_template(
    """
    The user shared these feelings:
    {all_responses}

    Their total PHQ-9 score is {score}.
    Based on this score and their descriptions, generate a short, warm, emotionally supportive paragraph. Start by stating the depression severity clearly (e.g., "Depression severity: Moderately severe").
    Then write a kind message that reflects understanding and reassurance based on the user's emotional experience.
    important: Keep it under 100 words.
    """
)

final_chain = final_message_prompt | model | StrOutputParser()

response_summary = "\n".join([f"Q: {q}\nA: {a}" for q, a in user_answers])
final_message = final_chain.invoke({"all_responses": response_summary, "score": total_score})

print("\n" + "="*60)
print(final_message)
print("="*60)
