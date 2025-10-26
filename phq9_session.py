from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate  # new
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
import random

load_dotenv()
model = ChatOpenAI(model="gpt-4o")

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

classification_chain = (
    ChatPromptTemplate.from_template("""
    You are a compassionate mental health assistant.
    Based on: "{response}", classify as:
    "Not at all", "Several days", "More than half the days", or "Nearly every day".
    Output only the phrase.
    """) | model | StrOutputParser()
)

empathetic_reply_chain = (
    ChatPromptTemplate.from_template("""
    After reading this:
    Q: {question}
    A: {response}
    Write one brief, caring sentence validating what they shared.
    """) | model | StrOutputParser()
)

final_chain = (
    ChatPromptTemplate.from_template("""
    The user shared:
    {all_responses}
    Their total PHQ-9 score is {score}.
    Write a short, warm message starting with:
    "Depression severity: ..."
    Then a 1-paragraph message under 100 words offering kindness and reassurance.
    """) | model | StrOutputParser()
)

score_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

confirmation_phrases = {
    "Not at all": [
        "I'll note that as 'not at all'.",
        "Thanks, marking 'not at all'.",
        "Got it — not at all."
    ],
    "Several days": [
        "Noted as 'several days'.",
        "I'll mark that down.",
        "Okay, several days it is."
    ],
    "More than half the days": [
        "I hear you — marking it accordingly.",
        "Noted: more than half the days.",
        "Understood, thank you."
    ],
    "Nearly every day": [
        "That sounds heavy — noted.",
        "Thanks for your honesty. Marking as 'nearly every day'.",
        "Got it, nearly every day."
    ]
}

class PHQ9Session:
    def __init__(self):
        self.answers = []
        self.total_score = 0
        self.current_index = 0
        self.started = False

    def start(self):
        self.started = True
        self.answers = []
        self.total_score = 0
        self.current_index = 0
        return {
            "bot_message": "Great! Let's begin.\n" + phq9_questions[0],
            "is_final": False
        }

    def process_response(self, user_response):
        if not self.started:
            return {"bot_message": "Please say 'start' to begin the test."}

        if self.current_index >= len(phq9_questions):
            return {"bot_message": "You've completed the test!"}

        question = phq9_questions[self.current_index]
        classification = classification_chain.invoke({"response": user_response}).strip()
        score = score_mapping.get(classification, 0)
        self.total_score += score
        empathetic = empathetic_reply_chain.invoke({"question": question, "response": user_response})
        confirm = random.choice(confirmation_phrases.get(classification, [f"Marked as {classification.lower()}"]))

        self.answers.append((question, user_response, classification))
        self.current_index += 1

        if self.current_index >= len(phq9_questions):
            summary = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in self.answers])
            final_message = final_chain.invoke({
                "all_responses": summary,
                "score": self.total_score
            })
            return {
                "is_final": True,
                "bot_message": f"{confirm}\n{empathetic}\n🧠 Final message:\n{final_message}",
                "final_message": final_message
            }

        next_q = phq9_questions[self.current_index]
        return {
            "bot_message": f"{confirm}\n{empathetic}\n {next_q}",
            "is_final": False
        }
