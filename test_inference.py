from model.text_analyzer import analyze_message

tests = [
    "I am happy today",
    "You are ugly",
    "kill yourself",
    "I feel left out",
]

for t in tests:
    print("\nMESSAGE:", t)
    print(analyze_message(t))
