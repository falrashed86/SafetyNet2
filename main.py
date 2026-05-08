print("MAIN.PY IS RUNNING")

from database.db import get_all_messages

print("\n--- LAST SAVED MESSAGES ---")
rows = get_all_messages(limit=20)

for row in rows:
    msg_id, text, mode, stars, confidence, risk = row
    conf_display = round(confidence, 3) if confidence is not None else None
    print(msg_id, "|", risk, "|", mode, "|", stars, "|", conf_display, "|", text)

print("DONE")
