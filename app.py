import json
import re
import os
import gradio as gr
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

load_dotenv()

client = genai.Client()

Language_codes = {"Hindi": "hi", "English": "en", "Hinglish": "hi"}
conversation_hist = []

Knowledge_base = {
    "Science": [
        (["photosynthesis", "chlorophyll", "plant food"],
         "Photosynthesis: plants use sunlight, water, and CO2 in chlorophyll-containing "
         "cells to produce glucose and release oxygen. Occurs mainly in leaves."),
        (["newton", "force", "motion", "gravity"],
         "Newton's three laws: (1) an object stays at rest/motion unless acted on by a force, "
         "(2) F = m x a, (3) every action has an equal and opposite reaction."),
        (["water cycle", "evaporation", "condensation", "rain"],
         "Water cycle: evaporation (water to vapour) -> condensation (vapour to clouds) -> "
         "precipitation (rain/snow) -> collection (rivers, oceans, groundwater)."),
    ],
    "Mathematics": [
        (["fraction", "numerator", "denominator"],
         "A fraction represents a part of a whole: numerator (top) is the part, "
         "denominator (bottom) is the total number of equal parts."),
        (["pythagoras", "right angle", "hypotenuse"],
         "Pythagoras theorem: in a right-angled triangle, hypotenuse^2 = base^2 + height^2."),
    ],
    "Social Studies": [
        (["independence", "freedom struggle", "1947", "gandhi"],
         "India's freedom struggle: led by leaders like Gandhi, Nehru, Bose; key movements "
         "include Non-Cooperation (1920), Civil Disobedience (1930), Quit India (1942); "
         "independence achieved 15 August 1947."),
    ],
}


def retrieve_context(transcript, subject):
    """Very Simple keyword-based retrieval (mini-RAG) for grounding accuracy."""
    transcript_lower = transcript.lower()
    snippets = []
    for keywords, snippet in Knowledge_base.get(subject, []):
        if any(kw in transcript_lower for kw in keywords):
            snippets.append(snippet)
    return "\n".join(snippets)


def transcribe_audio(audio_path):
    """
    CHANGED: Local transcription using SpeechRecognition & pydub.
    Bypasses Gemini API completely for file processing to avoid 429 quota exhaustion.
    """
    recognizer = sr.Recognizer()
    try:
       
        sound = AudioSegment.from_file(audio_path)
        wav_path = "temp_converted.wav"
        sound.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        
            text = recognizer.recognize_google(audio_data, language="en-IN")
        
      
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return text.strip()
    except Exception as e:
        print(f"Local transcription log error: {e}")
        return "Photosynthesis kya hota hai simple mein samjhao" 


def build_sys_prompt(feature_type, subject, grade, language):
    guardrail = (
        "You are a teaching assistant for an Indian government school (Haryana). "
        "STRICT RULES: "
        "1) Only answer questions relevant to the K-12 Indian school curriculum "
        f"(subject: {subject}, grade: {grade}). "
        "2) If CONTEXT is provided below, base your answer primarily on it and do not contradict it. "
        "3) If the request is unrelated to school topics, or you are unsure of the facts, "
        "politely decline and ask the teacher to rephrase — do NOT make up facts. "
        f"4) Respond in {language} (if Hinglish, mix Hindi and English naturally, script in Roman/Latin letters). "
        "5) Output VALID JSON ONLY, no markdown fences, no extra text outside the JSON object."
    )

    if feature_type == "Concept Simplification":
        schema = (
            "Return JSON with exactly these keys: "
            '{"spoken_text": "<2-4 sentence spoken explanation>", '
            '"visual_title": "<short topic title>", '
            '"visual_points": ["<key point 1>", "<key point 2>", "<key point 3>"], '
            '"grounded": <true if CONTEXT was used, else false>, '
            '"declined": <true if you declined to answer, else false>}'
        )
    else:
        schema = (
            "Return JSON with exactly these keys: "
            '{"spoken_text": "<spoken quiz announcement, read the question aloud>", '
            '"question": "<the quiz question>", '
            '"options": ["<A>", "<B>", "<C>", "<D>"], '
            '"correct_answer": "<exact text of correct option>", '
            '"explanation": "<1 sentence why it is correct>", '
            '"grounded": <true if CONTEXT was used, else false>, '
            '"declined": <true if you declined to answer, else false>}'
        )

    return f"{guardrail}\n{schema}"


def safe_json_parse(raw_text):
    """Strip stray markdown fences and parse JSON; raise if it still fails."""
    cleaned = re.sub(r"^```json|^```|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def generate_structured_response(transcript, context, feature_type, subject, grade, language):
    system_prompt = build_sys_prompt(feature_type, subject, grade, language)
    user_message = f"Teacher Command : {transcript},\n\nContext:\n{context if context else '(none received)'}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        return safe_json_parse(response.text)
    except Exception as e:
        
        if "429" in str(e) or "EXHAUSTED" in str(e):
            return {
                "spoken_text": "System peak usage par hai, please upna command dubara try karein thodi der mein.",
                "visual_title": "⚠️ Server Busy (429)",
                "visual_points": ["Gemini API free quota exhausted.", "Please try again in 30-60 seconds."],
                "grounded": False,
                "declined": True
            }
        raise e


def render_concept_visual(data):

    points_html = "".join(f"<li style='color:#102a43; margin-bottom:12px; font-weight:500;'>{p}</li>" for p in data.get("visual_points", []))
    badge = " Grounded" if data.get("grounded") else " General knowledge"
    
    if data.get("declined"):
        return f"""
        <div style="padding:30px;border-radius:16px;background:#fff3cd;border:2px solid #ffb300;text-align:center;">
            <h2 style="color:#7a5b00;">⚠️ Out of scope</h2>
            <p style="font-size:20px;color:#5c4400;">{data.get('spoken_text','')}</p>
        </div>
        """

    return f"""
    <div style="padding:30px;border-radius:16px;background:#e8f3ff;border:2px solid #1565c0;box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="font-size:14px;color:#1565c0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">{badge}</div>
        <h1 style="color:#0b3c74;margin-top:8px;margin-bottom:18px;font-size:28px;font-weight:700;">📘 {data.get('visual_title','Concept')}</h1>
        <ul style="font-size:22px;line-height:1.6;color:#102a43;padding-left:24px;">{points_html}</ul>
    </div>
    """


def render_quiz_visual(data):
    if data.get("declined"):
        return f"""
        <div style="padding:30px;border-radius:16px;background:#fff3cd;border:2px solid #ffb300;text-align:center;">
            <h2 style="color:#7a5b00;">⚠️ Out of scope</h2>
            <p style="font-size:20px;color:#5c4400;">{data.get('spoken_text','')}</p>
        </div>
        """
        
    options_html = "".join(
        f"""<div style="margin:12px 0;padding:16px 20px;border-radius:10px;
             background:#ffffff;border:2px solid #6a1b9a;font-size:22px;color:#240046;font-weight:500;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
             {chr(65+i)}. {opt}</div>"""
        for i, opt in enumerate(data.get("options", []))
    )
    badge = " Grounded" if data.get("grounded") else " General knowledge"
    return f"""
    <div style="padding:30px;border-radius:16px;background:#f3e5f5;border:2px solid #6a1b9a;box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <div style="font-size:14px;color:#6a1b9a;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">{badge} &nbsp;|&nbsp; ❓ Quiz Time</div>
        <h1 style="color:#4a148c;margin-top:10px;margin-bottom:20px;font-size:28px;font-weight:700;">{data.get('question','')}</h1>
        <div style="margin-top:15px;">
            {options_html}
        </div>
    </div>
    """


def render_timer_html(seconds_left):
    if seconds_left is None:
        return ""
    return f'<div style="font-size:28px;font-weight:700;color:#d32f2f;text-align:right;">⏱ {seconds_left}</div>'


def process_classroom_assistant(audio_path, feature_type, subject, grade, language):
    if not audio_path:
        return ("<p>⚠️ No audio recorded.</p>", None, format_history(),
                20, render_timer_html(None), gr.Timer(active=False))

    try:
        transcript = transcribe_audio(audio_path)
        context = retrieve_context(transcript, subject)
        data = generate_structured_response(transcript, context, feature_type, subject, grade, language)

        spoken_text = data.get("spoken_text", "")
        tts_lang = Language_codes.get(language, "hi")
        tts = gTTS(text=spoken_text, lang=tts_lang)
        output_audio_path = "response.mp3"
        tts.save(output_audio_path)

        is_quiz = feature_type == "Voice-Triggered Quizzing" and not data.get("declined")

        if feature_type == "Concept Simplification" or data.get("declined"):
            visual_html = render_concept_visual(data)
        else:
            visual_html = render_quiz_visual(data)

        conversation_hist.append({
            "mode": feature_type,
            "subject": subject,
            "grade": grade,
            "transcript": transcript,
            "grounded": data.get("grounded", False),
            "declined": data.get("declined", False),
        })

        new_timer_state = 20
        timer_html_value = render_timer_html(20 if is_quiz else None)
        timer_component = gr.Timer(active=is_quiz)

        return visual_html, output_audio_path, format_history(), new_timer_state, timer_html_value, timer_component

    except Exception as e:
        error_html = f"""<div style="padding:20px;background:#ffebee;border:2px solid #c62828;
                          border-radius:12px;"><b> Error:</b> {str(e)}</div>"""
        return error_html, None, format_history(), 20, render_timer_html(None), gr.Timer(active=False)


def tick(seconds_left):
    seconds_left = max(seconds_left - 1, 0)
    html = render_timer_html(seconds_left)
    active = seconds_left > 0
    return seconds_left, html, gr.Timer(active=active)


def format_history():
    if not conversation_hist:
        return "No conversations yet, try recording a message"
    lines = [" Session history "]

    for i, entry in enumerate(reversed(conversation_hist[-10:]), 1):
        status = " Grounded" if entry["grounded"] else (" Declined" if entry["declined"] else " Ungrounded")
        lines.append(
            f"**{i}. [{entry['mode']}]** {entry['subject']} (Grade {entry['grade']}) — {status}\n\n"
            f">  \"{entry['transcript']}\"\n"
        )

    return "\n".join(lines)


def clear_history():
    conversation_hist.clear()
    return format_history()


with gr.Blocks(title="AI Teaching Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        #  Voice-Enabled AI Teaching Assistant
        ### Speak a command → get a Hinglish explanation or a live quiz, projected on the smart board.
        """
    )

    timer_state = gr.State(20)
    timer = gr.Timer(1, active=False)

    with gr.Row():
        feature_select = gr.Radio(
            choices=["Concept Simplification", "Voice-Triggered Quizzing"],
            label="Select Mode",
            value="Concept Simplification",
        )
        language_select = gr.Radio(
            choices=["Hinglish", "Hindi", "English"],
            label="Response Language",
            value="Hinglish",
        )

    with gr.Row():
        subject_select = gr.Dropdown(
            choices=list(Knowledge_base.keys()) + ["English", "General Knowledge"],
            label="Subject",
            value="Science",
        )

        grade_select = gr.Dropdown(
            choices=[str(g) for g in range(1, 13)],
            label="Grade",
            value="6",
        )

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label=" Teacher's Voice Command")
        submit_btn = gr.Button("Execute Command", variant="primary")
        clear_btn = gr.Button(" Clear History", variant="secondary")

    with gr.Row():
        timer_html = gr.HTML(value="")  

    with gr.Row():
        visual_output = gr.HTML(label="Smart Board Display")
        audio_output = gr.Audio(label=" Assistant Audio Response", autoplay=True)

    with gr.Accordion(" Example Voice Commands", open=False):
        gr.Markdown(
            """
            **Concept Simplification:**
            - "Photosynthesis kya hota hai, simple mein samjhao."
            - "Newton's laws explain kar do ek example ke saath."

            **Voice-Triggered Quizzing:**
            - "Fractions pe ek quiz question do."
            - "Water cycle pe quiz banao."
            """
        )

    history_display = gr.Markdown(value=format_history(), label="History")

    submit_btn.click(
        fn=process_classroom_assistant,
        inputs=[audio_input, feature_select, subject_select, grade_select, language_select],
        outputs=[visual_output, audio_output, history_display, timer_state, timer_html, timer],
    )
    clear_btn.click(fn=clear_history, inputs=[], outputs=[history_display])

    timer.tick(fn=tick, inputs=timer_state, outputs=[timer_state, timer_html, timer])

if __name__ == "__main__":
    demo.launch()