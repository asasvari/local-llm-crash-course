import chainlit as cl

from ctransformers import AutoModelForCausalLM

llm = None


def get_llama_prompt(instruction: str, history: list[str] | None = None) -> str:
    global llm
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."

    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"

    if history:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "

    prompt += f"{instruction} [/INST]</s>"
    return prompt


def get_orca_prompt(instruction: str, history: list[str] | None = None) -> str:
    global llm
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"

    if history:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "

    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    if "Llama" in llm.model_path:
        return get_llama_prompt(instruction, history)

    return get_orca_prompt(instruction, history)


@cl.on_message
async def on_message(message: cl.Message):
    global llm
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()
    if message.content == "forget everything":
        cl.user_session.set("message_history", [])
        for word in "Uh oh, I've just forgotten our conversation history":
            await msg.stream_token(word)
        await msg.update()
        return

    if message.content == "use orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )
        for word in "Model changed to Orca":
            await msg.stream_token(word)
        await msg.update()
        return

    if message.content == "use llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )
        for word in "Model changed to Llama":
            await msg.stream_token(word)
        await msg.update()
        return

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    message_history.append(response)
    await msg.update()


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
