import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
import streamlit as st



os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["XAI_API_KEY"] = st.secrets["XAI_API_KEY"]
model = init_chat_model("grok-3-mini", model_provider="xai", tiktoken_model_name="gpt-3.5-turbo", temperature=0.7)

def read_file_as_string(file_path):
    """
    Reads the content of a text file and returns it as a string.
    
    :param file_path: Path to the .txt file
    :return: Content of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def refine_question(question: str) -> str:
    system_prompt = read_file_as_string("refine_question.txt")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
    )
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    # Define a new graph
    workflow = StateGraph(state_schema=State) # state instead of message state
    # Define the function that calls the model
    def call_model(state: State): # is called whenever the node is invoked
        print("call_model")
        trimmed_messages = trimmer.invoke(state["messages"])
        # prompt = prompt_template.invoke(state) # Create the prompt from the template and state
        prompt = prompt_template.invoke(
            {"messages": trimmed_messages}
        )
        print("Prompt:", prompt)
        response = model.invoke(prompt)
        return {"messages": [response]}
    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "trial_1"}}
    query = question

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages}, # based on state
        config,
    )
    return output["messages"][-1].content

def binary_class(first_stage_response):
    first_stage_response = first_stage_response.replace("你", "我").replace("妳", "我")
    binary_text = read_file_as_string("binary_prompt.txt")
    binary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", binary_text,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    binary_workflow = StateGraph(state_schema=State) # state instead of message state
    def call_model(state: State): # is called whenever the node is invoked
        print("call_model")
        prompt = binary_prompt.invoke(state) # Create the prompt from the template and state
        print("Prompt:", prompt)
        response = model.invoke(prompt)
        return {"messages": [response]}
    # Define the (single) node in the graph
    binary_workflow.add_edge(START, "model")
    binary_workflow.add_node("model", call_model)
    # Add memory
    binary_memory = MemorySaver()
    binary_app = binary_workflow.compile(checkpointer=binary_memory)
    config = {"configurable": {"thread_id": "trial_3"}}
    input_messages = [HumanMessage(first_stage_response)]
    output = binary_app.invoke(
        {"messages": input_messages}, # based on state
        config,
    )
    return output["messages"][-1].content

def determine_person(second_stage_response: str):
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    從以下段落中提取所需資訊。
    僅提取「Classification」功能中提到的properties
    不要描述理由。
    不要描述理由。
    不要描述理由。
    段落：
    {input}
    """
    )

    class Classification(BaseModel):
        inner: str = Field(description="十個字以下寫出內的一方是誰, 請簡短寫出身份, 不要描述理由"),
        outer: str = Field(description="十個字以下寫出外的一方是誰, 請簡短寫出身份, 不要描述理由"),

    # Structured LLM
    structured_llm = model.with_structured_output(Classification)

    inp = second_stage_response
    prompt = tagging_prompt.invoke({"input": inp})
    third_stage_response = structured_llm.invoke(prompt)

    return third_stage_response.model_dump()['inner'], third_stage_response.model_dump()['outer']

def determine_scene(inner_num: int, outer_num: int):
    scene_list = ['乾', '兑', '離', '震', '巽', '坎', '艮', '坤']
    gossip_64 = [
        # 乾
        ['乾為天', '天澤履', '天火同人', '天雷无妄', '天風姤', '天水訟', '天山遯', '天地否'],
        # 兌
        ['澤天夬', '兌為澤', '澤火革', '澤雷隨', '澤風大過', '澤水困', '澤山咸', '澤地萃'],
        # 離
        ['火天大有', '火澤睽', '離為火', '火雷噬嗑', '火風鼎', '火水未濟', '火山旅', '火地晉'],
        # 震
        ['雷天大壯', '雷澤歸妹', '雷火豐', '震為雷', '雷風恆', '雷水解', '雷山小過', '雷地豫'],
        # 巽
        ['風天小畜', '風澤中孚', '風火家人', '風雷益', '巽為風', '風水渙', '風山漸', '風地觀'],
        # 坎
        ['水天需', '水澤節', '水火既濟', '水雷屯', '水風井', '坎為水', '水山蹇', '水地比'],
        # 艮
        ['山天大畜', '山澤損', '山火賁', '山雷頤', '山風蠱', '山水蒙', '艮為山', '山地剝'],
        # 坤
        ['地天泰', '地澤臨', '地火明夷', '地雷復', '地風升', '地水師', '地山謙', '坤為地'],
    ]
    return scene_list[inner_num - 1], scene_list[outer_num - 1], gossip_64[outer_num - 1][inner_num - 1]

def determine_linkage(person: str, first_stage_response: str, scene_num: int):
    scene_list = ['乾', '兑', '離', '震', '巽', '坎', '艮', '坤']
    first_stage_response = first_stage_response.replace("你", "我").replace("妳", "我")
    linkage_text = read_file_as_string("linkage_prompt.txt")
    linkage_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", linkage_text,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        person: str
        person_description: str
        scene: str
        scene_description: str

    linkage_workflow = StateGraph(state_schema=State) # state instead of message state
    def call_model(state: State): # is called whenever the node is invoked
        print("call_model")
        prompt = linkage_prompt.invoke(state) # Create the prompt from the template and state
        print("Prompt:", prompt)
        response = model.invoke(prompt)
        return {"messages": [response]}
    
    # Define the (single) node in the graph
    linkage_workflow.add_edge(START, "model")
    linkage_workflow.add_node("model", call_model)
    # Add memory
    #linkage_memory = MemorySaver()
    #linkage_app = linkage_workflow.compile(checkpointer=linkage_memory)
    linkage_app = linkage_workflow.compile()
    config = {"configurable": {"thread_id": "trial_3"}}

    scene_text = read_file_as_string(f"eight_gua/{scene_num}.txt")
    print("Scene Text:", scene_text)

    input_messages = [HumanMessage(first_stage_response)]
    output = linkage_app.invoke(
        {"messages": input_messages,
        "person": person,
        "person_description": first_stage_response,
        "scene": scene_list[scene_num - 1],
        "scene_description": scene_text
        }, # based on state
        config,
    )
    return output["messages"][-1].content

def determine_deduction(first_stage_response: str, inner_person_description: str, outer_person_description: str, question: str):
    first_stage_response = first_stage_response.replace("你", "我").replace("妳", "我")
    deduction_text = read_file_as_string("deduction_prompt.txt")
    deduction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", deduction_text,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        description: str
        inner_person_description: str
        outer_person_description: str
        question: str
    deduction_workflow = StateGraph(state_schema=State) # state instead of message state
    def call_model(state: State): # is called whenever the node is invoked
        print("call_model")
        prompt = deduction_prompt.invoke(state) # Create the prompt from the template and state
        print("Prompt:", prompt)
        response = model.invoke(prompt)
        return {"messages": [response]}
    deduction_workflow.add_edge(START, "model")
    deduction_workflow.add_node("model", call_model)
    deduction_app = deduction_workflow.compile()
    config = {"configurable": {"thread_id": "trial_3"}}
    output = deduction_app.invoke(
        {
            "description": first_stage_response,
            "inner_person_description": inner_person_description,
            "outer_person_description": outer_person_description,
            "question": question
        }, 
        config,
    )
    print("Output:", output)
    return output["messages"][-1].content