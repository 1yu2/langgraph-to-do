from typing_extensions import TypedDict
from pydantic import BaseModel,field_validator
# 使用 Pydantic 定义状态，并进行数据验证
from pydantic import BaseModel,field_validator
# 使用 TypedDict 定义状态
class TypedDictState(TypedDict):
    user_input: str
    agent_response: str
    tool_output: str

class PydanticState(BaseModel):
    user_input: str
    agent_response: str
    tool_output: str
    mood: str = "neutral" # 默认情绪状态为 neutral
    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        if value not in ["happy", "sad", "neutral"]:
            raise ValueError("情绪状态必须是 'happy', 'sad' 或'neutral'")
        return value