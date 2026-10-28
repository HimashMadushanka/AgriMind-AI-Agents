from pydantic import BaseModel, Field

class FarmerQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

class AssistantResponse(BaseModel):
    question: str
    advisory: str
    result: dict
