from pydantic import BaseModel


class BasePayload(BaseModel):
    class Config:
        frozen = True

    def todict(self):
        return model.model_dump(exclude_none=True)
