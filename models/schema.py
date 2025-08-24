from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, Union, List
from pydantic import Field


class Transaction(BaseModel):
   # Transaction_ID: int
    #Date: str = ""
    Description: str = ""
    #Amount: float = 0.0
    Money_In: float = 0.0
    Money_Out: float = 0.0
    #TaxCode: str = ""



class PredictionRequest(BaseModel):

    description: str
    money_in: float = 0.0
    money_out: float = 0.0

class PredictionResponse(BaseModel):
    recommend_category: str


class TransactionInput(BaseModel):
    date: str
    supplier_name: str
    money_in: float
    money_out: float