from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PlantUpload(Base):
    __tablename__ = "plant_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    location = Column(String)
    plant_type = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)