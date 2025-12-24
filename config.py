import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_HOST = os.getenv("DB_HOST")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT", "5432")

    @property
    def DB_CONFIG(self):
        return {
            "dbname": self.DB_NAME,
            "user": self.DB_USER,
            "host": self.DB_HOST,
            "password": self.DB_PASSWORD,
            "port": self.DB_PORT
        }

settings = Settings()