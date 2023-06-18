from datetime import datetime
from typing import Any, Optional, Dict, Literal

from sqlalchemy import insert, update
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, JSON, ForeignKey, Integer, DateTime, Float, ARRAY, VARCHAR

import polyai
import pylogg
log = pylogg.New('db')

# declare our own base class that all of the modules in orm can import
class ORMBase(DeclarativeBase):
    def serialize(self):
        """ Serialize the data """
        res = {}
        for attr in vars(self):
            if attr.startswith("_"):
                continue
            res[attr] = self.__getattribute__(attr)
        return res

    def get_one(self, session, criteria = {}) -> 'ORMBase':
        """ Get the first element from current table using a criteria ."""
        return session.query(self.__class__).filter_by(**criteria).first()

    def get_all(self, session, criteria = {}) -> list['ORMBase']:
        """ Get all the elements from current table using a criteria ."""
        return session.query(self.__class__).filter_by(**criteria).all()

    def insert(self, session, *, test=False):
        payload = self.serialize()
        try:
            session.execute(insert(self.__class__), payload)
        except Exception as err:
            log.error("Insert ({}) - {}", self.__tablename__, err)
        if test:
            session.rollback()
            log.trace("Insert ({}) - rollback", self.__tablename__)
        else:
            session.commit()

    def update(self, session, newObj, *, test=False):
        values = newObj.serialize()
        try:
            sql = update(self.__class__).where(self.id == newObj.id).values(**values)
            self.session.execute(sql)
        except Exception as err:
            log.error("Update ({}) - {}", self.__tablename__, err)
        if test:
            session.rollback()
            log.trace("Update ({}) - rollback", self.__tablename__)
        else:
            session.commit()

    def upsert(self, session, which: dict, payload, name : str, *,
               update=False, test=False) -> 'ORMBase':
        """
        Update the database by inserting or updating a record.
        Args:
            which dict:     The criteria to check if the record already exists.
            payload:        The object to insert to the table.
            name str:       Name or ID of the object, for logging purposes.
            update bool:    Whether to update the record if already exists.
        """

        table = self.__tablename__

        # select existing record by "which" criteria
        x = self.get_one(session, which)

        # set the foreign keys
        payload.__dict__.update(which)

        if x is None:
            self.insert(session, test=test)
            log.trace(f"{self.__tablename__} add: {name}")
        else:
            if update:
                self.update(session, x, payload, test=test)
                log.trace(f"{self.__tablename__} update: {name}")
            else:
                log.trace(f"{self.__tablename__} ok: {name}")

        return self.get_one(session, which)


class APIRequest(ORMBase):
    """
    Persistence of API interaction.

    Attributes:
        codeversion: version of the current code implementation in date format
        model: model used to query ex. gpt-3.5-turbo (string)
        output: the actual output of the model
        request: the request object sent to the api endpoint (json)
        response: model response
        request_tokens: number of tokens used for request
        response_tokens: number of tokens used for response
    """

    __tablename__ = "api_request"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_added: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    codeversion: Mapped[str] = mapped_column(VARCHAR(length=15))
    apikey: Mapped[str] = mapped_column(VARCHAR(length=32))
    requrl: Mapped[str] = mapped_column(Text)
    reqmethod: Mapped[str] = mapped_column(VARCHAR(length=6))
    model: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    request_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output: Mapped[str] = mapped_column(Text)
    response_tokens: Mapped[int] = mapped_column(Integer, default=0)
    request: Mapped[Dict] = mapped_column(JSON)
    response: Mapped[Dict] = mapped_column(JSON)

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        if self.date_added is None:
            self.date_added = datetime.now()
        if self.codeversion is None:
            self.codeversion = polyai.__version__


if __name__ == "__main__":
    # Run python polyai/server/orm.py to create the table(s).

    import dotenv
    from polyai.server import database

    if not dotenv.load_dotenv():
        raise RuntimeError("ENV not loaded")
    
    en = database.engine()

    # Create all tables if not already created
    ORMBase.metadata.create_all(en)
    print("Tables Created. Done!")
