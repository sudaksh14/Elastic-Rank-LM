# Python 3.10 compat for FlexRank core (written for >=3.11): provide enum.StrEnum if missing.
import enum
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __new__(cls, *a):
            o = str.__new__(cls, *a); o._value_ = str(*a); return o
        def __str__(self): return str(self.value)
        @staticmethod
        def _generate_next_value_(name, start, count, last_values): return name.lower()
    enum.StrEnum = StrEnum
