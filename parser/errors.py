class DSLCompileError(Exception):
    def __init__(self, message: str, span=None):
        super().__init__(message)
        self.span = span


class ParseError(DSLCompileError):
    pass


class LoweringError(DSLCompileError):
    pass
